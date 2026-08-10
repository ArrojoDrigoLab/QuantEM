"""A self-crossing stroke keeps every area it encloses, and the response says so.

``confirm-batch`` is the only endpoint the drawing tools use --
``useDrawing.handleDrawComplete`` closes the raw freehand path with no
self-intersection check, ``useReviewDrawController`` only simplifies it, and
``shared/api/segmentations/annotations.ts`` posts it here -- so a stroke that
crosses itself arrives on the ordinary path, not a rare one.

``_parse_geometry_polygon`` ended ``polygons.sort(key=area, reverse=True);
return polygons[0]``. Measured on a 256 px image before the fix:

===========================================  ======  ======  ======================
drawn                                        stored  lost    response
===========================================  ======  ======  ======================
figure-of-eight, two 2500 px lobes           2500    2500    ``200 {"created": 1}``
stroke crossing itself twice, 8750 px        2500    6250    ``200 {"created": 1}``
erase stroke over a 5000 px figure-of-eight  n/a     2500    ``200 {"updated": 1}``
===========================================  ======  ======  ======================

Half an object's area vanishing under a plain success, with nothing anywhere for
the user to find. Now every enclosed area is stored as its own object, the
``outlines`` block names any outline that separated, and the erase path
subtracts the union of the whole stroke.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.segmentation.api_views.segments.shared import (
    outline_geometry,
    parse_drawn_outline,
    parse_outline_pieces,
    separated_outlines_payload,
)
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256

#: A figure-of-eight: one stroke, two 2500 px lobes meeting at (50, 50).
FIGURE_OF_EIGHT = [[0, 0], [100, 100], [100, 0], [0, 100]]
FIGURE_OF_EIGHT_AREA = 5000.0

#: A stroke that crosses itself twice: four enclosed areas, 8750 px in total.
CROSSES_TWICE = [
    [0, 0], [100, 100], [100, 0], [0, 100],
    [0, 150], [100, 150], [100, 200], [0, 200],
    [50, 200], [50, 150],
]
CROSSES_TWICE_AREA = 8750.0

#: A square drawn with a flick at the end that crosses the first edge. Three
#: enclosed areas, but two of them are hairs: 50x0.6 px and 0.1 px, both under
#: the "more than 1 pixel in both dimensions" rule an object has to meet.
SQUARE_WITH_A_FLICK = [
    [10, 10], [110, 10], [110, 110], [10, 110],
    [10, 9.4], [60, 9.4], [60, 10.2], [9, 10.2],
]


def _square(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class ConfirmBatchKeepsEveryLobeTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Self-crossing outlines", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.url = f"/api/segmentations/{self.segmentation.id}/segments/confirm-batch/"

    def _confirm(self, coords, **flags):
        return self.client.post(
            self.url,
            {
                "segments": [{"geometry_coords": coords}],
                "manual_creation": True,
                **flags,
            },
            format="json",
        )

    def _stored_area(self) -> float:
        return sum(
            float(segment.geometry.area)
            for segment in SegmentObject.objects.filter(segmentation=self.segmentation)
        )

    def test_a_figure_of_eight_keeps_both_lobes(self):
        response = self._confirm(FIGURE_OF_EIGHT)

        self.assertEqual(response.status_code, 200, response.data)
        # The reported number: 2500 stored against 5000 drawn.
        self.assertAlmostEqual(self._stored_area(), FIGURE_OF_EIGHT_AREA, places=3)
        self.assertEqual(
            SegmentObject.objects.filter(segmentation=self.segmentation).count(), 2
        )
        self.assertEqual(response.data["created"], 2)
        self.assertEqual(len(response.data["confirmed_ids"]), 2)

    def test_a_stroke_that_crosses_twice_keeps_all_four_areas(self):
        response = self._confirm(CROSSES_TWICE)

        self.assertEqual(response.status_code, 200, response.data)
        self.assertAlmostEqual(self._stored_area(), CROSSES_TWICE_AREA, places=3)
        self.assertEqual(response.data["created"], 4)

    def test_the_response_says_the_outline_separated(self):
        """One shape drawn, two objects back, is a surprise worth stating.

        A ``created`` count on its own does not carry it: the caller sent one
        outline and has no baseline that makes ``2`` look wrong or right.
        """
        response = self._confirm(FIGURE_OF_EIGHT)

        outlines = response.data["outlines"]
        self.assertIsNotNone(outlines)
        self.assertEqual(
            outlines["separated"], [{"index": 0, "areas": 2, "kept": 2}]
        )
        self.assertIn("crosses itself", outlines["detail"])
        self.assertIn("2 separate areas", outlines["detail"])
        self.assertIn("its own object", outlines["detail"])

    def test_an_ordinary_outline_adds_no_block_at_all(self):
        response = self._confirm(_square(20, 20, 90, 90))

        self.assertEqual(response.status_code, 200, response.data)
        self.assertIsNone(response.data["outlines"])
        self.assertEqual(response.data["created"], 1)

    def test_a_piece_too_thin_to_store_is_named_rather_than_dropped_in_silence(self):
        """The narrow-bbox rule still applies -- it is now reported.

        ``filter_supported_confirmed_polygons`` refuses anything spanning a
        pixel or less in either dimension, and used to do it without a word.
        """
        response = self._confirm(SQUARE_WITH_A_FLICK)

        self.assertEqual(response.status_code, 200, response.data)
        outlines = response.data["outlines"]
        self.assertEqual(outlines["separated"], [{"index": 0, "areas": 3, "kept": 1}])
        # The count in the response matches what actually landed.
        self.assertEqual(response.data["created"], 1)
        self.assertEqual(
            SegmentObject.objects.filter(segmentation=self.segmentation).count(), 1
        )
        self.assertIn("1 pixel or less", outlines["detail"])
        self.assertIn("could not be stored", outlines["detail"])

    def test_the_er_merge_path_keeps_both_lobes_too(self):
        """ER confirms a drawn area with ``merge_overlaps``; same outline, same rule."""
        response = self._confirm(FIGURE_OF_EIGHT, merge_overlaps=True)

        self.assertEqual(response.status_code, 200, response.data)
        self.assertAlmostEqual(self._stored_area(), FIGURE_OF_EIGHT_AREA, places=3)

    def test_the_index_named_is_the_one_that_crossed(self):
        response = self.client.post(
            self.url,
            {
                "segments": [
                    {"geometry_coords": _square(20, 20, 60, 60)},
                    {"geometry_coords": FIGURE_OF_EIGHT},
                ],
                "manual_creation": True,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(
            response.data["outlines"]["separated"],
            [{"index": 1, "areas": 2, "kept": 2}],
        )
        self.assertIn("segments[1]", response.data["outlines"]["detail"])

    def test_each_lobe_is_measured_as_itself(self):
        """Two objects, two areas -- not one number reported twice."""
        self._confirm(FIGURE_OF_EIGHT)

        segments = list(SegmentObject.objects.filter(segmentation=self.segmentation))
        self.assertEqual(len(segments), 2)
        for segment in segments:
            self.assertAlmostEqual(
                segment.features["area"], segment.geometry.area, delta=120
            )
            self.assertLess(segment.features["area"], FIGURE_OF_EIGHT_AREA)


class RemoveAreaErasesTheWholeStrokeTests(TestCase):
    """An eraser that rubs out half of what it was drawn round, under a 200."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Erase stroke", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        target = Polygon(((0, 0), (SIZE, 0), (SIZE, SIZE), (0, SIZE)))
        SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=target,
            centroid=target.centroid,
            bbox=target.envelope,
            label_state="CONFIRMED",
        )
        self.url = f"/api/segmentations/{self.segmentation.id}/segments/remove-area/"

    def _remaining_area(self) -> float:
        return sum(
            float(segment.geometry.area)
            for segment in SegmentObject.objects.filter(segmentation=self.segmentation)
        )

    def test_both_lobes_of_a_self_crossing_erase_stroke_are_subtracted(self):
        response = self.client.post(
            self.url,
            {"areas": [{"geometry_coords": FIGURE_OF_EIGHT}]},
            format="json",
        )

        self.assertEqual(response.status_code, 200, response.data)
        removed = float(SIZE * SIZE) - self._remaining_area()
        # Was 2500: the lobe that was not the largest stayed in the object, and
        # in objects.csv.
        self.assertAlmostEqual(removed, FIGURE_OF_EIGHT_AREA, places=3)

    def test_an_ordinary_erase_stroke_is_unchanged(self):
        response = self.client.post(
            self.url,
            {"areas": [{"geometry_coords": _square(10, 10, 60, 60)}]},
            format="json",
        )

        self.assertEqual(response.status_code, 200, response.data)
        self.assertAlmostEqual(
            float(SIZE * SIZE) - self._remaining_area(), 2500.0, places=3
        )


class RoutedFromTheSingleSegmentEndpointTests(TestCase):
    """The 400 on ``POST /segments/`` sends people here. It has to be true."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Routing", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.base = f"/api/segmentations/{self.segmentation.id}/segments"

    def test_what_the_refusal_promises_is_what_confirm_batch_does(self):
        refused = self.client.post(
            f"{self.base}/",
            {"geometry_coords": FIGURE_OF_EIGHT},
            format="json",
        )
        self.assertEqual(refused.status_code, 400, refused.data)
        self.assertIn("segments/confirm-batch/", refused.data["error"])
        self.assertIn("2 enclosed areas", refused.data["error"])
        self.assertFalse(
            SegmentObject.objects.filter(segmentation=self.segmentation).exists()
        )

        accepted = self.client.post(
            f"{self.base}/confirm-batch/",
            {
                "segments": [{"geometry_coords": FIGURE_OF_EIGHT}],
                "manual_creation": True,
            },
            format="json",
        )
        self.assertEqual(accepted.status_code, 200, accepted.data)
        stored = sum(
            float(segment.geometry.area)
            for segment in SegmentObject.objects.filter(segmentation=self.segmentation)
        )
        self.assertAlmostEqual(stored, FIGURE_OF_EIGHT_AREA, places=3)


class ParseOutlinePiecesTests(TestCase):
    """The rule itself, without a database."""

    def test_every_lobe_comes_back_largest_first(self):
        pieces = parse_outline_pieces(CROSSES_TWICE)
        self.assertEqual(len(pieces), 4)
        areas = [piece.area for piece in pieces]
        self.assertEqual(areas, sorted(areas, reverse=True))
        self.assertAlmostEqual(sum(areas), CROSSES_TWICE_AREA, places=3)

    def test_an_ordinary_outline_is_one_piece(self):
        pieces = parse_outline_pieces(_square(0, 0, 10, 10))
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0].geom_type, "Polygon")

    def test_coordinates_that_are_not_a_polygon_come_back_empty(self):
        for case in (
            None,
            [],
            [[0, 0], [1, 1]],
            [["a", "b"], [1, 2], [3, 4]],
            [[0, 0], [5, 5], [10, 10]],  # collinear: no area at all
        ):
            with self.subTest(case=case):
                self.assertEqual(parse_outline_pieces(case), [])

    def test_the_geometry_handed_to_the_service_holds_every_lobe(self):
        """A union would be free to weld lobes meeting at a point back together."""
        geometry = outline_geometry(parse_outline_pieces(FIGURE_OF_EIGHT))
        self.assertEqual(geometry.geom_type, "MultiPolygon")
        self.assertEqual(len(geometry.geoms), 2)
        self.assertAlmostEqual(geometry.area, FIGURE_OF_EIGHT_AREA, places=3)

    def test_a_single_lobe_is_handed_over_as_a_plain_polygon(self):
        geometry = outline_geometry(parse_outline_pieces(_square(0, 0, 10, 10)))
        self.assertEqual(geometry.geom_type, "Polygon")

    def test_the_single_segment_endpoint_still_refuses_and_points_here(self):
        polygon, error = parse_drawn_outline(FIGURE_OF_EIGHT, image_size=(SIZE, SIZE))
        self.assertIsNone(polygon)
        self.assertIn("separates into 2 pieces", error)
        self.assertIn("2 enclosed areas as its own object", error)


class SeparatedOutlinesPayloadTests(TestCase):
    def test_nothing_separated_is_no_block(self):
        self.assertIsNone(separated_outlines_payload([]))

    def test_every_piece_stored(self):
        payload = separated_outlines_payload([{"index": 0, "areas": 2, "kept": 2}])
        self.assertIn("All 2 were kept", payload["detail"])

    def test_some_pieces_too_thin(self):
        payload = separated_outlines_payload([{"index": 3, "areas": 4, "kept": 1}])
        self.assertIn("segments[3]", payload["detail"])
        self.assertIn("1 were kept", payload["detail"])
        self.assertIn("3 spanned", payload["detail"])

    def test_no_piece_could_be_stored(self):
        payload = separated_outlines_payload([{"index": 0, "areas": 2, "kept": 0}])
        self.assertIn("None of them could be stored", payload["detail"])

    def test_more_than_one_outline_is_named_one_by_one(self):
        payload = separated_outlines_payload(
            [{"index": 0, "areas": 2, "kept": 2}, {"index": 2, "areas": 3, "kept": 3}]
        )
        self.assertIn("segments[0]", payload["detail"])
        self.assertIn("segments[2]", payload["detail"])
