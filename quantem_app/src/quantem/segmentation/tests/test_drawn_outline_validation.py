"""A bad outline is refused with a sentence, not with a traceback.

``POST /api/segmentations/<id>/segments/`` built ``Polygon(coords)`` straight
from the request and handed it to ``SegmentObject.objects.create``.
``models.save`` repairs geometry through ``geometry/fields.repair_geometry``,
which raises ``ValueError: SegmentObject.geometry must be a valid Polygon, got
MultiPolygon`` for any ring whose ``make_valid`` splits it. Nothing caught it,
so DRF re-raised and the response was an HTTP 500 with a Django traceback --
for a free-hand lasso that crossed its own path, which is one careless stroke.
Every other bad geometry on the same view came back 400 with an explanation.

The second half is quieter and worse: a polygon at ``(1e12, 1e12)``, or entirely
at negative coordinates, was accepted with 201 and became a **confirmed** object
that covers no pixel of the image, so it reached ``objects.csv`` as a row with
every morphometric column empty.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.segmentation.api_views.segments.shared import (
    parse_drawn_outline,
    segmentation_image_size,
)
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256


def _square(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class DrawnOutlineValidationTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Outline validation", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.url = f"/api/segmentations/{self.segmentation.id}/segments/"

    def _post(self, coords, **extra):
        return self.client.post(self.url, {"geometry_coords": coords, **extra}, format="json")

    def test_a_self_crossing_lasso_is_a_400_naming_what_went_wrong(self):
        # A bowtie: the stroke crosses itself once, and make_valid splits it.
        response = self._post([[10, 10], [100, 100], [100, 10], [10, 100]])

        self.assertEqual(response.status_code, 400, response.data)
        error = response.data["error"]
        self.assertIn("crosses itself", error)
        # It says what to do, not only what is wrong.
        self.assertIn("Redraw", error)
        self.assertFalse(SegmentObject.objects.filter(segmentation=self.segmentation).exists())

    def test_a_self_touching_ring_that_repairs_cleanly_is_still_accepted(self):
        """Only a repair that *splits* the shape is refused.

        shapely calls some rings invalid that GEOS tolerated, and
        ``repair_geometry`` fixes those in place. The view runs the same repair,
        so the view and the model can never disagree about what is storable.
        """
        response = self._post(
            [[10, 10], [110, 10], [110, 110], [60, 110], [60, 60], [60, 110], [10, 110]]
        )

        self.assertEqual(response.status_code, 201, response.data)
        segment = SegmentObject.objects.get(id=response.data["id"])
        self.assertEqual(segment.geometry.geom_type, "Polygon")

    def test_an_outline_outside_the_image_is_refused(self):
        cases = {
            "far beyond the image": _square(1_000_000, 1_000_000, 1_000_100, 1_000_100),
            "entirely negative": _square(-500, -500, -400, -400),
            "1e12": _square(1e12, 1e12, 1e12 + 50, 1e12 + 50),
        }
        for name, coords in cases.items():
            with self.subTest(case=name):
                response = self._post(coords)
                self.assertEqual(response.status_code, 400, response.data)
                self.assertIn("outside the image", response.data["error"])
                # The sentence carries both numbers the user needs to see.
                self.assertIn("256x256", response.data["error"])

        self.assertFalse(SegmentObject.objects.filter(segmentation=self.segmentation).exists())

    def test_an_outline_partly_over_the_edge_is_kept(self):
        """Objects genuinely run off the edge of a field of view."""
        response = self._post(_square(-20, -20, 60, 60))

        self.assertEqual(response.status_code, 201, response.data)
        segment = SegmentObject.objects.get(id=response.data["id"])
        self.assertGreater(segment.features["area"], 0.0)

    def test_non_finite_and_non_numeric_coordinates_are_refused(self):
        for name, coords in (
            ("strings", [["a", "b"], [1, 2], [3, 4]]),
            ("nulls", [[None, 0], [1, 2], [3, 4]]),
            ("too few points", [[10, 10], [20, 20]]),
            ("not pairs", [[10], [20, 20], [30, 30]]),
        ):
            with self.subTest(case=name):
                response = self._post(coords)
                self.assertEqual(response.status_code, 400, response.data)
                self.assertIn("geometry_coords", response.data["error"])

    def test_an_outline_with_no_area_is_refused(self):
        response = self._post([[10, 10], [50, 50], [90, 90]])

        self.assertEqual(response.status_code, 400, response.data)
        self.assertIn("no area", response.data["error"])

    def test_the_size_is_read_off_the_asset_not_by_opening_the_image(self):
        """Validating a request body must not depend on the file being reachable."""
        self.assertEqual(segmentation_image_size(self.segmentation), (SIZE, SIZE))
        self.segmentation.asset.logical_width = None
        self.assertIsNone(segmentation_image_size(self.segmentation))


class ParseDrawnOutlineTests(TestCase):
    """The rule without a database, including the case the view cannot reach."""

    def test_an_unknown_image_size_skips_the_bounds_check(self):
        """An asset whose dimensions were never recorded must not start
        rejecting every outline drawn on it."""
        polygon, error = parse_drawn_outline(_square(1e9, 1e9, 1e9 + 5, 1e9 + 5), image_size=None)
        self.assertEqual(error, "")
        self.assertIsNotNone(polygon)

    def test_a_bowtie_names_the_number_of_pieces(self):
        polygon, error = parse_drawn_outline(
            [[10, 10], [100, 100], [100, 10], [10, 100]], image_size=(256, 256)
        )
        self.assertIsNone(polygon)
        self.assertIn("separates into 2 pieces", error)

    def test_infinite_coordinates_are_named_as_such(self):
        polygon, error = parse_drawn_outline(
            [[0, 0], [float("inf"), 0], [10, 10]], image_size=(256, 256)
        )
        self.assertIsNone(polygon)
        self.assertIn("finite", error)
