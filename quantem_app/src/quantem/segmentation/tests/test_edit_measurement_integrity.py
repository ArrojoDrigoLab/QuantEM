"""An edit either measures what it made, or says it did not.

Reported: ``POST /segments/remove-area/`` on an image that had been moved onto
an unavailable share returned **200** ``{"created": 1, "updated": 1}``, and both
resulting objects stored ``area = 25600.0`` -- the parent's area -- while their
real polygon areas were 11200 and 9600. ``objects.csv`` then reported both at
2.3x and 2.7x their true size, with ``calibrated=True`` in the same row, and the
queued refresh that might have corrected them is off by default
(``QUANTEM_ENABLE_SEGMENT_FEATURE_REFRESH_TRIGGERS``).

``features/measure.py`` reached that by ``continue``-ing past a failed
measurement, which keeps whatever was there. One file over,
``tasks._apply_prob_map_stats`` had already ruled on the same class of value for
the same event -- *"a value left over from the previous outline describes a
shape that no longer exists; stale is a different kind of wrong from fabricated,
not a lesser one"* -- and deletes the keys. These tests pin the two writers to
that one answer, and pin the API to reporting the failure rather than a plain
200.
"""

from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.analysis.morphometrics import derive
from quantem.segmentation.confidence import segment_confidence_score
from quantem.segmentation.features.measure import (
    MEASUREMENT_KEYS,
    MeasurementOutcome,
    measure_segments,
)
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256

#: The reported numbers: a 160x160 parent cut by a 20px-wide stripe.
PARENT_AREA = 25600.0


def _square_coords(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _square_polygon(x0: int, y0: int, x1: int, y1: int) -> Polygon:
    return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))


class UnmeasurableEditTests(TestCase):
    """The image cannot be read, and the edit happens anyway. What is stored?"""

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Unmeasurable edit", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = _square_polygon(40, 40, 200, 200)
        self.segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            source_model="quantem:mito",
            confidence_score=0.82,
            features={
                "area": PARENT_AREA,
                "perimeter": 640.0,
                "intensity_mean": 128.0,
                "mean_prob": 0.82,
                "mito_generated": True,
            },
        )
        self.base = f"/api/segmentations/{self.segmentation.id}"

    def _cut_with_unreadable_image(self):
        with patch(
            "quantem.segmentation.features.measure.get_asset_openable",
            side_effect=OSError("image is on an unavailable share"),
        ):
            return self.client.post(
                f"{self.base}/segments/remove-area/",
                {"areas": [{"geometry_coords": _square_coords(110, 20, 130, 240)}]},
                format="json",
            )

    def test_neither_half_keeps_the_parents_area(self):
        self._cut_with_unreadable_image()

        pieces = list(SegmentObject.objects.filter(segmentation=self.segmentation))
        self.assertEqual(len(pieces), 2)
        for piece in pieces:
            # The reported symptom: 25600 stored against a polygon of 11200.
            self.assertNotEqual(piece.features.get("area"), PARENT_AREA)
            for key in MEASUREMENT_KEYS:
                self.assertNotIn(
                    key,
                    piece.features,
                    f"{key} survived an edit that could not re-measure it",
                )

    def test_the_edit_is_not_reported_as_a_plain_success(self):
        response = self._cut_with_unreadable_image()

        # 207, not 200: the outlines were rewritten and committed, so this is
        # not an error -- but part of the operation did not happen, and a
        # caller checking for 200 has to be able to tell.
        self.assertEqual(response.status_code, 207, response.data)
        self.assertEqual(response.data["created"], 1)
        self.assertEqual(response.data["updated"], 1)

        measurement = response.data["measurement"]
        self.assertEqual(measurement["measured"], 0)
        self.assertEqual(len(measurement["unmeasured_ids"]), 2)
        self.assertIn("could not be read", measurement["detail"])
        # It says what is now true of the objects, not only that something failed.
        self.assertIn("empty", measurement["detail"])

        expected = set(response.data["created_ids"]) | set(response.data["updated_ids"])
        self.assertEqual(set(measurement["unmeasured_ids"]), expected)

    def test_the_response_is_still_usable_by_a_client_that_checks_ok(self):
        """207 is in the 2xx range on purpose: the edit *did* happen.

        The viewer refreshes its overlay from this body. A 4xx/5xx would leave
        the screen showing the pre-cut outlines over a database that no longer
        holds them.
        """
        response = self._cut_with_unreadable_image()
        self.assertLess(response.status_code, 300)
        self.assertIsNotNone(response.data["overlay"])

    def test_objects_csv_reports_a_blank_rather_than_a_wrong_number(self):
        self._cut_with_unreadable_image()

        for piece in SegmentObject.objects.filter(segmentation=self.segmentation):
            metrics = derive(
                piece.features,
                object_id=str(piece.id),
                pixel_size_nm=self.image.asset.pixel_size_nm,
            )
            row = metrics.as_row()
            self.assertIsNone(row["area_px"])
            self.assertIsNone(row["area_um2"])

    def test_identity_is_not_collateral_damage(self):
        """Clearing measurements must not clear what the object *is*.

        ``analysis.morphometrics._coverage_note`` reads ``source_model`` to
        explain a partly-populated column, and ``SegmentObject.save`` infers it
        from the ``*_generated`` markers. Destroying those turned a lost
        measurement into "a model that produced no probability".
        """
        self._cut_with_unreadable_image()

        for piece in SegmentObject.objects.filter(segmentation=self.segmentation):
            self.assertIs(piece.features["mito_generated"], True)
            self.assertEqual(piece.source_model, "quantem:mito")

    def test_a_successful_edit_still_answers_200_with_no_measurement_block(self):
        response = self.client.post(
            f"{self.base}/segments/remove-area/",
            {"areas": [{"geometry_coords": _square_coords(110, 20, 130, 240)}]},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        self.assertIsNone(response.data["measurement"])
        for piece in SegmentObject.objects.filter(segmentation=self.segmentation):
            self.assertAlmostEqual(piece.features["area"], piece.geometry.area, delta=400)

    def test_the_cleared_objects_are_findable_again_once_the_image_is_back(self):
        """Clearing is what makes the failure recoverable.

        ``jobs.handlers._unmeasured_segment_ids`` finds work by looking for
        objects with no ``area``. A piece that kept the parent's numbers looked
        exactly like a correctly measured one, so nothing would ever have come
        back for it; a piece with none is queued by the next refresh.
        """
        from quantem.jobs.handlers import _unmeasured_segment_ids
        from quantem.segmentation.tasks import compute_segment_features_task

        self._cut_with_unreadable_image()

        pending = _unmeasured_segment_ids(str(self.segmentation.id))
        self.assertEqual(len(pending), 2)

        for segment_id in pending:
            compute_segment_features_task(segment_id)

        for piece in SegmentObject.objects.filter(segmentation=self.segmentation):
            self.assertAlmostEqual(piece.features["area"], piece.geometry.area, delta=400)
        self.assertEqual(_unmeasured_segment_ids(str(self.segmentation.id)), [])

    def test_a_drawn_object_that_cannot_be_measured_says_so_on_create(self):
        with patch(
            "quantem.segmentation.features.measure.get_asset_openable",
            side_effect=OSError("image is on an unavailable share"),
        ):
            response = self.client.post(
                f"{self.base}/segments/",
                {"geometry_coords": _square_coords(40, 40, 100, 100)},
                format="json",
            )
        # 201 stands: the object exists, and the response carries it.
        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(response.data["measurement"]["measured"], 0)
        segment = SegmentObject.objects.get(id=response.data["id"])
        self.assertNotIn("area", segment.features)


class MeasureSegmentsOutcomeTests(TestCase):
    """The rule itself, at the function that both API paths go through."""

    def setUp(self):
        self.image = create_small_test_image(
            "Measure outcome", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _segment(self, polygon: Polygon, **features) -> SegmentObject:
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            confidence_score=0.5,
            features=features,
        )

    def test_measuring_nothing_is_not_a_failure(self):
        outcome = measure_segments(self.segmentation, [])
        self.assertEqual(outcome, MeasurementOutcome())
        self.assertTrue(outcome.ok)

    def test_a_segmentation_with_no_image_clears_rather_than_keeps(self):
        orphan = ImageSegmentation.objects.create(
            asset=None,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        segment = SegmentObject.objects.create(
            segmentation=orphan,
            geometry=_square_polygon(10, 10, 50, 50),
            centroid=_square_polygon(10, 10, 50, 50).centroid,
            bbox=_square_polygon(10, 10, 50, 50).envelope,
            label_state="CONFIRMED",
            features={"area": 999.0},
        )

        outcome = measure_segments(orphan, [segment])

        self.assertFalse(outcome.ok)
        self.assertIn("no image behind it", outcome.reason)
        segment.refresh_from_db()
        self.assertNotIn("area", segment.features)

    def test_a_geometry_edit_drops_the_old_probability_even_when_it_measures(self):
        segment = self._segment(
            _square_polygon(40, 40, 120, 120), mean_prob=0.82, mean_prob_dino=0.79
        )

        outcome = measure_segments(self.segmentation, [segment], geometry_changed=True)

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.updated, 1)
        segment.refresh_from_db()
        self.assertGreater(segment.features["area"], 0.0)
        self.assertNotIn("mean_prob", segment.features)
        self.assertNotIn("mean_prob_dino", segment.features)
        self.assertIsNone(segment.confidence_score)
        self.assertIsNone(segment_confidence_score(segment))

    def test_a_re_measure_of_an_unchanged_outline_keeps_it(self):
        segment = self._segment(_square_polygon(40, 40, 120, 120), mean_prob=0.82)

        measure_segments(self.segmentation, [segment])

        segment.refresh_from_db()
        self.assertEqual(segment.features["mean_prob"], 0.82)
        self.assertEqual(segment.confidence_score, 0.5)


#: A parent measured before the edit. Every one of these keys describes the
#: outline the cut destroys.
PARENT_MEASUREMENTS: dict[str, float] = {
    "area": PARENT_AREA,
    "perimeter": 640.0,
    "eccentricity": 0.1,
    "solidity": 0.99,
    "elongation": 1.0,
    "major_axis_length": 181.0,
    "minor_axis_length": 160.0,
    "feret_diameter_max": 226.0,
    "intensity_mean": 128.0,
    "intensity_p10": 100.0,
    "intensity_p50": 130.0,
    "intensity_p90": 160.0,
}


class PartialMeasurementTests(TestCase):
    """A measurement that came back half-done is not a success.

    The failure branch used to be ``if measurements:``, so only an entirely
    empty dict counted as a failure, and ``merge_measured_features`` only ever
    *added* -- every key the pass did not return survived from the parent. With
    ``measure_polygon`` returning ``{area, perimeter}`` and nothing else, a cut
    answered ``200`` / ``measurement: None`` while both halves reported the
    parent's ``intensity_mean = 128.0``, ``intensity_p50 = 130.0`` and
    ``eccentricity = 0.1`` against outlines the parent never had.

    Half-refreshed is the one state a reader cannot detect: every column in
    ``objects.csv`` is populated, and some of them describe a shape that no
    longer exists. Today's extractor is all-or-nothing off a single mask so this
    is not reachable through it, which is exactly why the rule has to hold at
    the function both feature writers go through rather than at the one caller
    that happens to be safe.
    """

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Partial measurement", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        polygon = _square_polygon(40, 40, 200, 200)
        self.segment = SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state="CONFIRMED",
            source_model="quantem:mito",
            features={**PARENT_MEASUREMENTS, "mito_generated": True},
        )
        self.base = f"/api/segmentations/{self.segmentation.id}"

    def _cut_returning(self, measurements: dict[str, float]):
        with patch(
            "quantem.segmentation.features.measure.measure_polygon",
            return_value=dict(measurements),
        ):
            return self.client.post(
                f"{self.base}/segments/remove-area/",
                {"areas": [{"geometry_coords": _square_coords(110, 20, 130, 240)}]},
                format="json",
            )

    def test_neither_half_keeps_a_column_the_re_measure_did_not_produce(self):
        self._cut_returning({"area": 11200.0, "perimeter": 500.0})

        pieces = list(SegmentObject.objects.filter(segmentation=self.segmentation))
        self.assertEqual(len(pieces), 2)
        for piece in pieces:
            for key in ("intensity_mean", "intensity_p50", "eccentricity"):
                self.assertNotIn(
                    key,
                    piece.features,
                    f"{key} survived from the parent on a half-done re-measure",
                )
            # What the pass did produce is written.
            self.assertEqual(piece.features["area"], 11200.0)
            self.assertEqual(piece.features["perimeter"], 500.0)

    def test_identity_still_survives_a_partial_measurement(self):
        """Clearing is scoped to measurements: it is not a reset."""
        self._cut_returning({"area": 11200.0, "perimeter": 500.0})

        for piece in SegmentObject.objects.filter(segmentation=self.segmentation):
            self.assertIs(piece.features["mito_generated"], True)
            self.assertEqual(piece.source_model, "quantem:mito")

    def test_objects_csv_shows_a_blank_rather_than_the_parents_intensity(self):
        self._cut_returning({"area": 11200.0, "perimeter": 500.0})

        for piece in SegmentObject.objects.filter(segmentation=self.segmentation):
            row = derive(
                piece.features,
                object_id=str(piece.id),
                pixel_size_nm=self.image.asset.pixel_size_nm,
            ).as_row()
            self.assertIsNone(row["intensity_mean"])

    def test_a_measurement_without_an_area_is_a_failure_not_a_success(self):
        """``area`` is the marker every other reader uses for "measured at all".

        ``jobs.handlers._unmeasured_segment_ids`` finds work by its absence, so
        a pass that came back without it counted as a success while leaving the
        object looking never-measured to the refresh meant to come back for it.
        """
        response = self._cut_returning({"perimeter": 500.0, "solidity": 0.9})

        self.assertEqual(response.status_code, 207, response.data)
        measurement = response.data["measurement"]
        self.assertEqual(measurement["measured"], 0)
        self.assertEqual(len(measurement["unmeasured_ids"]), 2)
        for piece in SegmentObject.objects.filter(segmentation=self.segmentation):
            for key in MEASUREMENT_KEYS:
                self.assertNotIn(key, piece.features)

    def test_a_complete_measurement_is_still_a_plain_success(self):
        response = self._cut_returning(PARENT_MEASUREMENTS)

        self.assertEqual(response.status_code, 200, response.data)
        self.assertIsNone(response.data["measurement"])
        for piece in SegmentObject.objects.filter(segmentation=self.segmentation):
            for key in MEASUREMENT_KEYS:
                self.assertIn(key, piece.features)
