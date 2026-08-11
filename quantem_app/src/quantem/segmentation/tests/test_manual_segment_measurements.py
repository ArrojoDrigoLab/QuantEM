"""A hand-drawn object must arrive measured, like a model-extracted one.

``SegmentCreateView`` used to store ``features = {"sam_score": 1.0}`` and nothing
else. Only ``CONFIRMED`` objects reach ``objects.csv``, and a drawn one is
confirmed the moment it is drawn, so every morphometric column in that file was
blank for it while ``calibrated=True`` sat in the same row — a table that reads
as measured and is not. The polygon is available at create time, so it is
measured there.
"""

from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.analysis.morphometrics import PIXEL_METRIC_KEYS, derive
from quantem.segmentation.features.measure import MEASUREMENT_KEYS
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

SIZE = 256

#: The one metric a drawn object legitimately has no value for. There is no
#: model probability behind a polygon a person traced, and writing 0.0 would put
#: "the model was confident this is background" into a paper's table.
UNMEASURABLE_BY_HAND = {"mean_prob"}


def _square_coords(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class ManualSegmentMeasurementTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Manual measurement", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _create(self, coords, **extra) -> SegmentObject:
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/",
            {"geometry_coords": coords, **extra},
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.data)
        return SegmentObject.objects.get(id=response.data["id"])

    def test_a_drawn_object_carries_the_full_measurement_set(self):
        segment = self._create(_square_coords(40, 40, 100, 100))

        for key in MEASUREMENT_KEYS:
            self.assertIn(key, segment.features, f"{key} missing from a drawn object")

        # A 60x60 square, so the numbers are checkable rather than merely
        # present. ``delta=200`` used to be wide enough to swallow the 61 * 61 =
        # 3721 the boundary-inclusive fill produced; the area of a drawn square
        # is now exact, so it is pinned exactly
        # (quantem.segmentation.tests.test_pixel_area_convention).
        self.assertEqual(segment.features["area"], 60.0 * 60.0)
        self.assertAlmostEqual(segment.features["major_axis_length"], 69.3, delta=3.0)
        self.assertAlmostEqual(segment.features["elongation"], 1.0, delta=0.05)
        self.assertGreater(segment.features["solidity"], 0.95)
        self.assertGreater(segment.features["feret_diameter_max"], 60.0)
        self.assertGreaterEqual(
            segment.features["intensity_p90"], segment.features["intensity_p10"]
        )

    def test_the_vestigial_sam_score_is_gone(self):
        """SAM is not in this product; the 1.0 default was the only thing a
        drawn object used to carry."""
        segment = self._create(_square_coords(40, 40, 100, 100))
        self.assertNotIn("sam_score", segment.features)

    def test_every_objects_csv_column_a_polygon_can_fill_is_filled(self):
        segment = self._create(_square_coords(40, 40, 100, 100))

        metrics = derive(
            segment.features,
            object_id=str(segment.id),
            pixel_size_nm=self.image.asset.pixel_size_nm,
        )
        self.assertTrue(metrics.calibrated)

        blank = {
            key
            for key in PIXEL_METRIC_KEYS
            if key not in UNMEASURABLE_BY_HAND and metrics.values.get(key) is None
        }
        self.assertEqual(blank, set(), f"blank objects.csv columns: {sorted(blank)}")

        row = metrics.as_row()
        self.assertIsNotNone(row["area_um2"])
        self.assertIsNotNone(row["perimeter_um"])
        self.assertIsNotNone(row["feret_max_um"])

    def test_a_geometry_edit_remeasures_instead_of_leaving_the_old_shape(self):
        segment = self._create(_square_coords(40, 40, 160, 160))
        original_area = segment.features["area"]

        # Cut the right half away.
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/remove-area/",
            {"areas": [{"geometry_coords": _square_coords(100, 20, 200, 200)}]},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data["updated"], 1)

        segment.refresh_from_db()
        self.assertLess(segment.features["area"], original_area * 0.6)
        self.assertAlmostEqual(segment.features["area"], 60 * 120, delta=800)
        # The descriptors follow the new outline too, not just the area.
        self.assertGreater(segment.features["elongation"], 1.5)


class ConfirmBatchMeasurementTests(TestCase):
    """Drawn polygons confirmed in a batch are objects too, and reach the same CSV."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Confirm batch measurement", width=SIZE, height=SIZE, textured=True
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def test_confirmed_drawn_polygons_are_measured(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/segments/confirm-batch/",
            {
                "segments": [
                    {"geometry_coords": _square_coords(30, 30, 90, 90)},
                    {"geometry_coords": _square_coords(120, 120, 200, 180)},
                ],
                "manual_creation": True,
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)

        segments = list(
            SegmentObject.objects.filter(segmentation=self.segmentation, label_state="CONFIRMED")
        )
        self.assertEqual(len(segments), 2)
        for segment in segments:
            for key in MEASUREMENT_KEYS:
                self.assertIn(key, segment.features, f"{key} missing after confirm-batch")
            self.assertGreater(segment.features["area"], 0.0)
