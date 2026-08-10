from __future__ import annotations

from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.segmentation.models import CompletedROI, ImageSegmentation
from quantem.segmentation.type_service import get_or_create_er_type, get_or_create_mitochondria_type
from quantem.testing import create_small_test_image


def _polygon_coords(polygon: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in polygon.exterior.coords]


class CompletedRoiViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image(
            "Completed ROI Test Image",
            width=256,
            height=256,
        )
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )

    @staticmethod
    def _square(x0: float, y0: float, x1: float, y1: float) -> Polygon:
        return Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)))

    @staticmethod
    def _l_shape() -> Polygon:
        return Polygon(
            (
                (0, 0),
                (12, 0),
                (12, 4),
                (4, 4),
                (4, 12),
                (0, 12),
                (0, 0),
            )
        )

    def _create_completed_roi(
        self,
        polygon: Polygon,
        *,
        segmentation: ImageSegmentation | None = None,
    ) -> CompletedROI:
        return CompletedROI.objects.create(
            segmentation=segmentation or self.segmentation,
            geometry=polygon,
        )

    def test_create_and_list_completed_rois(self):
        polygon = self._square(10, 20, 40, 60)

        create_response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/completed-rois/",
            {"polygon_coords": _polygon_coords(polygon)},
            format="json",
        )

        self.assertEqual(create_response.status_code, 201)
        self.assertEqual(CompletedROI.objects.filter(segmentation=self.segmentation).count(), 1)
        self.assertEqual(
            create_response.data["bbox"],
            {"x0": 10.0, "y0": 20.0, "x1": 40.0, "y1": 60.0},
        )

        list_response = self.client.get(
            f"/api/segmentations/{self.segmentation.id}/completed-rois/"
        )

        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.data), 1)
        self.assertEqual(list_response.data[0]["id"], create_response.data["id"])

    def test_create_skips_bbox_only_overlap_without_geometry_intersection(self):
        existing = self._create_completed_roi(self._l_shape())

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/completed-rois/",
            {
                "polygon_coords": _polygon_coords(self._square(6, 6, 10, 10)),
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(CompletedROI.objects.filter(segmentation=self.segmentation).count(), 2)
        existing.refresh_from_db()
        self.assertTrue(existing.geometry.equals(self._l_shape()))

    def test_create_merges_transitively_with_intersecting_completed_rois(self):
        first = self._create_completed_roi(self._square(10, 10, 20, 20))
        second = self._create_completed_roi(self._square(24, 10, 34, 20))

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/completed-rois/",
            {
                "polygon_coords": _polygon_coords(self._square(18, 10, 26, 20)),
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        remaining = list(
            CompletedROI.objects.filter(segmentation=self.segmentation).order_by("created_at", "id")
        )
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].id, first.id)
        self.assertFalse(CompletedROI.objects.filter(id=second.id).exists())
        self.assertEqual(
            response.data["bbox"],
            {"x0": 10.0, "y0": 10.0, "x1": 34.0, "y1": 20.0},
        )

    def test_create_leaves_corner_touching_completed_rois_separate(self):
        existing = self._create_completed_roi(self._square(10, 10, 20, 20))

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/completed-rois/",
            {
                "polygon_coords": _polygon_coords(self._square(20, 20, 30, 30)),
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(CompletedROI.objects.filter(segmentation=self.segmentation).count(), 2)
        existing.refresh_from_db()
        self.assertTrue(existing.geometry.equals(self._square(10, 10, 20, 20)))

    def test_create_does_not_merge_completed_rois_from_other_segmentations(self):
        other_image = create_small_test_image(
            "Completed ROI Other Image",
            width=256,
            height=256,
        )
        other_segmentation = ImageSegmentation.objects.create(
            asset=other_image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )
        self._create_completed_roi(
            self._square(10, 10, 20, 20),
            segmentation=other_segmentation,
        )

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/completed-rois/",
            {
                "polygon_coords": _polygon_coords(self._square(10, 10, 20, 20)),
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(CompletedROI.objects.filter(segmentation=self.segmentation).count(), 1)
        self.assertEqual(
            CompletedROI.objects.filter(segmentation=other_segmentation).count(),
            1,
        )

    def test_create_rejects_polygon_outside_image_bounds(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/completed-rois/",
            {
                "polygon_coords": _polygon_coords(self._square(250, 250, 300, 300)),
            },
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("image bounds", response.data["error"])

    def test_create_does_not_merge_other_segmentation_type_on_same_image(self):
        other_segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_er_type(),
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )
        self._create_completed_roi(
            self._square(10, 10, 20, 20),
            segmentation=other_segmentation,
        )

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/completed-rois/",
            {
                "polygon_coords": _polygon_coords(self._square(10, 10, 20, 20)),
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(CompletedROI.objects.filter(segmentation=self.segmentation).count(), 1)
        self.assertEqual(
            CompletedROI.objects.filter(segmentation=other_segmentation).count(),
            1,
        )
