from django.test import TestCase
from rest_framework.test import APIClient

from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.type_service import get_or_create_er_type
from quantem.testing import create_image_from_test_tiff


class RemovedSegmentationRoutesTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Removed Routes Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_er_type(),
        )

    def test_removed_task_routes_return_404(self):
        seg_id = self.segmentation.id
        removed_paths = [
            f"/api/segmentations/{seg_id}/er/",
            f"/api/segmentations/{seg_id}/er/inference/",
            f"/api/segmentations/{seg_id}/er/segments/",
            f"/api/segmentations/{seg_id}/er/feedback/",
            f"/api/segmentations/{seg_id}/er/retrain/",
            f"/api/segmentations/{seg_id}/er/apply-global/",
            f"/api/segmentations/{seg_id}/retrain/",
            f"/api/segmentations/{seg_id}/roi-annotations",
            f"/api/segmentations/{seg_id}/pixel-classifier",
        ]

        for path in removed_paths:
            response = self.client.post(path, {}, format="json")
            self.assertEqual(response.status_code, 404, path)

    def test_probability_maps_post_returns_405(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/probability-maps/",
            {},
            format="json",
        )
        self.assertEqual(response.status_code, 405)
