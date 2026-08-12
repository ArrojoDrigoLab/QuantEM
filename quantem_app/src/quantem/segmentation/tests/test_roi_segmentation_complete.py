import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient

from quantem.assets.utils import create_roi_image_from_image
from quantem.seg_core.db.prob_maps import get_prob_map_file_path, save_probability_map
from quantem.segmentation.models import ImageSegmentation, ProbabilityMap, RoiSegmentationStatus
from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_mitochondria_type,
)
from quantem.testing import create_small_test_image


class RoiSegmentationCompleteViewTests(TestCase):
    """Per-(ROI, segmentation) 'mark ROI as done' for a specific organelle."""

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("ROI Segmentation Complete Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_er_type(),
        )
        self.roi = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="MANUAL",
        )
        self.roi.segmentations.add(self.segmentation)

    def _url(self, roi_id=None):
        roi_id = roi_id or self.roi.id
        return f"/api/segmentations/{self.segmentation.id}/roi/{roi_id}/complete"

    def test_post_marks_roi_complete_for_segmentation(self):
        response = self.client.post(self._url(), {}, format="json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["id"], str(self.roi.id))
        self.assertTrue(response.data["completed_for_segmentation"])

        status_row = RoiSegmentationStatus.objects.get(
            image_roi=self.roi, segmentation=self.segmentation
        )
        self.assertTrue(status_row.is_complete)
        self.assertIsNotNone(status_row.completed_at)

    def test_marking_an_roi_done_reclaims_only_that_rois_probability_maps(self):
        full_map = save_probability_map(
            segmentation=self.segmentation,
            model_name="DINO",
            prob_data=np.zeros((64, 64), dtype=np.uint8),
            prefix="er",
            generated_flag="er_generated",
        )
        roi_map = save_probability_map(
            segmentation=self.segmentation,
            model_name="DINO",
            prob_data=np.zeros((self.roi.height, self.roi.width), dtype=np.uint8),
            prefix="er",
            generated_flag="er_generated",
            roi_id=str(self.roi.id),
        )
        roi_path = get_prob_map_file_path(
            self.segmentation,
            "DINO",
            "er",
            roi_id=str(self.roi.id),
        )
        self.assertTrue(roi_path.exists())
        self.assertEqual(
            ProbabilityMap.objects.filter(segmentation=self.segmentation).count(),
            3,
        )

        response = self.client.post(self._url(), {}, format="json")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(ProbabilityMap.objects.filter(id=full_map.id).exists())
        self.assertFalse(ProbabilityMap.objects.filter(id=roi_map.id).exists())
        self.assertFalse(roi_path.exists())
        self.assertFalse(
            ProbabilityMap.objects.filter(
                segmentation=self.segmentation,
                metadata__composite=True,
            ).exists()
        )

    def test_delete_reverts_completion(self):
        self.client.post(self._url(), {}, format="json")

        response = self.client.delete(self._url())

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["completed_for_segmentation"])

        status_row = RoiSegmentationStatus.objects.get(
            image_roi=self.roi, segmentation=self.segmentation
        )
        self.assertFalse(status_row.is_complete)
        self.assertIsNone(status_row.completed_at)

    def test_completion_is_scoped_per_segmentation(self):
        other_segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.roi.segmentations.add(other_segmentation)

        self.client.post(self._url(), {}, format="json")

        # The ROI is done for ER but not for the mitochondria segmentation.
        response = self.client.get(f"/api/segmentations/{other_segmentation.id}/roi/")
        self.assertEqual(response.status_code, 200)
        roi_entry = next(r for r in response.data if r["id"] == str(self.roi.id))
        self.assertFalse(roi_entry["completed_for_segmentation"])

    def test_roi_list_reflects_completion_flag(self):
        response = self.client.get(f"/api/segmentations/{self.segmentation.id}/roi/")
        roi_entry = next(r for r in response.data if r["id"] == str(self.roi.id))
        self.assertFalse(roi_entry["completed_for_segmentation"])

        self.client.post(self._url(), {}, format="json")

        response = self.client.get(f"/api/segmentations/{self.segmentation.id}/roi/")
        roi_entry = next(r for r in response.data if r["id"] == str(self.roi.id))
        self.assertTrue(roi_entry["completed_for_segmentation"])

    def test_unknown_roi_returns_404(self):
        import uuid

        response = self.client.post(self._url(roi_id=uuid.uuid4()), {}, format="json")
        self.assertEqual(response.status_code, 404)
