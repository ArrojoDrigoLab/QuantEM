import uuid

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.assets.models import ImageROI
from quantem.assets.roi_state import activate_roi
from quantem.assets.utils import create_roi_image_from_image
from quantem.segmentation.models import ImageSegmentation, RoiSegmentationStatus
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image


class SegmentationRoiViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("ROI View Test Image")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def test_roi_list_returns_active_first_with_flags(self):
        roi_old = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="AUTO",
            is_complete=True,
        )
        roi_old.segmentations.add(self.segmentation)
        roi_new = create_roi_image_from_image(
            self.image,
            x=32,
            y=32,
            width=min(self.image.width, 96),
            height=min(self.image.height, 96),
            source="MANUAL",
        )
        roi_new.segmentations.add(self.segmentation)
        activate_roi(roi_old)

        response = self.client.get(f"/api/segmentations/{self.segmentation.id}/roi/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data[0]["id"], str(roi_old.id))
        self.assertEqual(response.data[0]["is_active"], True)
        self.assertEqual(response.data[0]["is_complete"], True)
        self.assertEqual(response.data[1]["id"], str(roi_new.id))
        self.assertEqual(response.data[1]["is_active"], False)

    def test_roi_create_manual_activates_new_roi(self):
        existing = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="AUTO",
        )
        existing.segmentations.add(self.segmentation)

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/roi/",
            {
                "x": 40,
                "y": 50,
                "width": 96,
                "height": 112,
                "source": "MANUAL",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        existing.refresh_from_db()
        self.assertEqual(existing.is_active, False)
        self.assertEqual(response.data["is_active"], True)
        self.assertEqual(response.data["is_complete"], False)
        self.assertEqual(response.data["source"], "MANUAL")

    def test_roi_activate_switches_active_without_creating_duplicate(self):
        roi_first = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="AUTO",
        )
        roi_first.segmentations.add(self.segmentation)
        roi_second = create_roi_image_from_image(
            self.image,
            x=20,
            y=24,
            width=min(self.image.width, 96),
            height=min(self.image.height, 96),
            source="MANUAL",
        )
        roi_second.segmentations.add(self.segmentation)

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/roi/activate/",
            {"roi_id": str(roi_first.id)},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        roi_first.refresh_from_db()
        roi_second.refresh_from_db()
        self.assertEqual(roi_first.is_active, True)
        self.assertEqual(roi_second.is_active, False)
        self.assertEqual(response.data["id"], str(roi_first.id))
        self.assertEqual(response.data["is_active"], True)
        self.assertEqual(self.image.asset.rois.count(), 2)

    def test_roi_delete_removes_roi_and_cascades_status(self):
        roi = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="MANUAL",
        )
        roi.segmentations.add(self.segmentation)
        RoiSegmentationStatus.objects.create(
            image_roi=roi,
            segmentation=self.segmentation,
            is_complete=True,
        )

        response = self.client.delete(
            f"/api/segmentations/{self.segmentation.id}/roi/{roi.id}/"
        )

        self.assertEqual(response.status_code, 204)
        self.assertFalse(ImageROI.objects.filter(id=roi.id).exists())
        self.assertFalse(
            RoiSegmentationStatus.objects.filter(image_roi_id=roi.id).exists()
        )

    def test_roi_delete_activates_next_roi_when_active_removed(self):
        roi_old = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="AUTO",
        )
        roi_old.segmentations.add(self.segmentation)
        roi_new = create_roi_image_from_image(
            self.image,
            x=16,
            y=16,
            width=min(self.image.width, 96),
            height=min(self.image.height, 96),
            source="MANUAL",
        )
        roi_new.segmentations.add(self.segmentation)
        # roi_new is the most recently created; activate roi_old then delete it.
        activate_roi(roi_old)

        response = self.client.delete(
            f"/api/segmentations/{self.segmentation.id}/roi/{roi_old.id}/"
        )

        self.assertEqual(response.status_code, 204)
        self.assertFalse(ImageROI.objects.filter(id=roi_old.id).exists())
        roi_new.refresh_from_db()
        self.assertTrue(roi_new.is_active)

    def test_roi_delete_missing_returns_404(self):
        response = self.client.delete(
            f"/api/segmentations/{self.segmentation.id}/roi/{uuid.uuid4()}/"
        )

        self.assertEqual(response.status_code, 404)
