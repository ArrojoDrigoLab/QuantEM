import uuid

from django.test import TestCase
from rest_framework.test import APIClient

from quantem.assets.models import ImageROI
from quantem.assets.roi_state import activate_roi
from quantem.assets.utils import create_roi_image_from_image
from quantem.segmentation.models import ImageSegmentation, RoiSegmentationStatus
from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_mitochondria_type,
)
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


class SegmentationRoiCompletionLockTests(TestCase):
    """A segmentation marked done refuses the ROI writes addressed to it.

    ``POST .../roi/`` shipped open. A segmentation marked ``COMPLETED`` still
    accepted a new ROI: the row was created, associated with the finished
    segmentation and made active, so a locked image gained a ROI window that
    every subsequent operation on it -- "mark ROI done", ``rerun-roi`` -- then
    refused. State written past a write lock, and a to-do that can never be
    finished. Deleting a ROI was open too, and that one cascades away the
    ``RoiSegmentationStatus`` rows recording which windows were exhaustively
    labeled: the exact rows "mark ROI done" refuses to touch once locked.

    Activation stays open on purpose and is asserted below, because it is how
    the viewer is pointed at a ROI and a locked segmentation stays browsable.
    """

    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("ROI completion lock")
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.base = f"/api/segmentations/{self.segmentation.id}"
        self.roi = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="MANUAL",
        )
        self.roi.segmentations.add(self.segmentation)
        activate_roi(self.roi)
        self.status_row = RoiSegmentationStatus.objects.create(
            image_roi=self.roi,
            segmentation=self.segmentation,
            is_complete=True,
        )

    def _mark_done(self):
        response = self.client.post(f"{self.base}/complete", {}, format="json")
        self.assertEqual(response.status_code, 200, response.data)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "COMPLETED")

    def _writes(self):
        """Every ROI request through this segmentation that changes something."""
        return [
            (
                "place a ROI",
                "post",
                f"{self.base}/roi/",
                {"x": 20, "y": 24, "width": 64, "height": 64, "source": "MANUAL"},
            ),
            (
                "delete a ROI",
                "delete",
                f"{self.base}/roi/{self.roi.id}/",
                None,
            ),
            (
                "mark the active ROI done",
                "post",
                f"{self.base}/roi/complete",
                {},
            ),
            (
                "mark this ROI done for this organelle",
                "post",
                f"{self.base}/roi/{self.roi.id}/complete",
                {},
            ),
        ]

    def test_every_roi_write_is_refused_while_the_segmentation_is_done(self):
        self._mark_done()
        for name, method, url, body in self._writes():
            with self.subTest(action=name):
                response = getattr(self.client, method)(url, body, format="json")
                self.assertEqual(
                    response.status_code,
                    409,
                    f"{name} was accepted on a segmentation marked done",
                )
                self.assertTrue(response.data.get("locked"))
                self.assertIn("marked done", response.data["detail"])
                self.assertIn("Unlock", response.data["detail"])
                self.assertEqual(response.data["unlock"]["method"], "DELETE")

    def test_the_refusal_is_written_in_the_users_vocabulary(self):
        """No verb, no route, no raw id in the sentence -- only in the fields."""
        self._mark_done()
        response = self.client.post(
            f"{self.base}/roi/",
            {"x": 20, "y": 24, "width": 64, "height": 64, "source": "MANUAL"},
            format="json",
        )
        self.assertEqual(response.status_code, 409, response.data)
        detail = response.data["detail"]
        for forbidden in ("POST", "DELETE", "PATCH", "/api/", "endpoint"):
            self.assertNotIn(forbidden, detail)
        self.assertNotIn(str(self.segmentation.id), detail)
        # The machine-readable half is still there, in fields.
        self.assertEqual(response.data["segmentation_id"], str(self.segmentation.id))
        self.assertEqual(response.data["unlock"]["method"], "DELETE")

    def test_nothing_was_written_by_the_refused_roi_requests(self):
        self._mark_done()
        for _name, method, url, body in self._writes():
            getattr(self.client, method)(url, body, format="json")

        self.assertEqual(ImageROI.objects.filter(asset=self.image.asset).count(), 1)
        self.roi.refresh_from_db()
        self.assertFalse(self.roi.is_complete)
        self.assertEqual(
            list(self.roi.segmentations.values_list("id", flat=True)),
            [self.segmentation.id],
        )
        self.status_row.refresh_from_db()
        self.assertTrue(self.status_row.is_complete)

    def test_reads_and_switching_the_displayed_roi_still_work(self):
        """Locking freezes the result, not the screen."""
        self._mark_done()
        other = create_roi_image_from_image(
            self.image,
            x=32,
            y=32,
            width=96,
            height=96,
            source="MANUAL",
            is_active=False,
        )
        other.segmentations.add(self.segmentation)

        self.assertEqual(self.client.get(f"{self.base}/roi/").status_code, 200)
        self.assertEqual(self.client.get(f"{self.base}/roi/segments").status_code, 200)

        activate = self.client.post(
            f"{self.base}/roi/activate/",
            {"roi_id": str(other.id)},
            format="json",
        )
        self.assertEqual(activate.status_code, 200, activate.data)
        other.refresh_from_db()
        self.assertTrue(other.is_active)

    def test_unlocking_lets_the_roi_writes_through_again(self):
        self._mark_done()
        unlock = self.client.delete(f"{self.base}/complete")
        self.assertEqual(unlock.status_code, 200, unlock.data)

        response = self.client.post(
            f"{self.base}/roi/",
            {"x": 20, "y": 24, "width": 64, "height": 64, "source": "MANUAL"},
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(ImageROI.objects.filter(asset=self.image.asset).count(), 2)

    def test_another_organelle_on_the_same_image_is_not_frozen(self):
        """The rectangle is shared; locking one organelle must not freeze it."""
        other_segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_er_type(),
        )
        self._mark_done()

        response = self.client.post(
            f"/api/segmentations/{other_segmentation.id}/roi/",
            {"x": 20, "y": 24, "width": 64, "height": 64, "source": "MANUAL"},
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(ImageROI.objects.filter(asset=self.image.asset).count(), 2)
