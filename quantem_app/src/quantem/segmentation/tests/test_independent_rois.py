from unittest.mock import Mock, patch

from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from rest_framework.test import APIClient
from shapely.geometry import box

from quantem.assets.models import ImageROI
from quantem.assets.roi_state import activate_roi, get_active_roi_for_segmentation
from quantem.core.config import ROIS_DIR
from quantem.segmentation.models import ImageSegmentation, RoiSegmentationStatus, SegmentObject
from quantem.segmentation.type_service import (
    get_or_create_mitochondria_type,
    get_or_create_nucleus_type,
)
from quantem.testing import create_small_test_image


class IndependentRoiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("Independent organelles", width=256, height=256)
        self.mito = ImageSegmentation.objects.create(
            asset=self.image.asset, segmentation_type=get_or_create_mitochondria_type()
        )
        self.nucleus = ImageSegmentation.objects.create(
            asset=self.image.asset, segmentation_type=get_or_create_nucleus_type()
        )

    def url(self, segmentation, suffix=""):
        return f"/api/segmentations/{segmentation.pk}/roi/{suffix}"

    def rois(self, segmentation):
        response = self.client.get(self.url(segmentation))
        self.assertEqual(response.status_code, 200, response.data)
        return response.data

    def create(self, segmentation, x):
        response = self.client.post(
            self.url(segmentation),
            {"x": x, "y": 24, "width": 48, "height": 64, "source": "MANUAL"},
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.data)
        return response.data

    def test_two_done_mito_rois_and_independent_nucleus_workspace(self):
        default = self.rois(self.mito)[0]
        self.client.delete(self.url(self.mito, f"{default['id']}/"))
        first, second = self.create(self.mito, 16), self.create(self.mito, 100)
        for roi in (first, second):
            response = self.client.post(self.url(self.mito, f"{roi['id']}/complete"))
            self.assertEqual(response.status_code, 200)
        before = self.rois(self.mito)
        self.assertEqual(len(before), 2)
        self.assertTrue(all(roi["completed_for_segmentation"] for roi in before))

        nucleus_default = self.rois(self.nucleus)
        self.assertEqual(len(nucleus_default), 1)
        self.assertNotIn(nucleus_default[0]["id"], {first["id"], second["id"]})
        self.assertFalse(nucleus_default[0]["completed_for_segmentation"])
        self.client.delete(self.url(self.nucleus, f"{nucleus_default[0]['id']}/"))
        self.assertEqual(self.rois(self.nucleus), [])
        self.assertEqual(self.rois(self.nucleus), [])  # survives a subsequent load
        own = self.create(self.nucleus, 180)
        self.assertEqual([roi["id"] for roi in self.rois(self.nucleus)], [own["id"]])
        self.assertEqual(self.rois(self.mito), before)

    def test_unassigned_template_does_not_override_an_organelle_workspace(self):
        first = self.create(self.mito, 16)
        template = ImageROI.objects.create(asset=self.image.asset, x=0, y=0, width=64, height=64)
        activate_roi(template)
        self.assertEqual([roi["id"] for roi in self.rois(self.mito)], [first["id"]])

    def test_creating_an_organelle_through_api_does_not_adopt_existing_labeling_rois(self):
        roi = self.create(self.mito, 16)
        self.nucleus.delete()
        response = self.client.post(
            f"/api/assets/{self.image.asset.pk}/segmentations/",
            {"segmentation_type_name": "Nucleus", "run_inference": False},
            format="json",
        )
        self.assertEqual(response.status_code, 201, response.data)
        nucleus = ImageSegmentation.objects.get(pk=response.data["id"])
        defaults = self.rois(nucleus)
        self.assertEqual(len(defaults), 1)
        self.assertNotEqual(defaults[0]["id"], roi["id"])

    def test_inference_worker_rejects_another_organelles_roi(self):
        from quantem.segmentation.organelle_tasks import run_segmentation_roi_task

        roi = self.create(self.mito, 16)
        with (
            patch("quantem.segmentation.organelle_tasks.get_segmenter"),
            self.assertRaises(ValueError),
        ):
            run_segmentation_roi_task(
                segmentation_id=str(self.nucleus.pk),
                segmentation_type=self.nucleus.segmentation_type.internal_name,
                roi_id=roi["id"],
            )

    def test_foreign_roi_cannot_be_activated_deleted_completed_or_run(self):
        roi = self.create(self.mito, 16)
        self.rois(self.nucleus)
        for method, url, payload in [
            ("post", self.url(self.nucleus, "activate/"), {"roi_id": roi["id"]}),
            ("delete", self.url(self.nucleus, f"{roi['id']}/"), {}),
            ("post", self.url(self.nucleus, f"{roi['id']}/complete"), {}),
            ("delete", self.url(self.nucleus, f"{roi['id']}/complete"), {}),
            ("post", f"/api/segmentations/{self.nucleus.pk}/rerun-roi/", {"roi_id": roi["id"]}),
        ]:
            response = getattr(self.client, method)(url, payload, format="json")
            self.assertIn(response.status_code, (400, 404), (url, response.data))
        self.assertEqual(self.rois(self.mito)[0]["id"], roi["id"])
        self.assertFalse(
            RoiSegmentationStatus.objects.filter(
                segmentation=self.nucleus, image_roi_id=roi["id"]
            ).exists()
        )

    def shared_roi(self):
        roi = ImageROI.objects.get(pk=self.create(self.mito, 16)["id"])
        roi.segmentations.add(self.nucleus)
        activate_roi(roi, self.nucleus)
        for segmentation in (self.mito, self.nucleus):
            state = RoiSegmentationStatus.objects.get(image_roi=roi, segmentation=segmentation)
            state.set_complete(True)
            state.save()
        return roi

    def test_removing_migrated_shared_roi_preserves_other_organelle_and_annotations(self):
        roi = self.shared_roi()
        polygon = box(20, 30, 32, 42)
        obj = SegmentObject.objects.create(
            segmentation=self.mito,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon,
            label_state="CONFIRMED",
            source_model="manual",
        )
        crop = ROIS_DIR / f"{roi.pk}.png"
        original = crop.read_bytes()
        before = self.rois(self.mito)
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.delete(self.url(self.nucleus, f"{roi.pk}/"))
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.rois(self.nucleus), [])
        self.assertEqual(self.rois(self.mito), before)
        self.assertEqual(crop.read_bytes(), original)
        self.assertTrue(SegmentObject.objects.filter(pk=obj.pk).exists())
        self.assertTrue(
            RoiSegmentationStatus.objects.get(image_roi=roi, segmentation=self.mito).is_complete
        )

    def test_replacing_shared_roi_and_selection_are_independent(self):
        shared = self.shared_roi()
        other = self.create(self.mito, 110)
        self.assertEqual(get_active_roi_for_segmentation(self.nucleus).pk, shared.pk)
        replacement = self.create(self.nucleus, 180)
        self.client.delete(self.url(self.nucleus, f"{shared.pk}/"))
        self.assertEqual(
            get_active_roi_for_segmentation(self.mito).pk, ImageROI.objects.get(pk=other["id"]).pk
        )
        self.assertEqual({roi["id"] for roi in self.rois(self.mito)}, {str(shared.pk), other["id"]})
        self.assertEqual([roi["id"] for roi in self.rois(self.nucleus)], [replacement["id"]])
        shared.refresh_from_db()
        self.assertEqual(shared.x, 16)

    def test_legacy_complete_endpoint_only_completes_selected_organelle(self):
        shared = self.shared_roi()
        RoiSegmentationStatus.objects.filter(image_roi=shared).update(is_complete=False)
        response = self.client.post(self.url(self.mito, "complete"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.rois(self.mito)[0]["is_complete"])
        self.assertFalse(self.rois(self.nucleus)[0]["is_complete"])

    def test_late_default_creation_does_not_recreate_deleted_workspace(self):
        from quantem.segmentation.api_views.shared import get_or_create_roi_image

        stale = ImageSegmentation.objects.get(pk=self.mito.pk)
        roi = self.rois(self.mito)[0]
        self.client.delete(self.url(self.mito, f"{roi['id']}/"))
        with patch("quantem.assets.utils._save_roi_png_from_ngff", return_value=True):
            self.assertIsNone(get_or_create_roi_image(self.image, stale))
        self.assertEqual(self.rois(self.mito), [])

    def test_roi_list_query_count_does_not_grow_with_roi_count(self):
        self.create(self.mito, 16)
        with CaptureQueriesContext(connection) as single:
            self.rois(self.mito)
        for x in (70, 130, 190):
            self.create(self.mito, x)
        with CaptureQueriesContext(connection) as multiple:
            self.rois(self.mito)
        self.assertEqual(len(multiple), len(single))

    def test_delayed_upload_preserves_existing_and_deleted_workspaces(self):
        from quantem.jobs.handlers.upload import handle_upload_image_pipeline

        own = self.create(self.mito, 16)
        deleted = self.rois(self.nucleus)[0]
        self.client.delete(self.url(self.nucleus, f"{deleted['id']}/"))
        template = ImageROI.objects.create(
            asset=self.image.asset, x=0, y=0, width=128, height=128, is_active=True
        )
        with (
            patch("quantem.assets.tasks.prepare_asset_renditions_task"),
            patch("quantem.assets.tasks.ensure_roi_for_asset_task", return_value=template),
            patch("quantem.jobs.handlers.upload.Job.enqueue") as enqueue,
        ):
            handle_upload_image_pipeline(
                {
                    "asset_id": str(self.image.asset.pk),
                    "segment_mito": True,
                    "segment_nucleus": True,
                },
                Mock(),
                Mock(),
            )
        self.assertEqual([roi["id"] for roi in self.rois(self.mito)], [own["id"]])
        self.assertEqual(self.rois(self.nucleus), [])
        enqueue.assert_called_once()
        self.assertEqual(enqueue.call_args.kwargs["payload"]["roi_id"], own["id"])

    def test_revision_detached_labeling_roi_is_not_adopted_as_a_new_default(self):
        from io import BytesIO
        from zipfile import ZipFile

        from quantem.assets.roi_state import initialize_roi_from_template
        from quantem.segmentation.revision_rois import restore_roi_reviews, save_roi_reviews

        template = ImageROI.objects.create(
            asset=self.image.asset, x=5, y=6, width=40, height=40, is_active=True
        )
        with ZipFile(BytesIO(), "w") as archive:
            save_roi_reviews(archive, self.mito)  # empty workspace
            initialize_roi_from_template(self.mito, template)
            restore_roi_reviews(archive, self.mito)
        self.nucleus.delete()
        response = self.client.post(
            f"/api/assets/{self.image.asset.pk}/segmentations/",
            {"segmentation_type_name": "Nucleus", "run_inference": False},
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        nucleus = ImageSegmentation.objects.get(pk=response.data["id"])
        self.assertNotEqual(self.rois(nucleus)[0]["id"], str(template.pk))
