from django.test import TestCase
from rest_framework.test import APIClient
from shapely.geometry import Polygon

from quantem.assets.roi_state import activate_roi
from quantem.assets.utils import create_roi_image_from_image
from quantem.jobs.constants import (
    JOB_TYPE_LABELS,
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
    QUEUE_P3_ROI,
    QUEUE_P4_FULL,
)
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation, SegmentationConfig, SegmentObject
from quantem.segmentation.type_service import (
    get_or_create_er_type,
    get_or_create_mitochondria_type,
)
from quantem.testing import create_image_from_test_tiff, create_small_test_image


class OrganelleApplyFullImageViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Organelle View Test Image")
        self.seg_type = get_or_create_mitochondria_type()
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=self.seg_type,
        )
        SegmentationConfig.objects.create(segmentation=self.segmentation)

    def test_apply_full_image_enqueues_unified_full_job(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/apply-full-image/"
        )

        self.assertEqual(response.status_code, 202)
        queued = Job.objects.get(id=response.data["job_id"])
        self.assertEqual(queued.type, JOB_TYPE_RUN_SEGMENTATION_FULL)
        self.assertEqual(queued.resource_class, "gpu")
        self.assertEqual(queued.queue_name, QUEUE_P4_FULL)
        self.assertEqual(
            queued.payload_json,
            {
                "segmentation_id": str(self.segmentation.id),
                "segmentation_type": self.seg_type.internal_name,
                "asset_id": str(self.segmentation.asset_id),
            },
        )

    def test_apply_full_image_refuses_an_unknown_source_model_up_front(self):
        """Adversarial round 13, finding 4: any string 202'd and died minutes
        later inside the worker ("No segmenter registered for type: ...").
        The refusal happens at the door, naming the ids that would work."""
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/apply-full-image/",
            {"source_model": "quantem_internal_mito"},
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        detail = response.data["detail"]
        self.assertIn("quantem_internal_mito", detail)
        self.assertIn("quantem:mito", detail)
        self.assertIn("omniem:mito", detail)
        self.assertEqual(
            response.data["valid_source_models"], ["quantem:mito", "omniem:mito"]
        )
        self.assertFalse(Job.objects.exists(), "nothing may be queued on a refusal")

    def test_apply_full_image_refuses_a_pack_for_another_organelle(self):
        """quantem:er is a real pack -- for ER. On a mitochondria segmentation
        it used to blow up later as an unhandled ValueError (a 500)."""
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/apply-full-image/",
            {"source_model": "quantem:er"},
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("quantem:mito", response.data["detail"])
        self.assertFalse(Job.objects.exists())

    def test_apply_full_image_still_accepts_the_organelle_packs(self):
        for source_model in ("quantem:mito", "omniem:mito"):
            Job.objects.all().delete()
            response = self.client.post(
                f"/api/segmentations/{self.segmentation.id}/apply-full-image/",
                {"source_model": source_model},
                format="json",
            )
            self.assertEqual(response.status_code, 202, source_model)
            queued = Job.objects.get(id=response.data["job_id"])
            self.assertEqual(queued.payload_json["source_model"], source_model)

    def test_rerun_roi_refuses_an_unknown_source_model_up_front(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/rerun-roi/",
            {"source_model": "not-a-model"},
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("quantem:mito", response.data["detail"])
        self.assertFalse(Job.objects.exists())

    def test_apply_full_image_blocks_when_roi_or_full_job_is_active(self):
        Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            status="PENDING",
            payload_json={
                "segmentation_id": str(self.segmentation.id),
                "segmentation_type": self.seg_type.internal_name,
            },
        )

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/apply-full-image/"
        )
        self.assertEqual(response.status_code, 409)

    def test_the_conflict_names_the_blocking_job_and_how_to_clear_it(self):
        """"A task is already queued or running" on its own is a dead end: the
        user cannot clear what the message will not name.

        What names it changed in wave 0c (finding V12). It used to be the job's
        uuid and ``POST /api/jobs/<id>/cancel/`` -- an identifier that appears
        nowhere on screen and a request a biologist cannot issue. It is now the
        task's name as the Tasks & Queues panel writes it, and that panel, which
        is where the Cancel button actually is. The uuid is still on the payload
        for clients; it is no longer in the sentence.
        """
        blocking = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            status="RUNNING",
            payload_json={
                "segmentation_id": str(self.segmentation.id),
                "segmentation_type": self.seg_type.internal_name,
            },
            queue_name=QUEUE_P4_FULL,
        )

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/apply-full-image/"
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["job_id"], str(blocking.id))
        self.assertEqual(response.data["job_status"], "RUNNING")
        self.assertEqual(response.data["job_type"], JOB_TYPE_RUN_SEGMENTATION_FULL)
        detail = response.data["detail"]
        self.assertIn(JOB_TYPE_LABELS[JOB_TYPE_RUN_SEGMENTATION_FULL], detail)
        self.assertIn("Cancel it in Tasks & Queues", detail)
        self.assertNotIn(str(blocking.id), detail)
        self.assertNotIn("/api/", detail)

    def test_a_running_job_is_named_ahead_of_one_queued_behind_it(self):
        Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            status="PENDING",
            payload_json={"segmentation_id": str(self.segmentation.id)},
            queue_name=QUEUE_P3_ROI,
        )
        running = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            status="RUNNING",
            payload_json={"segmentation_id": str(self.segmentation.id)},
            queue_name=QUEUE_P4_FULL,
        )

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/rerun-roi/"
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["job_id"], str(running.id))


class SegmentationConfigViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Segmentation Config View Test Image")

        self.mito_segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.mito_config = SegmentationConfig.objects.create(
            segmentation=self.mito_segmentation
        )

        self.er_segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_er_type(),
        )
        self.er_config = SegmentationConfig.objects.create(segmentation=self.er_segmentation)

    def test_get_config_for_supported_type_returns_defaults_and_capability_flag(self):
        response = self.client.get(
            f"/api/segmentations/{self.mito_segmentation.id}/config/"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["supports_instance_params"], True)
        self.assertEqual(
            response.data["instance_params"],
            {
                "center_min_distance": 8,
                "center_confidence_threshold": 0.3,
                "segmentation_threshold": 0.5,
                "downsampling_factor": None,
            },
        )

    def test_get_config_for_er_reports_not_supported(self):
        response = self.client.get(
            f"/api/segmentations/{self.er_segmentation.id}/config/"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["supports_instance_params"], False)
        self.assertIsNone(response.data["instance_params"])

    def test_patch_config_updates_supported_instance_params(self):
        response = self.client.patch(
            f"/api/segmentations/{self.mito_segmentation.id}/config/",
            {
                "instance_params": {
                    "center_min_distance": 14,
                    "center_confidence_threshold": 0.42,
                    "segmentation_threshold": 0.61,
                    "downsampling_factor": 2,
                }
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.mito_config.refresh_from_db()
        self.assertEqual(
            self.mito_config.get_instance_params(),
            {
                "center_min_distance": 14,
                "center_confidence_threshold": 0.42,
                "segmentation_threshold": 0.61,
                "downsampling_factor": 2,
            },
        )

    def test_patch_config_rejects_invalid_ranges(self):
        response = self.client.patch(
            f"/api/segmentations/{self.mito_segmentation.id}/config/",
            {"instance_params": {"center_confidence_threshold": 1.2}},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("center_confidence_threshold", response.data)

    def test_patch_config_rejects_er(self):
        response = self.client.patch(
            f"/api/segmentations/{self.er_segmentation.id}/config/",
            {"instance_params": {"center_min_distance": 12}},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.data)


class OrganelleRerunRoiViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("Organelle Rerun ROI Test Image")
        self.seg_type = get_or_create_mitochondria_type()
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=self.seg_type,
        )
        SegmentationConfig.objects.create(segmentation=self.segmentation)

    def test_rerun_roi_enqueues_with_active_roi(self):
        roi = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="AUTO",
        )

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/rerun-roi/",
            {},
            format="json",
        )
        self.assertEqual(response.status_code, 202)
        queued = Job.objects.get(id=response.data["job_id"])
        self.assertEqual(queued.type, JOB_TYPE_RUN_SEGMENTATION_ROI)
        self.assertEqual(queued.queue_name, QUEUE_P3_ROI)
        self.assertEqual(queued.payload_json["roi_id"], str(roi.id))

    def test_rerun_roi_uses_explicit_roi_id(self):
        roi_old = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="AUTO",
        )
        create_roi_image_from_image(
            self.image,
            x=32,
            y=32,
            width=min(self.image.width, 96),
            height=min(self.image.height, 96),
            source="MANUAL",
        )

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/rerun-roi/",
            {"roi_id": str(roi_old.id)},
            format="json",
        )
        self.assertEqual(response.status_code, 202)
        queued = Job.objects.get(id=response.data["job_id"])
        self.assertEqual(queued.payload_json["roi_id"], str(roi_old.id))

    def test_rerun_roi_uses_active_roi_not_latest_created(self):
        roi_old = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="AUTO",
        )
        create_roi_image_from_image(
            self.image,
            x=32,
            y=32,
            width=min(self.image.width, 96),
            height=min(self.image.height, 96),
            source="MANUAL",
        )
        activate_roi(roi_old)

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/rerun-roi/",
            {},
            format="json",
        )

        self.assertEqual(response.status_code, 202)
        queued = Job.objects.get(id=response.data["job_id"])
        self.assertEqual(queued.payload_json["roi_id"], str(roi_old.id))


class SegmentationCompleteViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_small_test_image("Segmentation Complete Test Image")
        self.seg_type = get_or_create_mitochondria_type()
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=self.seg_type,
            status_stage="CANDIDATES_READY",
            status_progress=100.0,
        )

    def _create_segment(
        self,
        *,
        coords: tuple[tuple[int, int], ...],
        label_state: str,
    ) -> SegmentObject:
        polygon = Polygon(coords)
        return SegmentObject.objects.create(
            segmentation=self.segmentation,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
            label_state=label_state,
            confidence_score=0.8,
            features={"sam_score": 0.8},
        )

    def _four_segments(self):
        """One of each label state. Returns the confirmed one."""
        confirmed = self._create_segment(
            coords=((10, 10), (18, 10), (18, 18), (10, 18), (10, 10)),
            label_state="CONFIRMED",
        )
        self._create_segment(
            coords=((20, 20), (28, 20), (28, 28), (20, 28), (20, 20)),
            label_state="CANDIDATE",
        )
        self._create_segment(
            coords=((30, 30), (38, 30), (38, 38), (30, 38), (30, 30)),
            label_state="INFERRED",
        )
        self._create_segment(
            coords=((40, 40), (48, 40), (48, 48), (40, 48), (40, 40)),
            label_state="EXCLUDED",
        )
        return confirmed

    def test_complete_without_an_explicit_discard_keeps_every_object(self):
        """The destructive half is opt-in, and a bare POST is not an opt-in.

        This test used to assert the opposite: a POST with no body deleted every
        non-confirmed object. A user pruned a 32-object run with zero
        confirmations on one click of the greenest button on the screen, and
        recovery was a fresh inference pass.
        """
        self._four_segments()

        response = self.client.post(f"/api/segmentations/{self.segmentation.id}/complete")

        self.assertEqual(response.status_code, 200)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "COMPLETED")
        self.assertEqual(
            SegmentObject.objects.filter(segmentation=self.segmentation).count(), 4
        )
        self.assertEqual(response.data["completion"]["discarded_count"], 0)
        self.assertFalse(response.data["completion"]["discarded_unconfirmed"])

    def test_complete_prunes_non_confirmed_segments_and_queues_full_overlay_rebuild(self):
        confirmed = self._four_segments()

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/complete",
            data={"discard_unconfirmed": True, "acknowledged_discard_count": 3},
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "COMPLETED")
        self.assertEqual(
            set(
                SegmentObject.objects.filter(segmentation=self.segmentation).values_list(
                    "id", flat=True
                )
            ),
            {confirmed.id},
        )
        self.assertEqual(response.data["completion"]["discarded_count"], 3)
        self.assertTrue(response.data["completion"]["restorable"])
        self.assertEqual(
            Job.objects.filter(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY).count(),
            1,
        )
        queued_job = Job.objects.get(type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY)
        self.assertEqual(
            queued_job.payload_json,
            {"segmentation_id": str(self.segmentation.id), "mode": "full"},
        )

    def test_delete_unlocks_completed_segmentation(self):
        self.segmentation.status_stage = "COMPLETED"
        self.segmentation.save(update_fields=["status_stage"])

        response = self.client.delete(f"/api/segmentations/{self.segmentation.id}/complete")

        self.assertEqual(response.status_code, 200)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "CANDIDATES_READY")
        self.assertEqual(response.data["is_complete"], False)
