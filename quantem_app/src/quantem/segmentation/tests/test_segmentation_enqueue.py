from django.test import TestCase
from rest_framework.test import APIClient

from quantem.assets.utils import create_roi_image_from_image
from quantem.jobs.constants import JOB_TYPE_RUN_SEGMENTATION_ROI, QUEUE_P3_ROI
from quantem.jobs.models import Job
from quantem.segmentation.type_service import (
    get_or_create_analysis_mask_type,
    get_or_create_mitochondria_type,
    get_or_create_tissue_type,
)
from quantem.testing import create_image_from_test_tiff


class SegmentationCreateEnqueueTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.image = create_image_from_test_tiff("Segmentation Enqueue Test Image")
        self.asset = self.image.asset
        self.seg_type = get_or_create_mitochondria_type()

    def _segmentations_url(self) -> str:
        return f"/api/assets/{self.asset.id}/segmentations/"

    def test_create_segmentation_enqueues_roi_job_with_type(self):
        response = self.client.post(
            self._segmentations_url(),
            {"segmentation_type_id": str(self.seg_type.id)},
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.data["config"]["supports_instance_params"])

        queued = Job.objects.get(type=JOB_TYPE_RUN_SEGMENTATION_ROI)
        self.assertEqual(queued.queue_name, QUEUE_P3_ROI)
        self.assertEqual(
            queued.payload_json["segmentation_type"],
            self.seg_type.internal_name,
        )
        self.assertEqual(
            queued.payload_json["segmentation_id"],
            response.data["id"],
        )
        self.assertIsNone(queued.payload_json.get("roi_id"))

    def test_create_segmentation_uses_active_roi_id_when_available(self):
        roi = create_roi_image_from_image(
            self.image,
            x=0,
            y=0,
            width=min(self.image.width, 128),
            height=min(self.image.height, 128),
            source="AUTO",
        )

        response = self.client.post(
            self._segmentations_url(),
            {"segmentation_type_id": str(self.seg_type.id)},
            format="json",
        )
        self.assertEqual(response.status_code, 201)

        queued = Job.objects.get(type=JOB_TYPE_RUN_SEGMENTATION_ROI)
        self.assertEqual(queued.payload_json.get("roi_id"), str(roi.id))

    def test_manual_organelle_creation_is_ready_without_enqueuing_inference(self):
        response = self.client.post(
            self._segmentations_url(),
            {
                "segmentation_type_name": "Mitochondria",
                "run_inference": False,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["status_stage"], "CANDIDATES_READY")
        self.assertEqual(response.data["status_progress"], 100.0)
        # The config remains available for a later model run from labeling.
        self.assertIsNotNone(response.data["config"])
        self.assertFalse(Job.objects.filter(type=JOB_TYPE_RUN_SEGMENTATION_ROI).exists())

    def test_create_tissue_segmentation_is_manual_only_and_ready_immediately(self):
        tissue_type = get_or_create_tissue_type()

        response = self.client.post(
            self._segmentations_url(),
            {"segmentation_type_name": "Tissue Mask"},
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            response.data["segmentation_type"]["internal_name"],
            tissue_type.internal_name,
        )
        self.assertEqual(response.data["status_stage"], "CANDIDATES_READY")
        self.assertEqual(response.data["status_progress"], 100.0)
        self.assertIsNone(response.data["config"])
        # A tissue mask has no ML model, so it offers only the manual source.
        self.assertEqual(
            [entry["value"] for entry in response.data["source_models"]],
            ["manual"],
        )
        self.assertFalse(Job.objects.filter(type=JOB_TYPE_RUN_SEGMENTATION_ROI).exists())

    def test_analysis_masks_are_named_per_image_and_never_queue_a_model(self):
        response = self.client.post(
            self._segmentations_url(),
            {
                "segmentation_type_name": "Analysis Segmentation Mask",
                "analysis_name": "Tissue mask",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["display_name"], "Tissue mask")
        self.assertEqual(
            response.data["segmentation_type"]["id"],
            str(get_or_create_analysis_mask_type().id),
        )
        self.assertEqual(response.data["status_stage"], "CANDIDATES_READY")
        self.assertIsNone(response.data["config"])
        self.assertEqual([entry["value"] for entry in response.data["source_models"]], ["manual"])
        self.assertFalse(Job.objects.filter(type=JOB_TYPE_RUN_SEGMENTATION_ROI).exists())

        second = self.client.post(
            self._segmentations_url(),
            {
                "segmentation_type_name": "Analysis Segmentation Mask",
                "analysis_name": "Cells mask",
            },
            format="json",
        )
        self.assertEqual(second.status_code, 201)
        self.assertEqual(second.data["display_name"], "Cells mask")
        self.assertNotEqual(second.data["id"], response.data["id"])

    def test_custom_segmentation_is_manual_only_and_reusable(self):
        response = self.client.post(
            self._segmentations_url(),
            {"segmentation_type_name": "Vesicles"},
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["segmentation_type"]["long_name"], "Vesicles")
        self.assertEqual(response.data["segmentation_type"]["kind"], "custom")
        self.assertEqual(response.data["status_stage"], "CANDIDATES_READY")
        self.assertIsNone(response.data["config"])
        self.assertFalse(Job.objects.filter(type=JOB_TYPE_RUN_SEGMENTATION_ROI).exists())

    def test_custom_segmentation_records_its_global_measurement_mode(self):
        response = self.client.post(
            self._segmentations_url(),
            {
                "segmentation_type_name": "ER-like network",
                "measurement_mode": "global",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["segmentation_type"]["measurement_mode"], "global")
