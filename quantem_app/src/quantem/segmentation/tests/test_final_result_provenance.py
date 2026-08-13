from __future__ import annotations

import numpy as np
from django.test import TestCase
from rest_framework.test import APIClient

from quantem import __version__
from quantem.jobs.constants import (
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
)
from quantem.jobs.models import Job
from quantem.seg_core.db.prob_maps import delete_probability_maps_for_segmentation
from quantem.segmentation.final_result import persist_final_result_provenance
from quantem.segmentation.global_masks import save_global_mask
from quantem.segmentation.models import ImageSegmentation, ProbabilityMap
from quantem.segmentation.type_service import get_or_create_er_type
from quantem.testing import create_small_test_image, write_prob_map_png


class FinalResultProvenanceTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        image = create_small_test_image("Final provenance", width=32, height=32)
        self.segmentation = ImageSegmentation.objects.create(
            asset=image.asset,
            segmentation_type=get_or_create_er_type(),
            include_level=0.42,
            status_stage="CANDIDATES_READY",
        )
        self.probability_map = write_prob_map_png(
            self.segmentation,
            np.full((32, 32), 0.7, dtype=np.float32),
            name="ER_DINO",
        )
        self.probability_map.metadata = {
            "pack_id": "quantem:er",
            "threshold": 0.5,
            "adapter_id": "adapter-123",
        }
        self.probability_map.save(update_fields=["metadata", "updated_at"])

    def test_completion_persists_visible_note_before_silent_map_cleanup(self):
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/complete",
            {},
            format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        note = response.data["final_result_provenance"]
        self.assertEqual(note["model_identifier"], "quantem:er")
        self.assertEqual(note["quantem_version"], __version__)
        self.assertEqual(note["final_level"], 0.42)
        self.assertEqual(note["final_level_kind"], "include_level")
        self.assertEqual(note["adapter_identifier"], "adapter-123")
        self.assertFalse(ProbabilityMap.objects.filter(id=self.probability_map.id).exists())

    def test_note_is_immutable_across_retries_and_later_metadata_changes(self):
        first = persist_final_result_provenance(self.segmentation)
        ProbabilityMap.objects.filter(id=self.probability_map.id).update(
            metadata={"pack_id": "omniem:er", "threshold": 0.9}
        )
        self.segmentation.include_level = 0.9
        self.segmentation.save(update_fields=["include_level", "updated_at"])
        second = persist_final_result_provenance(self.segmentation)
        self.assertEqual(second, first)

    def test_newer_roi_composite_cannot_erase_pack_or_adapter_identity(self):
        ProbabilityMap.objects.create(
            segmentation=self.segmentation,
            name="ER_DINO",
            file_path="prob_maps/composite.png",
            metadata={"model_type": "DINO", "composite": True},
        )

        note = persist_final_result_provenance(self.segmentation)

        self.assertEqual(note["model_identifier"], "quantem:er")
        self.assertEqual(note["adapter_identifier"], "adapter-123")

    def test_absent_map_is_unknown_not_falsely_manual(self):
        ProbabilityMap.objects.filter(segmentation=self.segmentation).delete()

        note = persist_final_result_provenance(self.segmentation)

        self.assertEqual(note["model_identifier"], "unknown")
        self.assertEqual(note["adapter_identifier"], "unknown")

    def test_a_manual_remove_only_global_mask_is_still_manual_provenance(self):
        ProbabilityMap.objects.filter(segmentation=self.segmentation).delete()
        save_global_mask(
            self.segmentation,
            np.zeros((32, 32), dtype=bool),
            source="manual-remove",
        )

        note = persist_final_result_provenance(self.segmentation)

        self.assertEqual(note["model_identifier"], "manual")
        self.assertEqual(note["adapter_identifier"], "manual")

    def test_unlock_clears_the_note_so_a_later_final_result_is_not_misattributed(self):
        completed = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/complete",
            {},
            format="json",
        )
        self.assertEqual(completed.status_code, 200, completed.data)
        unlocked = self.client.delete(f"/api/segmentations/{self.segmentation.id}/complete")
        self.assertEqual(unlocked.status_code, 200, unlocked.data)
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.final_result_provenance, {})

        replacement = write_prob_map_png(
            self.segmentation,
            np.full((32, 32), 0.8, dtype=np.float32),
            name="ER_OMNIEM",
        )
        replacement.metadata = {
            "pack_id": "omniem:er",
            "threshold": 0.6,
            "adapter_id": None,
        }
        replacement.save(update_fields=["metadata", "updated_at"])
        self.segmentation.include_level = 0.6
        self.segmentation.save(update_fields=["include_level", "updated_at"])

        recompleted = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/complete",
            {},
            format="json",
        )
        self.assertEqual(recompleted.status_code, 200, recompleted.data)
        self.assertEqual(
            recompleted.data["final_result_provenance"]["model_identifier"],
            "omniem:er",
        )
        self.assertEqual(
            recompleted.data["final_result_provenance"]["adapter_identifier"],
            "unknown",
        )

    def test_active_rethreshold_reader_defers_cleanup(self):
        Job.objects.create(
            type=JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
            status="RUNNING",
            payload_json={"segmentation_id": str(self.segmentation.id)},
        )
        self.assertEqual(delete_probability_maps_for_segmentation(self.segmentation), 0)
        self.assertTrue(ProbabilityMap.objects.filter(id=self.probability_map.id).exists())

    def test_completion_refuses_to_freeze_a_result_an_active_job_can_still_change(self):
        job = Job.objects.create(
            type=JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
            status="RUNNING",
            payload_json={"segmentation_id": str(self.segmentation.id)},
        )

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/complete",
            {},
            format="json",
        )

        self.assertEqual(response.status_code, 409, response.data)
        self.assertEqual(response.data["job_id"], str(job.id))
        self.segmentation.refresh_from_db()
        self.assertNotEqual(self.segmentation.status_stage, "COMPLETED")
        self.assertEqual(self.segmentation.final_result_provenance, {})
        self.assertTrue(ProbabilityMap.objects.filter(id=self.probability_map.id).exists())

    def test_dataset_calibration_reader_is_recognised_by_asset_list(self):
        job = Job.objects.create(
            type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            status="PENDING",
            payload_json={"asset_ids": [str(self.segmentation.asset_id)]},
        )
        self.assertEqual(delete_probability_maps_for_segmentation(self.segmentation), 0)
        self.assertTrue(ProbabilityMap.objects.filter(id=self.probability_map.id).exists())

        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/complete",
            {},
            format="json",
        )
        self.assertEqual(response.status_code, 409, response.data)
        self.assertEqual(response.data["job_id"], str(job.id))
