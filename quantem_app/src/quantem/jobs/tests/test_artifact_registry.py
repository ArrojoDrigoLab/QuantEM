from uuid import uuid4

from django.test import TestCase

from quantem.jobs.artifact_registry import lease_paths_for_job, output_paths_for_job
from quantem.jobs.constants import (
    JOB_TYPE_ENSURE_IMAGE_NGFF,
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
    JOB_TYPE_RUN_ANALYSIS,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
)
from quantem.jobs.models import Job

# TODO(quantem): input-path coverage needs Asset/Rendition fixtures, so only
# the payload-derived output paths are asserted here.


class ArtifactRegistryOutputPathTests(TestCase):
    def test_ngff_job_declares_a_leased_ngff_output(self):
        asset_id = uuid4()
        job = Job.objects.create(
            type=JOB_TYPE_ENSURE_IMAGE_NGFF,
            payload_json={"asset_id": str(asset_id)},
        )

        outputs = output_paths_for_job(job)

        self.assertEqual(outputs[0].relpath, f"data/tmp/ngff/{asset_id}.zarr")
        self.assertTrue(outputs[0].lease_required)

    def test_overlay_job_declares_a_leased_overlay_output(self):
        segmentation_id = uuid4()
        job = Job.objects.create(
            type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
            payload_json={"segmentation_id": str(segmentation_id)},
        )

        outputs = output_paths_for_job(job)

        self.assertEqual(
            outputs[0].relpath,
            f"data/tmp/segmentation_overlays/{segmentation_id}",
        )
        self.assertTrue(outputs[0].lease_required)

    def test_segmentation_job_declares_probability_map_output_dirs(self):
        segmentation_id = uuid4()
        job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            payload_json={"segmentation_id": str(segmentation_id)},
        )

        output_relpaths = {path.relpath for path in output_paths_for_job(job)}

        self.assertIn(f"data/prob_maps/{segmentation_id}", output_relpaths)
        self.assertIn(f"data/tmp/prob_maps/{segmentation_id}", output_relpaths)

    def test_adapter_training_leases_its_adapted_model_dir(self):
        adapter_id = uuid4()
        job = Job.objects.create(
            type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            payload_json={
                "segmentation_id": str(uuid4()),
                "base_model": "quantem:mito",
                "adapter_id": str(adapter_id),
            },
        )

        self.assertEqual(
            [path.relpath for path in lease_paths_for_job(job)],
            [f"models/adapted/{adapter_id}"],
        )

    def test_analysis_run_leases_its_export_dir(self):
        analysis_run_id = uuid4()
        job = Job.objects.create(
            type=JOB_TYPE_RUN_ANALYSIS,
            payload_json={"analysis_run_id": str(analysis_run_id)},
        )

        self.assertEqual(
            [path.relpath for path in lease_paths_for_job(job)],
            [f"exports/{analysis_run_id}"],
        )

    def test_job_without_declared_artifacts_leases_nothing(self):
        job = Job.objects.create(
            type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            payload_json={"segmentation_id": str(uuid4())},
        )

        self.assertEqual(lease_paths_for_job(job), [])
