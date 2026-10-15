from unittest import mock

from quantem.analysis.creation import create_analysis_run, prepare_analysis
from quantem.analysis.history import experiment_history
from quantem.analysis.job import run_job
from quantem.analysis.models import AnalysisRun
from quantem.analysis.tests.test_analysis_runs import AnalysisRunTestCase
from quantem.finetune.tests.fixtures import FakeCancel, FakeReporter
from quantem.jobs.models import Job
from quantem.library.models import Experiment


class AnalysisCreationTests(AnalysisRunTestCase):
    def test_history_survives_move_and_segmentation_deletion(self):
        experiment = self.asset.experiment
        response = self.client.post(
            f"/api/segmentations/{self.segmentation.id}/analysis/", {}, format="json"
        )
        self.assertEqual(response.status_code, 202, response.data)
        run_id = response.data["analysis_run_id"]
        AnalysisRun.objects.filter(id=run_id).update(status="SUCCESS")
        other = Experiment.objects.create(name="Destination")
        self.asset.experiment = other
        self.asset.save(update_fields=["experiment"])
        self.segmentation.delete()
        original = experiment_history(experiment)
        self.assertEqual(original[0]["id"], run_id)
        self.assertNotEqual(original[0]["structures"], ["Deleted segmentation"])
        self.assertEqual(experiment_history(other), [])

    def test_vector_global_analysis_saves_results_and_settings(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                if empty:
                    self.tissue.segments.all().delete()
                response = self.client.post(
                    f"/api/segmentations/{self.tissue.id}/analysis/", {}, format="json"
                )
                self.assertEqual(response.status_code, 202, response.data)
                job = Job.objects.get(id=response.data["job_id"])
                run_job(job.payload_json, FakeReporter(), FakeCancel())
                run = AnalysisRun.objects.get(id=response.data["analysis_run_id"])
                self.assertEqual(run.status, "SUCCESS")
                self.assertEqual(
                    run.results["segmentation_settings"][0]["global_mask"]["source"],
                    "empty" if empty else "legacy_vectors",
                )

    def test_invalid_global_mask_ids_are_validation_errors(self):
        for ids in ([{}], ["not-a-uuid"], [str(self.segmentation.id)]):
            with self.subTest(ids=ids):
                response = self.client.post(
                    f"/api/segmentations/{self.tissue.id}/analysis/",
                    {"analysis_mask_ids": ids},
                    format="json",
                )
                self.assertEqual(response.status_code, 400, response.data)
        self.assertFalse(AnalysisRun.objects.exists())

    def test_queue_failure_does_not_leave_a_pending_run(self):
        prepared = prepare_analysis(self.segmentation, {})
        with mock.patch(
            "quantem.analysis.creation.enqueue_run", side_effect=RuntimeError("Queue unavailable")
        ):
            with self.assertRaises(RuntimeError):
                create_analysis_run(self.segmentation, prepared)
        self.assertFalse(AnalysisRun.objects.exists())
