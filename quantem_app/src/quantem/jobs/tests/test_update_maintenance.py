from django.test import TestCase
from rest_framework.test import APIClient

from quantem.jobs.constants import JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY, QUEUE_P1_INTERACTIVE
from quantem.jobs.models import Job, UpdateMaintenance
from quantem.jobs.update_maintenance import UpdateApplyInProgress, clear_stale_update_apply_lock


class UpdateMaintenanceTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _job(self, status="PENDING"):
        return Job.objects.create(
            type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
            status=status,
            payload_json={},
            queue_name=QUEUE_P1_INTERACTIVE,
        )

    def test_apply_lock_waits_for_open_jobs_then_fences_new_work(self):
        self._job()

        waiting = self.client.post("/api/update-maintenance/acquire/")
        self.assertEqual(waiting.status_code, 200)
        self.assertEqual(waiting.json(), {"ready": False, "open_jobs": 1, "reason": "jobs_running"})

        Job.objects.all().delete()
        ready = self.client.post("/api/update-maintenance/acquire/")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json(), {"ready": True, "open_jobs": 0, "reason": None})
        self.assertEqual(
            UpdateMaintenance.objects.get(pk=UpdateMaintenance.SINGLETON_ID).state,
            UpdateMaintenance.STATE_APPLYING,
        )

        with self.assertRaises(UpdateApplyInProgress):
            Job.enqueue(job_type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY, payload={})

    def test_release_reopens_the_job_creation_seam(self):
        UpdateMaintenance.objects.create(
            pk=UpdateMaintenance.SINGLETON_ID,
            state=UpdateMaintenance.STATE_APPLYING,
        )

        response = self.client.post("/api/update-maintenance/release/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"released": True})
        job = Job.enqueue(job_type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY, payload={})
        self.assertEqual(job.status, "PENDING")

    def test_startup_clears_a_stale_apply_lock(self):
        UpdateMaintenance.objects.create(
            pk=UpdateMaintenance.SINGLETON_ID,
            state=UpdateMaintenance.STATE_APPLYING,
        )

        self.assertTrue(clear_stale_update_apply_lock())
        self.assertFalse(clear_stale_update_apply_lock())
        self.assertEqual(
            UpdateMaintenance.objects.get(pk=UpdateMaintenance.SINGLETON_ID).state,
            UpdateMaintenance.STATE_IDLE,
        )

    def test_retry_is_refused_during_the_final_update_window(self):
        job = self._job(status="FAILED")
        UpdateMaintenance.objects.create(
            pk=UpdateMaintenance.SINGLETON_ID,
            state=UpdateMaintenance.STATE_APPLYING,
        )

        response = self.client.post(f"/api/jobs/{job.id}/retry/")

        self.assertEqual(response.status_code, 409)
        self.assertIn("applying an update", response.json()["detail"])
