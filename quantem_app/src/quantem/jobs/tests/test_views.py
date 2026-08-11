from datetime import datetime, timedelta
from uuid import uuid4

from django.test import TestCase
from django.urls import NoReverseMatch, reverse
from django.utils import timezone
from rest_framework.test import APIClient

from quantem.jobs.constants import (
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
    QUEUE_P1_INTERACTIVE,
    QUEUE_P2_UPLOAD,
)
from quantem.jobs.models import Job


class JobQueueStatusViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _create_job(
        self,
        *,
        status: str,
        finished_at: datetime | None = None,
    ) -> Job:
        job = Job.objects.create(
            type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
            status=status,
            payload_json={},
            queue_name=QUEUE_P1_INTERACTIVE,
        )
        if finished_at is not None:
            Job.objects.filter(id=job.id).update(
                finished_at=finished_at,
                updated_at=finished_at,
            )
            job.refresh_from_db()
        return job

    def test_queue_status_includes_terminal_jobs_sorted_by_recent_finish_time(self):
        now = timezone.now()
        completed_older = self._create_job(
            status="SUCCESS", finished_at=now - timedelta(minutes=12)
        )
        completed_newer = self._create_job(status="SUCCESS", finished_at=now - timedelta(minutes=3))
        failed_older = self._create_job(status="FAILED", finished_at=now - timedelta(minutes=10))
        failed_newer = self._create_job(status="CANCELLED", finished_at=now - timedelta(minutes=1))
        self._create_job(status="RUNNING")
        self._create_job(status="PENDING")

        response = self.client.get("/api/jobs/queue-status/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["running"]), 1)
        self.assertEqual(len(payload["queues"]), 1)
        self.assertEqual(payload["queues"][0]["queue_name"], QUEUE_P1_INTERACTIVE)
        self.assertEqual(payload["queues"][0]["display_name"], "P1 Interactive")

        completed_ids = [item["id"] for item in payload["completed"]]
        failed_ids = [item["id"] for item in payload["failed"]]
        self.assertEqual(completed_ids[:2], [str(completed_newer.id), str(completed_older.id)])
        self.assertEqual(failed_ids[:2], [str(failed_newer.id), str(failed_older.id)])
        self.assertIsNotNone(payload["completed"][0]["finished_at"])
        self.assertIsNotNone(payload["failed"][0]["finished_at"])

    def test_queue_status_reports_whether_the_scheduler_runs_in_process(self):
        response = self.client.get("/api/jobs/queue-status/")

        self.assertEqual(response.status_code, 200)
        worker = response.json()["worker"]
        self.assertIn("scheduler_in_process", worker)
        self.assertIsInstance(worker["scheduler_in_process"], bool)


class JobWorkerRestartEndpointTests(TestCase):
    def test_worker_restart_endpoint_does_not_exist(self):
        """Restarting a worker is operator tooling; a shipped app has no operator."""
        with self.assertRaises(NoReverseMatch):
            reverse("job-worker-restart")


class JobClearDoneViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_clear_done_removes_terminal_jobs_only(self):
        self._create_job(status="SUCCESS")
        self._create_job(status="FAILED")
        self._create_job(status="CANCELLED")
        running = self._create_job(status="RUNNING")
        pending = self._create_job(status="PENDING")

        response = self.client.post("/api/jobs/clear-done/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], 3)
        self.assertFalse(Job.objects.filter(status="SUCCESS").exists())
        self.assertFalse(Job.objects.filter(status="FAILED").exists())
        self.assertFalse(Job.objects.filter(status="CANCELLED").exists())
        self.assertTrue(Job.objects.filter(id=running.id).exists())
        self.assertTrue(Job.objects.filter(id=pending.id).exists())

    def _create_job(self, *, status: str) -> Job:
        return Job.objects.create(
            type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
            status=status,
            payload_json={},
            queue_name=QUEUE_P1_INTERACTIVE,
        )


class JobRetryViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_retry_failed_job_requeues_and_resets_runtime_fields(self):
        started_at = timezone.now() - timedelta(minutes=3)
        finished_at = timezone.now() - timedelta(minutes=1)
        job = Job.objects.create(
            type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            status="FAILED",
            progress=87.0,
            message="failed: RuntimeError: boom",
            cancel_requested=True,
            started_at=started_at,
            finished_at=finished_at,
            attempts=3,
            max_attempts=3,
            next_run_at=finished_at,
            payload_json={"asset_id": str(uuid4())},
            result_json={"old": "result"},
            error_traceback="traceback",
            queue_name=QUEUE_P2_UPLOAD,
        )

        retry_requested_at = timezone.now()
        response = self.client.post(f"/api/jobs/{job.id}/retry/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "queued")

        job.refresh_from_db()
        self.assertEqual(job.status, "PENDING")
        self.assertEqual(job.progress, 0.0)
        self.assertEqual(job.message, "retry queued")
        self.assertFalse(job.cancel_requested)
        self.assertIsNone(job.started_at)
        self.assertIsNone(job.finished_at)
        self.assertEqual(job.attempts, 0)
        self.assertIsNone(job.result_json)
        self.assertEqual(job.error_traceback, "")
        self.assertGreaterEqual(job.next_run_at, retry_requested_at)

    def test_retry_rejects_non_terminal_job(self):
        job = Job.objects.create(
            type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            status="RUNNING",
            payload_json={"asset_id": str(uuid4())},
            queue_name=QUEUE_P2_UPLOAD,
        )

        response = self.client.post(f"/api/jobs/{job.id}/retry/")

        self.assertEqual(response.status_code, 409)
        body = response.json()
        self.assertIn("Only failed or cancelled jobs can be retried", body["detail"])
        # The refusal has to name the way out. A running job whose worker is
        # gone is exactly the state a user reaches this endpoint from, and
        # cancelling is what makes it retryable.
        # Named ``POST /api/jobs/<id>/cancel/`` until 2026-08-10. This detail
        # lands in the Tasks & Queues panel, where the reader has a Cancel
        # button and no way to issue a request; invariant I-12 forbids the verb
        # and the route, and the id is a field, not a sentence.
        self.assertIn("Cancel", body["detail"])
        self.assertIn("Tasks & Queues", body["detail"])
        self.assertNotIn("/api/jobs/", body["detail"])
        self.assertNotIn(str(job.id), body["detail"])
        self.assertEqual(body["job_status"], "RUNNING")
        self.assertEqual(body["job_id"], str(job.id))


class JobCancelViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_cancel_flags_a_running_job(self):
        job = Job.objects.create(
            type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            status="RUNNING",
            payload_json={},
            queue_name=QUEUE_P2_UPLOAD,
        )

        response = self.client.post(f"/api/jobs/{job.id}/cancel/")

        self.assertEqual(response.status_code, 200)
        job.refresh_from_db()
        self.assertTrue(job.cancel_requested)
        self.assertEqual(job.message, "cancelling")

    def test_cancel_rejects_a_queued_job(self):
        job = Job.objects.create(
            type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            status="PENDING",
            payload_json={},
            queue_name=QUEUE_P2_UPLOAD,
        )

        response = self.client.post(f"/api/jobs/{job.id}/cancel/")

        self.assertEqual(response.status_code, 409)
        job.refresh_from_db()
        self.assertFalse(job.cancel_requested)
