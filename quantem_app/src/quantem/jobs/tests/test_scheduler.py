import os
from datetime import timedelta
from unittest.mock import patch
from uuid import uuid4

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from quantem.jobs.constants import (
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
    JOB_TYPE_RUN_ANALYSIS,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
    QUEUE_P1_INTERACTIVE,
    QUEUE_P2_UPLOAD,
    QUEUE_P3_ROI,
    QUEUE_P4_FULL,
)
from quantem.jobs.models import Job
from quantem.jobs.runner import RunningJob
from quantem.jobs.scheduler import JobScheduler


class _AliveProcess:
    def is_alive(self) -> bool:
        return True


def _track_started_jobs(scheduler: JobScheduler):
    def start_job_side_effect(
        job_id: str,
        resource_class: str,
        job_type: str = "",
    ) -> None:
        scheduler.runner.running[job_id] = RunningJob(
            _AliveProcess(),
            resource_class,
            job_type,
        )

    return start_job_side_effect


class JobSchedulerQueueOrderTests(TestCase):
    def test_ready_jobs_are_ordered_by_queue_rank(self):
        Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            status="PENDING",
            queue_name=QUEUE_P4_FULL,
            priority="high",
            payload_json={},
        )
        Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            status="PENDING",
            queue_name=QUEUE_P3_ROI,
            priority="high",
            payload_json={},
        )
        Job.objects.create(
            type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            status="PENDING",
            queue_name=QUEUE_P2_UPLOAD,
            priority="high",
            payload_json={},
        )
        Job.objects.create(
            type=JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
            status="PENDING",
            queue_name=QUEUE_P1_INTERACTIVE,
            priority="high",
            payload_json={},
        )

        scheduler = JobScheduler()
        jobs = list(scheduler._get_ready_jobs())

        self.assertEqual(
            [job.queue_name for job in jobs[:4]],
            [QUEUE_P1_INTERACTIVE, QUEUE_P2_UPLOAD, QUEUE_P3_ROI, QUEUE_P4_FULL],
        )

    @patch.dict(
        os.environ,
        {"JOB_CPU_WORKERS": "24", "JOB_UPLOAD_PIPELINE_WORKERS": "5"},
        clear=False,
    )
    def test_dispatch_ready_limits_upload_pipeline_jobs_to_five(self):
        for _ in range(7):
            Job.objects.create(
                type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
                status="PENDING",
                queue_name=QUEUE_P2_UPLOAD,
                resource_class="cpu",
                priority="high",
                payload_json={},
            )

        scheduler = JobScheduler()
        with patch.object(scheduler.runner, "start_job") as start_job:
            start_job.side_effect = _track_started_jobs(scheduler)
            scheduler.dispatch_ready()

        self.assertEqual(
            Job.objects.filter(
                status="RUNNING",
                type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            ).count(),
            5,
        )
        self.assertEqual(start_job.call_count, 5)

    @patch.dict(os.environ, {"JOB_GPU_WORKERS": "1"}, clear=False)
    def test_dispatch_ready_serialises_accelerator_jobs_onto_one_slot(self):
        Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            status="PENDING",
            queue_name=QUEUE_P4_FULL,
            resource_class="gpu",
            payload_json={},
        )
        Job.objects.create(
            type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            status="PENDING",
            queue_name=QUEUE_P4_FULL,
            resource_class="gpu",
            payload_json={},
        )

        scheduler = JobScheduler()
        with patch.object(scheduler.runner, "start_job") as start_job:
            start_job.side_effect = _track_started_jobs(scheduler)
            scheduler.dispatch_ready()

        self.assertEqual(start_job.call_count, 1)
        self.assertEqual(Job.objects.filter(status="RUNNING").count(), 1)
        self.assertEqual(Job.objects.filter(status="PENDING").count(), 1)

    @patch.dict(os.environ, {"JOB_CPU_WORKERS": "4"}, clear=False)
    def test_analysis_jobs_do_not_block_the_accelerator_pool(self):
        Job.objects.create(
            type=JOB_TYPE_RUN_ANALYSIS,
            status="PENDING",
            queue_name=QUEUE_P4_FULL,
            resource_class="cpu",
            payload_json={"analysis_run_id": str(uuid4())},
        )

        scheduler = JobScheduler()
        with patch.object(scheduler.runner, "start_job") as start_job:
            start_job.side_effect = _track_started_jobs(scheduler)
            scheduler.dispatch_ready()

        self.assertEqual(start_job.call_count, 1)
        self.assertTrue(scheduler.runner.can_dispatch("gpu"))


class JobSchedulerOrphanRecoveryTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _create_running_job(self, *, job_type: str, max_attempts: int, stale: bool):
        job = Job.objects.create(
            type=job_type,
            status="RUNNING",
            attempts=1,
            max_attempts=max_attempts,
            payload_json={},
            message="running",
            queue_name=QUEUE_P4_FULL,
        )
        heartbeat_at = timezone.now()
        if stale:
            heartbeat_at -= timedelta(hours=1)
        Job.objects.filter(id=job.id).update(heartbeat_at=heartbeat_at)
        job.refresh_from_db()
        return job

    def test_orphaned_job_without_retries_left_is_failed(self):
        job = self._create_running_job(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            max_attempts=1,
            stale=True,
        )

        scheduler = JobScheduler()
        scheduler._recover_orphaned_jobs()

        job.refresh_from_db()

        self.assertEqual(job.status, "FAILED")
        self.assertIn("worker stopped before job completion", job.message)

    def test_orphaned_job_with_retries_left_is_requeued(self):
        job = self._create_running_job(
            job_type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            max_attempts=3,
            stale=True,
        )

        scheduler = JobScheduler()
        scheduler._recover_orphaned_jobs()

        job.refresh_from_db()

        self.assertEqual(job.status, "RETRY")
        self.assertEqual(job.message, "recovered")

    def test_recently_heartbeating_job_is_left_alone(self):
        job = self._create_running_job(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            max_attempts=1,
            stale=False,
        )

        scheduler = JobScheduler()
        scheduler._recover_orphaned_jobs()

        job.refresh_from_db()
        self.assertEqual(job.status, "RUNNING")

    def test_a_job_this_process_is_running_is_never_reaped(self):
        """Ownership, not the clock, is what protects a live job.

        Its heartbeat is refreshed by ``JobRunner.poll``; reaping on the
        heartbeat alone would kill work that is progressing normally.
        """
        job = self._create_running_job(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            max_attempts=1,
            stale=True,
        )

        scheduler = JobScheduler()
        scheduler.runner.running[str(job.id)] = RunningJob(
            _AliveProcess(), "gpu", JOB_TYPE_RUN_SEGMENTATION_FULL
        )
        scheduler._recover_orphaned_jobs()

        job.refresh_from_db()
        self.assertEqual(job.status, "RUNNING")

    def test_a_cancelled_job_whose_worker_is_gone_is_cancelled_immediately(self):
        """The escape hatch. ``cancel`` only set a flag, and ``JobRunner.poll``
        only acts on processes it owns, so a job whose worker had died stayed
        RUNNING forever — wedging its segmentation, since every new run 409s
        while one is active, and blocking ``retry``, which refuses a running
        job."""
        job = self._create_running_job(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            max_attempts=3,
            stale=False,  # a fresh heartbeat must not delay an explicit cancel
        )
        Job.objects.filter(id=job.id).update(cancel_requested=True, message="cancelling")

        scheduler = JobScheduler()
        scheduler._recover_orphaned_jobs()

        job.refresh_from_db()
        self.assertEqual(job.status, "CANCELLED")
        self.assertIsNotNone(job.finished_at)
        self.assertIn("cancelled", job.message)

        # ...and now the documented way back is actually reachable.
        response = self.client.post(f"/api/jobs/{job.id}/retry/")
        self.assertEqual(response.status_code, 200)
        job.refresh_from_db()
        self.assertEqual(job.status, "PENDING")

    def test_a_cancelled_job_with_a_live_worker_is_left_to_the_runner(self):
        """``JobRunner.poll`` terminates the process and records the cancel;
        doing it here as well would mark the job cancelled while its worker was
        still writing to the database."""
        job = self._create_running_job(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            max_attempts=3,
            stale=False,
        )
        Job.objects.filter(id=job.id).update(cancel_requested=True)

        scheduler = JobScheduler()
        scheduler.runner.running[str(job.id)] = RunningJob(
            _AliveProcess(), "gpu", JOB_TYPE_RUN_SEGMENTATION_FULL
        )
        scheduler._recover_orphaned_jobs()

        job.refresh_from_db()
        self.assertEqual(job.status, "RUNNING")

    def _drive_real_scheduler_start(self, scheduler: JobScheduler) -> None:
        """Run the real startup path and stop before the first dispatch.

        ``run_forever`` probes the database, performs the startup reap, then
        calls ``tick()``. ``SystemExit`` is a ``BaseException``, deliberately
        not caught by the loop's keep-alive guard, so raising it from ``tick``
        halts the loop at exactly the point the startup guarantees must already
        hold.
        """
        with patch.object(scheduler, "tick", side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                scheduler.run_forever()

    def test_at_startup_a_running_job_is_reaped_before_its_heartbeat_goes_stale(self):
        """Crash + relaunch must not wait out the heartbeat TTL.

        After the app dies and is reopened within the stale window, the RUNNING
        row's heartbeat is still fresh -- but at scheduler startup no worker
        can predate the scheduler (single-process app), so every RUNNING job is
        an orphan *by definition*. Gating the startup reap on the heartbeat
        left the job wedged for ~8 minutes while the UI showed a frozen
        progress chip.
        """
        job = self._create_running_job(
            job_type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            max_attempts=3,
            stale=False,  # the crash was recent: the heartbeat is fresh
        )

        self._drive_real_scheduler_start(JobScheduler(poll_interval_seconds=0.01))

        job.refresh_from_db()
        self.assertEqual(job.status, "RETRY")
        self.assertEqual(job.message, "recovered")

    def test_at_startup_a_fresh_running_job_with_no_retries_left_is_failed(self):
        """Same startup reap, fail path: no attempts left means FAILED now,
        with the lease released and the domain object reconciled -- not after
        the TTL."""
        job = self._create_running_job(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            max_attempts=1,
            stale=False,
        )

        self._drive_real_scheduler_start(JobScheduler(poll_interval_seconds=0.01))

        job.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertIn("worker stopped before job completion", job.message)

    def test_reaping_happens_on_the_normal_tick_not_only_at_startup(self):
        """A worker can die at any point in a session. Recovering only when the
        database first became ready meant the app had to be restarted."""
        job = self._create_running_job(
            job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
            max_attempts=1,
            stale=True,
        )

        scheduler = JobScheduler()
        with patch.object(scheduler.runner, "start_job"):
            scheduler.tick()

        job.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
