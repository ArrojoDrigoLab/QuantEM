"""One scheduler, and no job left claimed but unrun.

Both properties here were reported by a user as "the app hangs and then blames
the worker", and both come from the same place: a claim is committed before the
work is handed off, so anything that goes wrong in between leaves a row saying
RUNNING with nobody running it.
"""

from __future__ import annotations

import os
from unittest import mock

from django.test import TestCase
from django.utils import timezone

from quantem.jobs.apps import _should_autostart_scheduler
from quantem.jobs.models import Job
from quantem.jobs.runner import WORKER_PROCESS_ENV_VAR
from quantem.jobs.scheduler import JobScheduler


class SchedulerAutostartTests(TestCase):
    """Exactly one process dispatches jobs."""

    def test_server_process_autostarts(self):
        with mock.patch.dict(os.environ, {"QUANTEM_AUTOSTART_JOBS": "1"}, clear=False):
            os.environ.pop(WORKER_PROCESS_ENV_VAR, None)
            os.environ.pop("QUANTEM_DISABLE_JOB_AUTOSTART", None)
            self.assertTrue(_should_autostart_scheduler())

    def test_spawned_worker_does_not(self):
        """The regression. Windows spawns workers, which re-import everything.

        The child inherits QUANTEM_AUTOSTART_JOBS=1, runs django.setup(), and
        would start a scheduler of its own -- N schedulers racing for the same
        rows, and the persistent GPU worker (daemon=True) throwing
        "daemonic processes are not allowed to have children" forever.
        """
        env = {"QUANTEM_AUTOSTART_JOBS": "1", WORKER_PROCESS_ENV_VAR: "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("QUANTEM_DISABLE_JOB_AUTOSTART", None)
            self.assertFalse(_should_autostart_scheduler())

    def test_worker_bootstrap_sets_the_marker(self):
        """_setup_django() is the worker's entry into Django; it must claim first."""
        from quantem.jobs import runner

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(WORKER_PROCESS_ENV_VAR, None)
            runner._setup_django()
            self.assertEqual(os.environ.get(WORKER_PROCESS_ENV_VAR), "1")


class DispatchFailureTests(TestCase):
    """A claim that cannot be handed to a worker is handed back."""

    def _job(self, **kwargs):
        return Job.objects.create(
            type="rebuild_segmentation_overlay",
            payload_json={},
            status="PENDING",
            next_run_at=timezone.now(),
            **kwargs,
        )

    def _dispatch_with_failing_runner(self, job):
        scheduler = JobScheduler()
        with mock.patch.object(
            scheduler.runner, "can_dispatch", return_value=True
        ), mock.patch.object(
            scheduler.runner, "start_job", side_effect=OSError("no worker slot")
        ):
            scheduler.dispatch_ready()
        job.refresh_from_db()
        return job

    def test_retryable_job_is_requeued_not_stranded(self):
        job = self._job(max_attempts=3)
        job = self._dispatch_with_failing_runner(job)
        self.assertEqual(job.status, "RETRY")
        self.assertIsNone(job.started_at)
        self.assertIn("could not be started", job.message)

    def test_last_attempt_fails_honestly(self):
        job = self._job(max_attempts=1)
        job = self._dispatch_with_failing_runner(job)
        self.assertEqual(job.status, "FAILED")
        self.assertIsNotNone(job.finished_at)
        # Names the queue as the cause so the user does not go looking at
        # their image or their model for a fault that is neither.
        self.assertIn("queue fault", job.message)

    def test_without_the_guard_the_job_is_stranded(self):
        """Pin the failure mode this replaced."""
        scheduler = JobScheduler()
        job = self._job(max_attempts=3)
        with mock.patch.object(scheduler.runner, "can_dispatch", return_value=True):
            claimed = scheduler._claim_next_ready_job()
        self.assertEqual(claimed.status, "RUNNING")
        job.refresh_from_db()
        # This is the state the old code left behind on a start_job exception.
        self.assertEqual(job.status, "RUNNING")
        self.assertIsNotNone(job.started_at)


class ReaperMessageTests(TestCase):
    """A dying worker's own words survive the reaper."""

    def _stale_running_job(self, traceback: str) -> Job:
        old = timezone.now() - timezone.timedelta(hours=2)
        job = Job.objects.create(
            type="rebuild_segmentation_overlay",
            payload_json={},
            status="RUNNING",
            attempts=3,
            max_attempts=3,
            error_traceback=traceback,
        )
        Job.objects.filter(id=job.id).update(heartbeat_at=old, updated_at=old)
        job.refresh_from_db()
        return job

    def test_recorded_error_is_not_replaced_by_worker_stopped(self):
        real = (
            "Model pack 'quantem:mito' is not installed.\n"
            "Install it from a local path with:\n"
            "  python -m quantem.registry.install local --all"
        )
        job = self._stale_running_job(real)
        JobScheduler()._recover_orphaned_jobs()
        job.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(job.error_traceback, real)
        self.assertNotIn("worker stopped", job.message)

    def test_silent_death_still_gets_the_generic_message(self):
        job = self._stale_running_job("")
        JobScheduler()._recover_orphaned_jobs()
        job.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertIn("worker stopped before job completion", job.message)
