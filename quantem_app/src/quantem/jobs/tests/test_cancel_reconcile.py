"""Cancel is terminal, and a background failure cannot revoke "Done".

Two holes in the reconciler, both found by driving the app rather than reading
it, and both with the same shape: a job reached a conclusion and the thing it
was carrying did not.

* **Cancel never reconciled.** ``reconcile_domain_objects_for_failed_job`` was
  wired into every FAILED path and neither of the two CANCELLED ones. A
  cancelled analysis left its ``AnalysisRun`` at PENDING forever beside a queue
  row reading CANCELLED -- the exact two-rows-two-truths screen the reconciler
  was written to end. Worse for fine-tuning: the ``Adapter`` stayed RUNNING, and
  that row is what the Adapt wizard reads to decide what is in flight, so one
  Cancel click made fine-tuning permanently unusable for that segmentation.
  Cancel is also the button the app offers on work it says takes "tens of
  minutes", so this is the likely path, not the rare one.

  **A first fix for this was incomplete, and its tests hid that.** Wiring the
  reconciler into the worker's own ``except JobCancelledError`` arm and the
  scheduler's orphan-cancel branch misses the ordinary case: pressing Cancel on
  a job the runner owns. ``JobRunner.poll`` *terminates* the process, so the
  worker's arm never runs, and the scheduler's branch only sees jobs with no
  live worker. The tests passed because they called the reconciler directly --
  green tests, and the app still did the wrong thing. ``PollCancelTests`` below
  drives ``poll()`` instead.

* **A failure could un-mark an image the user marked Done.**
  ``_reconcile_segmentation`` wrote ``FAILED`` unconditionally while the other
  three reconcilers filtered on their own unfinished sets. ``COMPLETED`` is a
  segmentation's SUCCESS *and* carries the completion lock, so an unrelated
  background failure silently dropped a guarantee the user had set by hand.
"""

from __future__ import annotations

from django.test import TestCase
from django.utils import timezone

from quantem.analysis.models import AnalysisRun
from quantem.assets.models import Asset
from quantem.finetune.models import Adapter
from quantem.jobs.constants import ACTIVE_SEGMENTATION_JOB_TYPES
from quantem.jobs.failure_reconcile import (
    CANCELLED_DETAIL,
    reconcile_domain_objects_for_cancelled_job,
    reconcile_domain_objects_for_failed_job,
)
from quantem.jobs.models import Job
from quantem.jobs.runner import JobRunner, RunningJob
from quantem.segmentation.models import ImageSegmentation, SegmentationType

#: The real registered name, not a guess: a job type the reconciler table does
#: not know is silently a no-op, which would make these tests pass for the
#: wrong reason.
JOB_TYPE_RUN_SEGMENTATION_FULL = "run_segmentation_full_task"


def test_the_job_type_under_test_is_a_real_one():
    assert JOB_TYPE_RUN_SEGMENTATION_FULL in ACTIVE_SEGMENTATION_JOB_TYPES


def _segmentation() -> ImageSegmentation:
    asset = Asset.objects.create(display_name="img", original_filename="img.tif")
    seg_type, _ = SegmentationType.objects.get_or_create(
        internal_name="dino_mito",
        defaults={"short_name": "Mito", "long_name": "Mitochondria"},
    )
    return ImageSegmentation.objects.create(asset=asset, segmentation_type=seg_type)


class CancelledJobReconcilesTests(TestCase):
    def test_a_cancelled_analysis_run_does_not_stay_pending(self):
        segmentation = _segmentation()
        run = AnalysisRun.objects.create(segmentation=segmentation, status="PENDING")

        reconcile_domain_objects_for_cancelled_job("run_analysis", {"analysis_run_id": str(run.id)})

        run.refresh_from_db()
        self.assertEqual(run.status, "FAILED")
        self.assertEqual(run.error, CANCELLED_DETAIL)
        self.assertIsNotNone(run.finished_at)

    def test_a_cancelled_training_releases_the_adapt_wizard(self):
        """The regression that made fine-tuning unusable for a segmentation."""
        segmentation = _segmentation()
        adapter = Adapter.objects.create(
            segmentation=segmentation, base_model="quantem:mito", status="RUNNING"
        )

        reconcile_domain_objects_for_cancelled_job(
            "train_organelle_adapter", {"adapter_id": str(adapter.id)}
        )

        adapter.refresh_from_db()
        self.assertNotEqual(
            adapter.status,
            "RUNNING",
            "a RUNNING adapter keeps the wizard insisting a run is live, with no way back",
        )
        self.assertEqual(adapter.status, "FAILED")
        self.assertIn("Cancelled", adapter.error)

    def test_the_message_says_it_was_cancelled_not_that_it_crashed(self):
        segmentation = _segmentation()
        run = AnalysisRun.objects.create(segmentation=segmentation, status="RUNNING")
        reconcile_domain_objects_for_cancelled_job("run_analysis", {"analysis_run_id": str(run.id)})
        run.refresh_from_db()
        self.assertIn("Cancelled", run.error)
        self.assertIn("start it again", run.error.lower())

    def test_a_finished_record_is_left_alone(self):
        segmentation = _segmentation()
        run = AnalysisRun.objects.create(segmentation=segmentation, status="SUCCESS")
        reconcile_domain_objects_for_cancelled_job("run_analysis", {"analysis_run_id": str(run.id)})
        run.refresh_from_db()
        self.assertEqual(run.status, "SUCCESS")


class DoneSurvivesABackgroundFailureTests(TestCase):
    def test_a_failed_run_does_not_un_mark_a_completed_segmentation(self):
        segmentation = _segmentation()
        segmentation.status_stage = "COMPLETED"
        segmentation.save(update_fields=["status_stage"])

        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_RUN_SEGMENTATION_FULL,
            {"segmentation_id": str(segmentation.id)},
            "worker died",
        )

        segmentation.refresh_from_db()
        self.assertEqual(
            segmentation.status_stage,
            "COMPLETED",
            "a background failure must not revoke a lock the user set by hand",
        )
        self.assertEqual(segmentation.status_error, "")

    def test_an_unfinished_segmentation_still_reports_the_failure(self):
        segmentation = _segmentation()
        segmentation.status_stage = "RUNNING_INFERENCE"
        segmentation.save(update_fields=["status_stage"])

        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_RUN_SEGMENTATION_FULL,
            {"segmentation_id": str(segmentation.id)},
            "worker died",
        )

        segmentation.refresh_from_db()
        self.assertEqual(segmentation.status_stage, "FAILED")
        self.assertEqual(segmentation.status_error, "worker died")

    def test_an_already_failed_segmentation_keeps_its_own_message(self):
        segmentation = _segmentation()
        segmentation.status_stage = "FAILED"
        segmentation.status_error = "the model pack is not installed"
        segmentation.save(update_fields=["status_stage", "status_error"])

        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_RUN_SEGMENTATION_FULL,
            {"segmentation_id": str(segmentation.id)},
            "worker stopped",
        )

        segmentation.refresh_from_db()
        self.assertEqual(segmentation.status_error, "the model pack is not installed")


class _AliveThenTerminated:
    """A worker the runner owns: alive until poll() terminates it."""

    def __init__(self) -> None:
        self.terminated = False
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return not self.terminated

    def terminate(self) -> None:
        self.terminated = True
        self.exitcode = -15

    def join(self, timeout: float | None = None) -> None:
        return None


class PollCancelTests(TestCase):
    """The path a user actually takes: Cancel on a running job.

    Driven through ``JobRunner.poll`` rather than the reconciler, because
    calling the reconciler directly is what made the previous version of this
    file pass while a cancelled analysis sat at RUNNING forever.
    """

    def _cancelled_job(self, job_type: str, payload: dict) -> Job:
        job = Job.objects.create(type=job_type, payload_json=payload, status="RUNNING")
        Job.objects.filter(id=job.id).update(cancel_requested=True)
        job.refresh_from_db()
        return job

    def _poll_with_live_worker(self, job: Job) -> _AliveThenTerminated:
        runner = JobRunner()
        proc = _AliveThenTerminated()
        runner.running[str(job.id)] = RunningJob(proc, "cpu", job.type)
        runner.poll()
        return proc

    def test_cancelling_a_running_analysis_concludes_its_run(self):
        segmentation = _segmentation()
        run = AnalysisRun.objects.create(segmentation=segmentation, status="RUNNING")
        job = self._cancelled_job("run_analysis", {"analysis_run_id": str(run.id)})

        proc = self._poll_with_live_worker(job)

        self.assertTrue(proc.terminated, "poll() should have terminated the worker")
        job.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(job.status, "CANCELLED")
        self.assertNotEqual(
            run.status,
            "RUNNING",
            "the queue said CANCELLED while the Analysis screen said RUNNING forever",
        )
        self.assertEqual(run.status, "FAILED")
        self.assertIn("Cancelled", run.error)

    def test_cancelling_a_running_training_concludes_its_adapter(self):
        segmentation = _segmentation()
        adapter = Adapter.objects.create(
            segmentation=segmentation, base_model="quantem:mito", status="RUNNING"
        )
        job = self._cancelled_job("train_organelle_adapter", {"adapter_id": str(adapter.id)})

        self._poll_with_live_worker(job)

        adapter.refresh_from_db()
        self.assertEqual(adapter.status, "FAILED")
        self.assertIn("Cancelled", adapter.error)

    def test_a_running_job_nobody_cancelled_is_left_alone(self):
        segmentation = _segmentation()
        run = AnalysisRun.objects.create(segmentation=segmentation, status="RUNNING")
        job = Job.objects.create(
            type="run_analysis",
            payload_json={"analysis_run_id": str(run.id)},
            status="RUNNING",
        )

        proc = self._poll_with_live_worker(job)

        self.assertFalse(proc.terminated)
        run.refresh_from_db()
        self.assertEqual(run.status, "RUNNING")


class DeadWorkerReleasesLeasesTests(TestCase):
    """A dead worker's storage leases must not outlive its job.

    Closing the app mid-inference left the job's leases ACTIVE with a 6-hour
    TTL. The reaper marked the job FAILED but never released them, so every
    retry -- including after a clean restart -- failed with "Storage artifact
    is leased by another active job", the labeling header kept showing the
    stale "worker stopped" message, and no screen offers a segmentation
    delete: there was no user-space escape for six hours. Reproduced three
    times before the fix; the only recovery was editing sqlite by hand.
    """

    def _running_job_with_lease(
        self, *, stale: bool, job_type: str = "run_segmentation_full_task", **job_kwargs
    ):
        from quantem.core.local_storage import StoragePath
        from quantem.jobs.storage_leases import (
            StorageArtifactLease,
            acquire_storage_artifact_leases,
        )

        job = Job.objects.create(
            type=job_type,
            payload_json={},
            status="RUNNING",
            **job_kwargs,
        )
        acquire_storage_artifact_leases(
            job,
            [StoragePath(relpath="data/prob_maps/lease-test", lease_required=True)],
        )
        if stale:
            old = timezone.now() - timezone.timedelta(hours=2)
            Job.objects.filter(id=job.id).update(heartbeat_at=old, updated_at=old)
            job.refresh_from_db()
        active = StorageArtifactLease.objects.filter(
            job=job, status=StorageArtifactLease.STATUS_ACTIVE
        )
        self.assertEqual(active.count(), 1, "fixture must hold a live lease")
        return job

    def _active_leases(self, job):
        from quantem.jobs.storage_leases import StorageArtifactLease

        return StorageArtifactLease.objects.filter(
            job=job, status=StorageArtifactLease.STATUS_ACTIVE
        ).count()

    def test_the_reaper_releases_leases_when_it_fails_a_job(self):
        from quantem.jobs.scheduler import JobScheduler

        job = self._running_job_with_lease(stale=True, attempts=3, max_attempts=3)
        JobScheduler()._recover_orphaned_jobs()

        job.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(self._active_leases(job), 0)

    def test_the_reaper_releases_leases_when_it_requeues_a_job(self):
        """The retry re-acquires; a lease left ACTIVE is what blocked it."""
        from quantem.jobs.scheduler import JobScheduler

        # A retryable type: full-segmentation runs are in NO_RETRY_JOB_TYPES,
        # so the RETRY branch needs a type the scheduler will requeue.
        job = self._running_job_with_lease(
            stale=True, job_type="rebuild_segmentation_overlay", max_attempts=3
        )
        JobScheduler()._recover_orphaned_jobs()

        job.refresh_from_db()
        self.assertEqual(job.status, "RETRY")
        self.assertEqual(self._active_leases(job), 0)

    def test_a_retry_can_acquire_the_path_the_dead_worker_held(self):
        """The user-visible claim: the segmentation is not bricked."""
        from quantem.core.local_storage import StoragePath
        from quantem.jobs.scheduler import JobScheduler
        from quantem.jobs.storage_leases import acquire_storage_artifact_leases

        dead = self._running_job_with_lease(stale=True, attempts=3, max_attempts=3)
        JobScheduler()._recover_orphaned_jobs()

        retry = Job.objects.create(
            type="run_segmentation_full_task", payload_json={}, status="RUNNING"
        )
        # Without the release this raised StorageError for the next 6 hours.
        acquire_storage_artifact_leases(
            retry,
            [StoragePath(relpath="data/prob_maps/lease-test", lease_required=True)],
        )
        self.assertEqual(self._active_leases(retry), 1)
        self.assertEqual(self._active_leases(dead), 0)

    def test_poll_terminate_on_cancel_releases_leases(self):
        job = self._running_job_with_lease(stale=False)
        Job.objects.filter(id=job.id).update(cancel_requested=True)
        job.refresh_from_db()

        runner = JobRunner()
        proc = _AliveThenTerminated()
        runner.running[str(job.id)] = RunningJob(proc, "cpu", job.type)
        runner.poll()

        self.assertTrue(proc.terminated)
        job.refresh_from_db()
        self.assertEqual(job.status, "CANCELLED")
        self.assertEqual(self._active_leases(job), 0)
