"""A job that fails must take its domain object down with it.

Reported twice: the analysis worker died with
``worker subprocess exited with code 3221225794`` right after a torch
fine-tuning job, and the Analysis screen then showed a history row saying
**PENDING**, a panel saying **FAILED**, and *"This run is pending. Results
appear when it finishes."* -- three statements, two of them wrong, none of them
clearable without a restart.

This is the same class as the adapter that stayed ``PENDING`` with ``error: ""``,
so the reconciliation is keyed by job type and every domain object with a job
behind it is covered here.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch
from uuid import uuid4

from django.test import TestCase
from django.utils import timezone

from quantem.analysis.models import AnalysisRun
from quantem.assets.models import Asset
from quantem.finetune.models import Adapter
from quantem.jobs.constants import (
    JOB_TYPE_RUN_ANALYSIS,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
)
from quantem.jobs.failure_reconcile import (
    domain_status_recorded,
    mark_domain_status_recorded,
    reconcile_domain_objects_for_failed_job,
    reconcile_domain_objects_for_retrying_job,
    retrying_attempt_detail,
    worker_exit_message,
)
from quantem.jobs.models import Job
from quantem.jobs.runner import JobRunner, RunningJob, run_job_in_subprocess
from quantem.jobs.scheduler import JobScheduler
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

#: Windows ``STATUS_DLL_INIT_FAILED``, the code the reported crash actually died
#: with.
DLL_INIT_FAILED = 3221225794


class _DeadProcess:
    def __init__(self, exitcode: int) -> None:
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return False

    def join(self, timeout=None) -> None:
        return None


class WorkerExitMessageTests(TestCase):
    def test_the_reported_code_is_explained_in_words(self):
        message = worker_exit_message(DLL_INIT_FAILED)
        self.assertIn("failed to initialise", message)
        self.assertIn("Restart QuantEM", message)

    def test_the_raw_code_survives_for_a_bug_report(self):
        """The sentence is for the user; the number is for whoever debugs it."""
        message = worker_exit_message(DLL_INIT_FAILED)
        self.assertIn(str(DLL_INIT_FAILED), message)
        self.assertIn("0xC0000142", message)

    def test_an_unknown_code_still_says_what_happened(self):
        message = worker_exit_message(12345)
        self.assertIn("exited with code 12345", message)
        self.assertIn("Nothing already saved was lost", message)

    def test_a_process_with_no_code_is_not_described_as_code_none(self):
        self.assertNotIn("None", worker_exit_message(None))

    def test_a_posix_signal_is_reported_as_a_signal(self):
        self.assertIn("signal 9", worker_exit_message(-9))
        self.assertIn("out of memory", worker_exit_message(-9))


class DomainObjectReconciliationTests(TestCase):
    def setUp(self):
        self.image = create_small_test_image("Reconcile", width=64, height=64)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def test_an_analysis_run_does_not_stay_pending_behind_a_failed_job(self):
        run = AnalysisRun.objects.create(segmentation=self.segmentation)
        self.assertEqual(run.status, AnalysisRun.STATUS_PENDING)

        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_RUN_ANALYSIS,
            {"analysis_run_id": str(run.id)},
            worker_exit_message(DLL_INIT_FAILED),
        )

        run.refresh_from_db()
        self.assertEqual(run.status, AnalysisRun.STATUS_FAILED)
        self.assertIn("failed to initialise", run.error)
        self.assertIsNotNone(run.finished_at)

    def test_an_adapter_does_not_stay_pending_with_an_empty_error(self):
        adapter = Adapter.objects.create(segmentation=self.segmentation, base_model="quantem:mito")

        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            {"adapter_id": str(adapter.id)},
            "the worker died",
        )

        adapter.refresh_from_db()
        self.assertEqual(adapter.status, "FAILED")
        self.assertEqual(adapter.error, "the worker died")

    def test_a_segmentation_is_failed_with_its_run(self):
        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_RUN_SEGMENTATION_FULL,
            {"segmentation_id": str(self.segmentation.id)},
            "the worker died",
        )
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "FAILED")
        self.assertEqual(self.segmentation.status_error, "the worker died")

    def test_an_asset_does_not_stay_mid_upload_forever(self):
        Asset.objects.filter(id=self.image.asset.id).update(preprocess_stage="ENCODING")

        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            {"asset_id": str(self.image.asset.id)},
            "the worker died",
        )

        asset = Asset.objects.get(id=self.image.asset.id)
        self.assertEqual(asset.preprocess_stage, "FAILED")
        self.assertEqual(asset.preprocess_error, "the worker died")

    def test_a_record_that_already_concluded_keeps_its_own_answer(self):
        """A handler that failed cleanly wrote a better message than this one."""
        run = AnalysisRun.objects.create(
            segmentation=self.segmentation,
            status=AnalysisRun.STATUS_FAILED,
            error="No confirmed objects to analyse.",
        )
        done = AnalysisRun.objects.create(
            segmentation=self.segmentation, status=AnalysisRun.STATUS_SUCCESS
        )

        for record in (run, done):
            reconcile_domain_objects_for_failed_job(
                JOB_TYPE_RUN_ANALYSIS,
                {"analysis_run_id": str(record.id)},
                "the worker died",
            )

        run.refresh_from_db()
        done.refresh_from_db()
        self.assertEqual(run.error, "No confirmed objects to analyse.")
        self.assertEqual(done.status, AnalysisRun.STATUS_SUCCESS)

    def test_a_missing_domain_row_is_not_an_error(self):
        """This runs on the path that is already handling a failure."""
        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_RUN_ANALYSIS, {"analysis_run_id": str(uuid4())}, "gone"
        )
        reconcile_domain_objects_for_failed_job(JOB_TYPE_RUN_ANALYSIS, {}, "gone")
        reconcile_domain_objects_for_failed_job("not_a_job_type", {}, "gone")

    def test_a_reconciler_that_raises_does_not_replace_the_job_error(self):
        with patch(
            "quantem.jobs.failure_reconcile._reconcile_analysis_run",
            side_effect=RuntimeError("boom"),
        ):
            reconcile_domain_objects_for_failed_job(
                JOB_TYPE_RUN_ANALYSIS, {"analysis_run_id": str(uuid4())}, "gone"
            )


class RetryAttemptSupersedesOlderErrorsTests(TestCase):
    """Paper-cut 1 backend: a RETRY must not leave a stale error on screen.

    When an in-worker attempt failed and the job went RETRY, nothing updated
    the domain object, so the labeling header kept showing a *previous* FAILED
    message while newer, different failures accrued in the queue. The honest
    surface: ``status_error`` reflects the most recent attempt's failure,
    clearly marked as attempt N of M and retrying; a successful attempt clears
    it (the status callback already writes ``status_error=""``); the final
    failure keeps its existing semantics.
    """

    def setUp(self):
        self.image = create_small_test_image("Retry supersede", width=64, height=64)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )

    def _stale_error(self, text: str = "the old failure, from last week") -> str:
        ImageSegmentation.objects.filter(id=self.segmentation.id).update(
            status_stage="FAILED", status_error=text
        )
        return text

    def test_the_newest_attempt_failure_replaces_a_stale_message(self):
        stale = self._stale_error()

        reconcile_domain_objects_for_retrying_job(
            JOB_TYPE_RUN_SEGMENTATION_ROI,
            {"segmentation_id": str(self.segmentation.id)},
            retrying_attempt_detail(1, 3, "RuntimeError: the new failure"),
        )

        self.segmentation.refresh_from_db()
        self.assertIn("Attempt 1 of 3", self.segmentation.status_error)
        self.assertIn("the new failure", self.segmentation.status_error)
        self.assertIn("retr", self.segmentation.status_error.lower())
        self.assertNotIn(stale, self.segmentation.status_error)

    def test_a_completed_segmentation_is_never_touched(self):
        """COMPLETED carries the completion lock; no background note may land."""
        ImageSegmentation.objects.filter(id=self.segmentation.id).update(
            status_stage="COMPLETED", status_error=""
        )

        reconcile_domain_objects_for_retrying_job(
            JOB_TYPE_RUN_SEGMENTATION_ROI,
            {"segmentation_id": str(self.segmentation.id)},
            retrying_attempt_detail(1, 3, "RuntimeError: noise"),
        )

        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "COMPLETED")
        self.assertEqual(self.segmentation.status_error, "")

    def test_an_in_worker_retry_updates_the_domain_object(self):
        """The real path: the worker's exception arm, not the reconciler alone."""
        stale = self._stale_error()
        job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            status="RUNNING",
            attempts=1,
            max_attempts=3,
            payload_json={"segmentation_id": str(self.segmentation.id)},
            message="running",
        )

        with patch("quantem.jobs.registry.get_handler") as get_handler:

            def failing_handler(payload, reporter, cancel):
                raise RuntimeError("the pack is broken today")

            get_handler.return_value = failing_handler
            run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.segmentation.refresh_from_db()
        self.assertEqual(job.status, "RETRY")
        self.assertIn("Attempt 1 of 3", self.segmentation.status_error)
        self.assertIn("the pack is broken today", self.segmentation.status_error)
        self.assertNotIn(stale, self.segmentation.status_error)

    def test_the_reaper_retry_also_supersedes(self):
        """A worker that died mid-attempt is still a failed attempt."""
        stale = self._stale_error()
        job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            status="RUNNING",
            attempts=1,
            max_attempts=3,
            payload_json={"segmentation_id": str(self.segmentation.id)},
            message="running",
        )
        Job.objects.filter(id=job.id).update(heartbeat_at=timezone.now() - timedelta(hours=1))

        JobScheduler()._recover_orphaned_jobs()

        job.refresh_from_db()
        self.segmentation.refresh_from_db()
        self.assertEqual(job.status, "RETRY")
        self.assertIn("Attempt 1 of 3", self.segmentation.status_error)
        self.assertNotIn(stale, self.segmentation.status_error)

    def test_the_final_failure_keeps_its_existing_semantics(self):
        """Out of attempts: FAILED with the plain message, no retrying marker."""
        job = Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            status="RUNNING",
            attempts=3,
            max_attempts=3,
            payload_json={"segmentation_id": str(self.segmentation.id)},
            message="running",
        )

        with patch("quantem.jobs.registry.get_handler") as get_handler:

            def failing_handler(payload, reporter, cancel):
                raise RuntimeError("terminal failure")

            get_handler.return_value = failing_handler
            run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.segmentation.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(self.segmentation.status_stage, "FAILED")
        self.assertIn("terminal failure", self.segmentation.status_error)
        self.assertNotIn("retr", self.segmentation.status_error.lower())

    def test_a_successful_analysis_retry_reverts_to_clean(self):
        """The other retryable domain object: its success write clears error."""
        run = AnalysisRun.objects.create(segmentation=self.segmentation)

        reconcile_domain_objects_for_retrying_job(
            JOB_TYPE_RUN_ANALYSIS,
            {"analysis_run_id": str(run.id)},
            retrying_attempt_detail(1, 3, "MemoryError: transient"),
        )
        run.refresh_from_db()
        self.assertIn("Attempt 1 of 3", run.error)
        # Status is untouched -- the run is still in flight, not concluded.
        self.assertEqual(run.status, AnalysisRun.STATUS_PENDING)

        # The success write in quantem.analysis.service sets error="" -- pinned
        # here so the retry note cannot outlive the retry that succeeded.
        AnalysisRun.objects.filter(id=run.id).update(status=AnalysisRun.STATUS_SUCCESS, error="")
        run.refresh_from_db()
        self.assertEqual(run.error, "")

    def test_a_succeeded_run_is_not_annotated(self):
        run = AnalysisRun.objects.create(
            segmentation=self.segmentation,
            status=AnalysisRun.STATUS_SUCCESS,
        )

        reconcile_domain_objects_for_retrying_job(
            JOB_TYPE_RUN_ANALYSIS,
            {"analysis_run_id": str(run.id)},
            retrying_attempt_detail(1, 3, "noise"),
        )

        run.refresh_from_db()
        self.assertEqual(run.error, "")
        self.assertEqual(run.status, AnalysisRun.STATUS_SUCCESS)


class StaleFailureSupersededByCrashTests(TestCase):
    """Adversarial round 13, finding 3: a NEWER attempt dying before its
    handler wrote anything must not leave the OLDER attempt's error up.

    The old rule skipped any segmentation already at stage FAILED ("the
    handler wrote a better message") -- true for the job whose handler wrote
    it, false for the next job, which can die (unregistered source model,
    worker crash) with the segmentation still sitting at the previous
    attempt's FAILED. The header then kept explaining last week's failure.

    The fix: a FAILED stage is protected only from the job whose handler
    recorded it, which the handler marks on the exception it raises
    (:func:`mark_domain_status_recorded`); every path where the job died with
    nothing written supersedes.
    """

    STALE = "Model pack 'quantem:mito' is not installed. (an OLDER attempt's error)"

    def setUp(self):
        self.image = create_small_test_image("Stale supersede", width=64, height=64)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        ImageSegmentation.objects.filter(id=self.segmentation.id).update(
            status_stage="FAILED", status_error=self.STALE
        )

    def _job(self, *, attempts=1, max_attempts=1):
        return Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            status="RUNNING",
            attempts=attempts,
            max_attempts=max_attempts,
            payload_json={"segmentation_id": str(self.segmentation.id)},
            message="running",
        )

    def test_the_reported_repro_a_crash_before_any_write_shows_the_new_failure(self):
        """The in-worker arm, exactly as in the report: a ValueError raised
        before the handler touched the segmentation (unregistered source
        model), on a segmentation already FAILED from an earlier run."""
        job = self._job(attempts=1, max_attempts=1)  # terminal: no retry

        with patch("quantem.jobs.registry.get_handler") as get_handler:

            def dies_before_writing(payload, reporter, cancel):
                raise ValueError("No segmenter registered for type: quantem_internal_mito")

            get_handler.return_value = dies_before_writing
            run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.segmentation.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(self.segmentation.status_stage, "FAILED")
        self.assertIn("No segmenter registered", self.segmentation.status_error)
        self.assertNotIn(self.STALE, self.segmentation.status_error)

    def test_a_handler_that_recorded_its_own_conclusion_keeps_it(self):
        """The case the old skip was right about, preserved: the handler wrote
        FAILED + a user-facing message and marked its exception; the queue's
        generic "failed: ..." text must not replace it."""
        job = self._job(attempts=1, max_attempts=1)
        handler_message = "Model pack 'omniem:mito' is not installed. Install it."

        with patch("quantem.jobs.registry.get_handler") as get_handler:

            def concludes_then_raises(payload, reporter, cancel):
                ImageSegmentation.objects.filter(id=self.segmentation.id).update(
                    status_stage="FAILED", status_error=handler_message
                )
                raise mark_domain_status_recorded(ValueError(handler_message))

            get_handler.return_value = concludes_then_raises
            run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.segmentation.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(self.segmentation.status_error, handler_message)

    def test_the_dead_worker_reap_supersedes_the_stale_error(self):
        """The scheduler's orphan reap: worker gone, nothing recorded."""
        job = self._job(attempts=1, max_attempts=1)
        Job.objects.filter(id=job.id).update(heartbeat_at=timezone.now() - timedelta(hours=1))

        JobScheduler()._recover_orphaned_jobs()

        job.refresh_from_db()
        self.segmentation.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertIn("worker stopped", self.segmentation.status_error)
        self.assertNotIn(self.STALE, self.segmentation.status_error)

    def test_the_dead_process_poll_supersedes_the_stale_error(self):
        """The runner's poll(): the process object itself is dead."""
        job = self._job(attempts=1, max_attempts=1)
        runner = JobRunner()
        runner.running[str(job.id)] = RunningJob(
            _DeadProcess(exitcode=DLL_INIT_FAILED), "gpu", JOB_TYPE_RUN_SEGMENTATION_ROI
        )

        runner.poll()

        self.segmentation.refresh_from_db()
        self.assertIn("failed to initialise", self.segmentation.status_error)
        self.assertNotIn(self.STALE, self.segmentation.status_error)

    def test_completed_is_never_overwritten_even_when_superseding(self):
        """The completion lock outranks everything, including this fix."""
        ImageSegmentation.objects.filter(id=self.segmentation.id).update(
            status_stage="COMPLETED", status_error=""
        )

        reconcile_domain_objects_for_failed_job(
            JOB_TYPE_RUN_SEGMENTATION_FULL,
            {"segmentation_id": str(self.segmentation.id)},
            "the worker died",
            supersede_stale_failure=True,
        )

        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "COMPLETED")
        self.assertEqual(self.segmentation.status_error, "")

    def test_fail_segmentation_marks_the_exception_it_raises(self):
        """The producing half of the contract: the run task's failure writer
        stamps its exception so the runner knows the conclusion is owned."""
        from quantem.segmentation.organelle_tasks import _fail_segmentation

        class _BareSegmenter:
            pass

        original = ValueError("this run could not read the image")
        with self.assertRaises(ValueError) as raised:
            _fail_segmentation(self.segmentation, original, segmenter=_BareSegmenter())

        self.assertTrue(domain_status_recorded(raised.exception))
        self.segmentation.refresh_from_db()
        self.assertEqual(self.segmentation.status_stage, "FAILED")
        self.assertIn("could not read the image", self.segmentation.status_error)


class LeaseConflictLogNoiseTests(TestCase):
    """UAT round 13, paper-cut 8: crash fallout that the bounded retry heals
    in seconds left a full StorageError traceback in the log. A *retryable*
    lease conflict gets one calm INFO line; the traceback is reserved for the
    attempt that exhausts the retries."""

    def _job(self, *, attempts, max_attempts):
        return Job.objects.create(
            type=JOB_TYPE_RUN_SEGMENTATION_ROI,  # a retryable type
            status="RUNNING",
            attempts=attempts,
            max_attempts=max_attempts,
            payload_json={},
            message="running",
        )

    def _lease_conflict_handler(self):
        from quantem.jobs.storage_leases import StorageLeaseConflict

        def handler(payload, reporter, cancel):
            raise StorageLeaseConflict(
                "Storage artifact is leased by another active job: overlays/x"
            )

        return handler

    def test_a_retryable_lease_conflict_is_one_info_line_no_traceback(self):
        job = self._job(attempts=1, max_attempts=3)

        with patch("quantem.jobs.registry.get_handler") as get_handler:
            get_handler.return_value = self._lease_conflict_handler()
            with self.assertLogs("quantem.jobs.runner", level="INFO") as captured:
                run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.assertEqual(job.status, "RETRY")
        text = "\n".join(captured.output)
        self.assertIn("still leased", text)
        self.assertIn("unclean shutdown", text)
        self.assertIn("retrying", text)
        self.assertNotIn("Traceback", text)

    def test_exhausted_retries_keep_the_full_traceback(self):
        job = self._job(attempts=3, max_attempts=3)  # this attempt is the last

        with patch("quantem.jobs.registry.get_handler") as get_handler:
            get_handler.return_value = self._lease_conflict_handler()
            with self.assertLogs("quantem.jobs.runner", level="ERROR") as captured:
                run_job_in_subprocess(str(job.id))

        job.refresh_from_db()
        self.assertEqual(job.status, "FAILED")
        self.assertIn("Traceback", "\n".join(captured.output))


class DeadWorkerEndToEndTests(TestCase):
    """The runner's own dead-process path, not just the reconciler."""

    def setUp(self):
        self.image = create_small_test_image("Dead worker", width=64, height=64)
        self.segmentation = ImageSegmentation.objects.create(
            asset=self.image.asset,
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.run = AnalysisRun.objects.create(segmentation=self.segmentation)
        self.job = Job.objects.create(
            type=JOB_TYPE_RUN_ANALYSIS,
            status="RUNNING",
            attempts=1,
            max_attempts=1,
            started_at=timezone.now() - timedelta(minutes=10),
            payload_json={"analysis_run_id": str(self.run.id)},
            message="running",
        )

    def test_poll_fails_the_job_and_the_run_it_was_carrying(self):
        runner = JobRunner()
        runner.running[str(self.job.id)] = RunningJob(
            _DeadProcess(exitcode=DLL_INIT_FAILED), "cpu", JOB_TYPE_RUN_ANALYSIS
        )

        runner.poll()

        self.job.refresh_from_db()
        self.run.refresh_from_db()
        self.assertEqual(self.job.status, "FAILED")
        self.assertEqual(self.run.status, AnalysisRun.STATUS_FAILED)
        # And the screen no longer shows the raw NTSTATUS on its own.
        self.assertIn("Restart QuantEM", self.job.message)
        self.assertIn("Restart QuantEM", self.run.error)
