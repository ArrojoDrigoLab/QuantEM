import logging
import os
import time
from datetime import timedelta

from django.apps import apps
from django.core.exceptions import ImproperlyConfigured
from django.db import DatabaseError, OperationalError, connections, transaction
from django.db.models import Case, IntegerField, Value, When
from django.utils import timezone

from quantem.jobs.constants import (
    QUEUE_P1_INTERACTIVE,
    QUEUE_P2_UPLOAD,
    QUEUE_P3_ROI,
    QUEUE_P4_FULL,
)
from quantem.jobs.failure_reconcile import (
    reconcile_domain_objects_for_cancelled_job,
    reconcile_domain_objects_for_failed_job,
    reconcile_domain_objects_for_retrying_job,
    retrying_attempt_detail,
)
from quantem.jobs.runner import JobRunner, _job_should_retry
from quantem.jobs.storage_leases import release_storage_artifact_leases


def _get_job_model():
    return apps.get_model("jobs", "Job")


logger = logging.getLogger(__name__)


def _wait_for_database() -> bool:
    """True once the database is not just reachable but actually *migrated*.

    Checking only ``ensure_connection`` was not enough and produced a silent
    first-run failure: the scheduler autostarts on the first DB connection, and
    on a clean install that connection is the one ``cli.cmd_serve`` opens to run
    ``migrate`` — before any table exists. The scheduler then died on
    ``no such table: jobs_job``, the thread was gone, the started-flag stayed
    set, and every upload sat at "NGFF pending" forever with no error. It only
    worked from the second launch onward, which is exactly the case a developer
    tests.
    """
    try:
        connection = connections["default"]
        engine = connection.settings_dict.get("ENGINE")
        if not engine or engine == "django.db.backends.dummy":
            return False
        connection.ensure_connection()
        # The table the scheduler is about to query. Its absence means migrations
        # have not finished, which is normal for a few seconds on first launch.
        table = apps.get_model("jobs", "Job")._meta.db_table
        return table in connection.introspection.table_names()
    except (ImproperlyConfigured, OperationalError, DatabaseError):
        return False


#: How often :meth:`JobScheduler._recover_orphaned_jobs` re-checks for RUNNING
#: jobs that no live worker owns. Cheap (one indexed query over RUNNING rows) and
#: the thing it clears is a permanently wedged segmentation, so it runs on a
#: short cycle rather than only at startup.
REAP_INTERVAL_SECONDS = 15.0

#: How often :meth:`JobScheduler._sweep_abandoned_uploads` looks for upload
#: bytes nobody owns. Far slower than the job reaper: what it clears is disk,
#: not a wedged screen, and the scan reads every name in the uploads directory.
#: Fifteen minutes bounds a leaked import to that long plus the one-hour age
#: threshold the sweep itself applies.
UPLOAD_SWEEP_INTERVAL_SECONDS = 900.0

DEFAULT_HEARTBEAT_STALE_SECONDS = 300


def _heartbeat_stale_seconds() -> int:
    try:
        return max(
            60,
            int(
                str(
                    os.environ.get(
                        "QUANTEM_JOB_HEARTBEAT_STALE_SECONDS",
                        str(DEFAULT_HEARTBEAT_STALE_SECONDS),
                    )
                )
            ),
        )
    except (TypeError, ValueError):
        return DEFAULT_HEARTBEAT_STALE_SECONDS


class JobScheduler:
    def __init__(self, poll_interval_seconds: float = 1.0) -> None:
        self.poll_interval_seconds = poll_interval_seconds
        self.runner = JobRunner()
        self._database_ready = False
        self._last_reap_monotonic = 0.0
        # Not 0.0: that would make the first ``tick`` sweep, and ``tick`` is
        # called directly by tests that have nothing to do with uploads. The
        # start-up sweep is scheduled explicitly in ``run_forever`` instead.
        self._last_upload_sweep_monotonic = time.monotonic()

    def _owns(self, job_id) -> bool:
        """True when a worker in *this* process is running the job right now.

        Everything else that is ``RUNNING`` is either a leftover from a previous
        process or a job whose worker died: nobody will ever finish it, honour
        its cancel flag, or refresh its heartbeat.
        """
        return str(job_id) in self.runner.running

    def _recover_orphaned_jobs(self, *, startup: bool = False) -> None:
        """Clear RUNNING jobs that no live worker owns.

        Two states get a job stuck here forever, and both wedge the
        segmentation: every new run on it is refused with a 409 naming an
        "already queued or running" task.

        * The worker is gone (process died, app was killed mid-run). Its
          heartbeat stops, and after :func:`_heartbeat_stale_seconds` the job is
          retried or failed.
        * ``cancel`` set ``cancel_requested`` on a job whose worker was already
          gone. :meth:`JobRunner.poll` only terminates processes it owns, so
          nothing ever acted on the flag. Here it is honoured immediately —
          there is no process left to wait for — which also makes ``retry``
          reachable, since a cancelled job is retryable and a running one is not.

        ``startup=True`` drops the heartbeat gate. At scheduler startup — before
        the first dispatch — every RUNNING job is an orphan *by definition*:
        this is a single-process application, so no worker can predate the
        scheduler that would have owned it. After a crash and a quick relaunch
        the dead worker's last heartbeat is still fresh, and waiting out the
        stale window (~8 minutes at the defaults) left the user staring at a
        frozen progress chip on a job nothing was running. Reaped immediately
        instead, with exactly the retry/fail/lease-release/reconcile logic the
        timed reaper uses.
        """
        Job = _get_job_model()
        now = timezone.now()
        stale_cutoff = now - timedelta(seconds=_heartbeat_stale_seconds())
        for job in Job.objects.filter(status="RUNNING"):
            if self._owns(job.id):
                # A live worker: poll() heartbeats it and terminates it on cancel.
                continue

            if job.cancel_requested:
                Job.objects.filter(id=job.id, status="RUNNING").update(
                    status="CANCELLED",
                    finished_at=now,
                    message="cancelled (no worker was running it)",
                )
                release_storage_artifact_leases(job)
                reconcile_domain_objects_for_cancelled_job(job.type, job.payload_json)
                logger.info("Cancelled orphaned job %s on request.", job.id)
                continue

            heartbeat_at = job.heartbeat_at or job.updated_at
            if not startup and heartbeat_at > stale_cutoff:
                continue
            if _job_should_retry(job):
                Job.objects.filter(id=job.id, status="RUNNING").update(
                    status="RETRY",
                    next_run_at=now,
                    message="recovered",
                )
                release_storage_artifact_leases(job)
                # A worker that died mid-attempt is still a failed attempt;
                # without this the domain object keeps showing whatever error
                # an *older* run wrote while this one silently retries.
                reconcile_domain_objects_for_retrying_job(
                    job.type,
                    job.payload_json,
                    retrying_attempt_detail(
                        job.attempts,
                        job.max_attempts,
                        "The worker stopped before it finished.",
                    ),
                )
                logger.info("Requeued job %s after its worker stopped.", job.id)
            else:
                # Keep whatever the worker managed to record. A dying worker
                # often writes the one sentence the user needs -- "Model pack
                # 'quantem:mito' is not installed ... install it with ..." --
                # and overwriting that with "worker stopped" replaces an
                # actionable message with a shrug. The generic text is the
                # fallback for a worker that died with nothing to say.
                recorded = (job.error_traceback or "").strip()
                message = recorded or "worker stopped before job completion"
                Job.objects.filter(id=job.id, status="RUNNING").update(
                    status="FAILED",
                    finished_at=now,
                    message=message.splitlines()[-1][:500],
                    error_traceback=message,
                )
                # A dead worker cannot release what it acquired. Leaving its
                # storage leases ACTIVE bricked the segmentation for the whole
                # 6-hour lease TTL: every retry -- including after a clean app
                # restart -- failed with "Storage artifact is leased by another
                # active job", the labeling header kept showing the stale
                # "worker stopped" message, and no screen offers a delete, so
                # there was no user-space escape. Reproduced three times by
                # closing the app mid-inference.
                release_storage_artifact_leases(job)
                reconcile_domain_objects_for_failed_job(
                    job.type,
                    job.payload_json,
                    message,
                    # The worker is gone; nothing it recorded here belongs to
                    # an *older* attempt, so a stale FAILED stage must not
                    # keep the previous run's error over this one's.
                    supersede_stale_failure=True,
                )
                logger.warning("Failed job %s: %s", job.id, message)

    def _priority_order(self):
        return Case(
            When(priority="high", then=Value(0)),
            default=Value(1),
            output_field=IntegerField(),
        )

    def _queue_order(self):
        return Case(
            When(queue_name=QUEUE_P1_INTERACTIVE, then=Value(0)),
            When(queue_name=QUEUE_P2_UPLOAD, then=Value(1)),
            When(queue_name=QUEUE_P3_ROI, then=Value(2)),
            When(queue_name=QUEUE_P4_FULL, then=Value(3)),
            default=Value(4),
            output_field=IntegerField(),
        )

    def _get_ready_jobs(self):
        Job = _get_job_model()
        now = timezone.now()
        return (
            Job.objects.filter(status__in=["PENDING", "RETRY"], next_run_at__lte=now)
            .annotate(queue_rank=self._queue_order(), priority_rank=self._priority_order())
            .order_by("queue_rank", "priority_rank", "created_at")
        )

    def _lock_ready_jobs(self, queryset):
        connection = connections["default"]
        features = connection.features
        if getattr(features, "has_select_for_update_skip_locked", False):
            return queryset.select_for_update(skip_locked=True)
        if getattr(features, "has_select_for_update", False):
            return queryset.select_for_update()
        return queryset

    def _claim_next_ready_job(self):
        Job = _get_job_model()
        now = timezone.now()
        with transaction.atomic():
            queryset = self._lock_ready_jobs(self._get_ready_jobs())[:50]
            for job in queryset:
                if not self.runner.can_dispatch(job.resource_class, job.type):
                    continue
                claim = {
                    "status": "RUNNING",
                    "started_at": now,
                    "claimed_at": now,
                    "heartbeat_at": now,
                    "attempts": job.attempts + 1,
                    "message": "running",
                }
                if job.progress_units_total is not None:
                    # An attempt starts its tile walk from zero. Without this a
                    # retry inherits the previous attempt's count, so the row
                    # claims 19 tiles are done while the new attempt is on its
                    # first -- and the whole-image rollup adds that phantom
                    # progress up and then watches it go backwards when the
                    # writer catches up.
                    claim["progress_units_done"] = 0
                updated = Job.objects.filter(
                    id=job.id,
                    status__in=["PENDING", "RETRY"],
                    next_run_at__lte=now,
                ).update(**claim)
                if updated == 0:
                    continue
                job.refresh_from_db()
                return job
        return None

    def dispatch_ready(self) -> None:
        while True:
            job = self._claim_next_ready_job()
            if job is None:
                break
            # The claim above already set RUNNING/started_at/attempts+1 and
            # committed. If start_job raises -- no free worker slot, a spawn
            # failure, a daemonic-process error -- the row is RUNNING with
            # nobody running it, and the segmentation it holds is wedged until
            # the heartbeat goes stale minutes later. Hand it back instead.
            try:
                self.runner.start_job(str(job.id), job.resource_class, job.type)
            except Exception:
                logger.exception("Dispatch of job %s failed; releasing it.", job.id)
                self._release_undispatched_job(job)
                break

    def _release_undispatched_job(self, job) -> None:
        """Undo a claim whose worker never started."""
        Job = _get_job_model()
        now = timezone.now()
        if _job_should_retry(job):
            Job.objects.filter(id=job.id, status="RUNNING").update(
                status="RETRY",
                next_run_at=now,
                started_at=None,
                message="requeued: the worker could not be started",
            )
            return
        message = (
            "The job could not be handed to a worker process. This is a queue "
            "fault, not a problem with the image or the model; restarting "
            "QuantEM will clear it."
        )
        Job.objects.filter(id=job.id, status="RUNNING").update(
            status="FAILED",
            finished_at=now,
            message=message,
            error_traceback=message,
        )
        reconcile_domain_objects_for_failed_job(
            job.type,
            job.payload_json,
            message,
            # No worker ever ran, so nothing of this attempt was written; an
            # existing FAILED stage is an older attempt's stale conclusion.
            supersede_stale_failure=True,
        )

    def _reap_if_due(self) -> None:
        now = time.monotonic()
        if (now - self._last_reap_monotonic) < REAP_INTERVAL_SECONDS:
            return
        self._last_reap_monotonic = now
        self._recover_orphaned_jobs()

    def sweep_abandoned_uploads(self) -> None:
        """Free upload bytes no asset owns and no request is still writing.

        The scheduler thread is where this belongs: it is the one background
        loop the server always runs, it already owns start-up recovery, and the
        thing being cleaned up is left behind by exactly the events it handles
        -- a request that failed, or a process that was killed mid-import.

        Guarded on its own rather than relying on ``run_forever``'s guard: a
        failure here is housekeeping, and must not cost the tick its dispatch.
        """
        from quantem.assets.upload_staging import sweep_abandoned_uploads

        try:
            result = sweep_abandoned_uploads()
        except Exception:
            logger.exception("The abandoned-upload sweep failed; continuing.")
            return
        if result.removed:
            logger.info("Upload sweep: %s.", result.summary())

    def _sweep_uploads_if_due(self) -> None:
        now = time.monotonic()
        if (now - self._last_upload_sweep_monotonic) < UPLOAD_SWEEP_INTERVAL_SECONDS:
            return
        self._last_upload_sweep_monotonic = now
        self.sweep_abandoned_uploads()

    def tick(self) -> None:
        self.runner.poll()
        # Reaping runs on every tick cycle, not just at startup: a worker can
        # die at any point in a session, and until its job is cleared the
        # segmentation it holds cannot be run again at all.
        self._reap_if_due()
        self._sweep_uploads_if_due()
        self.dispatch_ready()

    def run_forever(self) -> None:
        """Poll forever. **Never** let one bad tick kill the thread.

        This runs on a daemon thread inside the app process. If it raises, the
        queue stops for the rest of the session and nothing says so — every
        upload just hangs. So each tick is guarded, failures are logged and
        backed off, and the loop continues.
        """
        consecutive_failures = 0
        while True:
            try:
                if not self._database_ready:
                    if _wait_for_database():
                        # Startup: every RUNNING row is an orphan by definition
                        # (no worker can predate this scheduler), so the reap is
                        # not gated on the heartbeat window here.
                        self._recover_orphaned_jobs(startup=True)
                        self._last_reap_monotonic = time.monotonic()
                        # Start-up is when the previous session's wreckage is
                        # still on disk: the bytes a killed server left mid
                        # import survive a restart, and nothing else looks.
                        self.sweep_abandoned_uploads()
                        self._last_upload_sweep_monotonic = time.monotonic()
                        self._database_ready = True
                        logger.info("Job scheduler: database ready, dispatching.")
                    else:
                        time.sleep(self.poll_interval_seconds)
                        continue
                self.tick()
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                # The database can go away mid-session (a file moved, a lock
                # held too long). Re-probe rather than assuming it is still there.
                self._database_ready = False
                logger.exception(
                    "Job scheduler tick failed (%d in a row); continuing.",
                    consecutive_failures,
                )
                time.sleep(min(30.0, self.poll_interval_seconds * 2**min(consecutive_failures, 5)))
                continue
            time.sleep(self.poll_interval_seconds)
