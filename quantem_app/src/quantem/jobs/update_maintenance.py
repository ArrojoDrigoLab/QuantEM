"""Coordinate the brief no-new-work window before a desktop update applies.

QuantEM has one local SQLite database and one server process.  The frontend
downloads an application update while normal work continues, then asks this
module to acquire an *apply* lock.  Acquiring it and checking that the queue is
empty happen in the same SQLite transaction.  That closes the race where a new
job could arrive between a client-side queue poll and the application restart.

The lock is intentionally short lived.  It is not a maintenance mode for a
long-running server: it exists only while the updater is about to replace the
application, and startup clears a lock left by an interrupted update.
"""

from __future__ import annotations

from dataclasses import dataclass

from django.apps import apps
from django.db import DatabaseError, OperationalError, transaction
from django.utils import timezone


class UpdateApplyInProgress(RuntimeError):
    """Raised when a request tries to enqueue work during an app restart."""

    message = "QuantEM is applying an update. It will reopen shortly; then try again."

    def __init__(self) -> None:
        super().__init__(self.message)


@dataclass(frozen=True)
class ApplyLockResult:
    """The outcome of an atomic update-apply lock attempt."""

    ready: bool
    open_jobs: int
    reason: str | None = None


def _maintenance_model():
    return apps.get_model("jobs", "UpdateMaintenance")


def _job_model():
    return apps.get_model("jobs", "Job")


def assert_job_submission_allowed() -> None:
    """Lock the update fence before inserting a job.

    Call this *inside the transaction that creates the job*.  The same locked
    singleton is used by :func:`try_acquire_update_apply_lock`, so a submission
    that reaches this point first is visible to the updater's open-job count,
    and one that reaches it second observes ``APPLYING``.  Checking a bare
    boolean outside the insert transaction would leave a restart race.
    """

    Maintenance = _maintenance_model()
    maintenance, _ = Maintenance.objects.select_for_update().get_or_create(
        pk=Maintenance.SINGLETON_ID
    )
    if maintenance.state == Maintenance.STATE_APPLYING:
        raise UpdateApplyInProgress()


def is_update_apply_locked() -> bool:
    """Whether the server has fenced new jobs for an imminent restart.

    During a first launch of an older database the new table may not yet exist;
    in that narrow migration interval there cannot be an updater request, so
    treating the lock as absent is the safe compatibility behavior.
    """

    try:
        Maintenance = _maintenance_model()
        return Maintenance.objects.filter(
            pk=Maintenance.SINGLETON_ID,
            state=Maintenance.STATE_APPLYING,
        ).exists()
    except (DatabaseError, OperationalError):
        return False


def try_acquire_update_apply_lock() -> ApplyLockResult:
    """Fence new jobs iff no queued, retrying, or running job remains."""

    Maintenance = _maintenance_model()
    Job = _job_model()
    with transaction.atomic():
        maintenance, _ = Maintenance.objects.select_for_update().get_or_create(
            pk=Maintenance.SINGLETON_ID
        )
        if maintenance.state == Maintenance.STATE_APPLYING:
            return ApplyLockResult(False, 0, "already_applying")

        open_jobs = Job.objects.filter(status__in=Job.OPEN_STATUSES).count()
        if open_jobs:
            return ApplyLockResult(False, open_jobs, "jobs_running")

        maintenance.state = Maintenance.STATE_APPLYING
        maintenance.acquired_at = timezone.now()
        maintenance.save(update_fields=["state", "acquired_at", "updated_at"])
        return ApplyLockResult(True, 0)


def release_update_apply_lock() -> None:
    """Re-open submissions after an updater error before the app exits."""

    Maintenance = _maintenance_model()
    Maintenance.objects.filter(pk=Maintenance.SINGLETON_ID).update(
        state=Maintenance.STATE_IDLE,
        acquired_at=None,
    )


def clear_stale_update_apply_lock() -> bool:
    """Clear an apply lock inherited from a process that did not restart cleanly."""

    Maintenance = _maintenance_model()
    changed = Maintenance.objects.filter(
        pk=Maintenance.SINGLETON_ID,
        state=Maintenance.STATE_APPLYING,
    ).update(state=Maintenance.STATE_IDLE, acquired_at=None)
    return bool(changed)
