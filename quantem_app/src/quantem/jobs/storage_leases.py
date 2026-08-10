"""Local artifact-write leases.

Serializes concurrent writes to the same storage artifact path so two jobs
running at once on this node cannot clobber the same output. Single-node: there
is no cross-node ownership — a lease is simply held by one job at a time.
"""

from __future__ import annotations

import os
from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from quantem.core.local_storage import StorageError, StoragePath
from quantem.jobs.models import Job, StorageArtifactLease


class StorageLeaseConflict(StorageError):
    """Another active job holds the lease on an artifact this job needs.

    A distinct type (still a ``StorageError``, so every existing ``except``
    keeps working) because the *runner* needs to tell this apart from a real
    storage fault: after an unclean shutdown a dead worker's lease survives
    until the reaper releases it, the first retry hits this, and the bounded
    retry then heals it in seconds. That deserves one calm INFO line, not a
    full traceback in the log of a session that fixed itself.
    """


def _lease_ttl() -> timedelta:
    raw_value = str(os.environ.get("QUANTEM_STORAGE_LEASE_TTL_SECONDS", "21600")).strip()
    try:
        seconds = max(60, int(raw_value))
    except (TypeError, ValueError):
        seconds = 21600
    return timedelta(seconds=seconds)


def acquire_storage_artifact_leases(job: Job, paths: list[StoragePath]) -> None:
    if not paths:
        return
    now = timezone.now()
    expires_at = now + _lease_ttl()

    for path in sorted({item.relpath for item in paths if item.lease_required}):
        with transaction.atomic():
            lease = (
                StorageArtifactLease.objects.select_for_update()
                .filter(artifact_path=path)
                .first()
            )
            if lease is None:
                StorageArtifactLease.objects.create(
                    artifact_path=path,
                    job=job,
                    status=StorageArtifactLease.STATUS_ACTIVE,
                    acquired_at=now,
                    expires_at=expires_at,
                )
                continue

            is_active = lease.status == StorageArtifactLease.STATUS_ACTIVE
            is_expired = lease.expires_at <= now
            if is_active and lease.job_id == job.id:
                lease.expires_at = expires_at
                lease.save(update_fields=["expires_at"])
                continue
            if is_active and not is_expired:
                raise StorageLeaseConflict(
                    f"Storage artifact is leased by another active job: {path}"
                )

            lease.job = job
            lease.status = StorageArtifactLease.STATUS_ACTIVE
            lease.acquired_at = now
            lease.expires_at = expires_at
            lease.released_at = None
            lease.save(
                update_fields=["job", "status", "acquired_at", "expires_at", "released_at"]
            )


def release_storage_artifact_leases(job: Job) -> None:
    StorageArtifactLease.objects.filter(
        job=job,
        status=StorageArtifactLease.STATUS_ACTIVE,
    ).update(
        status=StorageArtifactLease.STATUS_RELEASED,
        released_at=timezone.now(),
    )
