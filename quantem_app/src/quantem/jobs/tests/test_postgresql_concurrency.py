from __future__ import annotations

import threading
import uuid
from queue import Queue
from types import SimpleNamespace

import pytest
from django.db import close_old_connections, connection, connections

from quantem.core.local_storage import StoragePath
from quantem.jobs.constants import JOB_TYPE_RUN_SEGMENTATION_FULL
from quantem.jobs.models import Job, StorageArtifactLease
from quantem.jobs.scheduler import JobScheduler
from quantem.jobs.storage_leases import StorageLeaseConflict, acquire_storage_artifact_leases

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(connection.vendor != "postgresql", reason="PostgreSQL integration test"),
]


def _join(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert all(not thread.is_alive() for thread in threads)


def test_concurrent_jobs_join_one_batch_without_duplicate_positions():
    barrier = threading.Barrier(2)
    output: Queue[object] = Queue()
    asset_id = str(uuid.uuid4())

    def enqueue() -> None:
        close_old_connections()
        try:
            barrier.wait(timeout=10)
            output.put(
                Job.enqueue(
                    job_type=JOB_TYPE_RUN_SEGMENTATION_FULL,
                    payload={"asset_id": asset_id, "segmentation_id": str(uuid.uuid4())},
                    resource_class="cpu",
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            output.put(exc)
        finally:
            connections.close_all()

    _join([threading.Thread(target=enqueue) for _ in range(2)])
    rows = [output.get_nowait(), output.get_nowait()]
    assert not [row for row in rows if isinstance(row, Exception)]
    assert len({row.batch_id for row in rows}) == 1
    assert sorted(row.batch_seq for row in rows) == [0, 1]


def test_concurrent_storage_lease_has_one_winner():
    jobs = [Job.objects.create(type="LEASE_TEST", payload_json={}) for _ in range(2)]
    barrier = threading.Barrier(2)
    output: Queue[object] = Queue()

    def acquire(job_id) -> None:
        close_old_connections()
        try:
            barrier.wait(timeout=10)
            job = Job.objects.get(pk=job_id)
            try:
                acquire_storage_artifact_leases(
                    job,
                    [StoragePath("data/test/shared.bin", lease_required=True)],
                )
            except StorageLeaseConflict:
                output.put("conflict")
            else:
                output.put("acquired")
        except Exception as exc:  # pragma: no cover - asserted below
            output.put(exc)
        finally:
            connections.close_all()

    _join([threading.Thread(target=acquire, args=(job.pk,)) for job in jobs])
    outcomes = [output.get_nowait(), output.get_nowait()]
    assert not [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert sorted(outcomes) == ["acquired", "conflict"]
    assert StorageArtifactLease.objects.filter(
        artifact_path="data/test/shared.bin",
        status=StorageArtifactLease.STATUS_ACTIVE,
    ).count() == 1


def test_concurrent_schedulers_claim_each_job_once():
    queued = [Job.objects.create(type="CLAIM_TEST", payload_json={}) for _ in range(2)]
    barrier = threading.Barrier(2)
    output: Queue[object] = Queue()

    def claim() -> None:
        close_old_connections()
        try:
            scheduler = JobScheduler()
            scheduler.runner = SimpleNamespace(can_dispatch=lambda *_: True)
            barrier.wait(timeout=10)
            output.put(scheduler._claim_next_ready_job())
        except Exception as exc:  # pragma: no cover - asserted below
            output.put(exc)
        finally:
            connections.close_all()

    _join([threading.Thread(target=claim) for _ in range(2)])
    claimed = [output.get_nowait(), output.get_nowait()]
    assert not [row for row in claimed if isinstance(row, Exception)]
    assert {row.pk for row in claimed} == {row.pk for row in queued}
    assert all(row.status == "RUNNING" for row in claimed)
