"""Concurrent submissions use the same atomic queue contract as single clicks."""

import threading
from queue import Queue
from unittest.mock import patch

import pytest
from django.db import close_old_connections, connection, connections
from rest_framework.test import APIClient

from quantem.analysis.batches import start_batch
from quantem.analysis.models import AnalysisRun
from quantem.jobs.models import Job
from quantem.segmentation.models import (
    ImageSegmentation,
    ModelApplicationBatch,
    ProbabilityMap,
    SavedSegmentationResult,
)
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(connection.vendor != "postgresql", reason="PostgreSQL integration test"),
]


def _submit_together(url, payload):
    barrier = threading.Barrier(2)
    output = Queue()

    def submit():
        close_old_connections()
        try:
            barrier.wait(timeout=10)
            output.put(APIClient().post(url, payload, format="json").status_code)
        except Exception as exc:
            output.put(exc)
        finally:
            connections.close_all()

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert all(not thread.is_alive() for thread in threads)
    outcomes = [output.get_nowait() for _ in threads]
    assert not [item for item in outcomes if isinstance(item, Exception)], outcomes
    return sorted(outcomes)


@pytest.fixture
def segmentation():
    image = create_small_test_image("Concurrent bulk workflow", width=32, height=32)
    image.asset.preprocess_stage = "DONE"
    image.asset.save(update_fields=["preprocess_stage"])
    return ImageSegmentation.objects.create(
        asset=image.asset,
        segmentation_type=get_or_create_mitochondria_type(),
        manual_full_image=True,
    )


def test_concurrent_model_application_submissions_queue_only_once(segmentation):
    with patch(
        "quantem.segmentation.api_views.bulk_runs._build_plan",
        return_value={"organelles": [], "tiles_total": 1},
    ):
        outcomes = _submit_together(
            f"/api/experiments/{segmentation.asset.experiment_id}/model-runs/",
            {"asset_ids": [str(segmentation.asset_id)], "organelles": ["mito"]},
        )
    assert outcomes == [202, 400]
    assert ModelApplicationBatch.objects.count() == 1
    assert Job.objects.count() == 1


def test_concurrent_model_confirmation_keeps_old_outputs_until_worker_succeeds(segmentation):
    prediction = ProbabilityMap.objects.create(
        segmentation=segmentation, file_path="predictions/concurrent-test.npy"
    )
    old = SavedSegmentationResult.objects.create(
        segmentation=segmentation,
        name="Old",
        full_image=True,
        is_canonical=True,
        threshold=0.5,
        probability_map=prediction,
    )
    candidate = SavedSegmentationResult.objects.create(
        segmentation=segmentation,
        name="Candidate",
        full_image=True,
        threshold=0.7,
        probability_map=prediction,
    )
    assert _submit_together(
        f"/api/segmentations/{segmentation.pk}/results/",
        {"result_id": str(candidate.pk), "delete_result_ids": [str(old.pk)]},
    ) == [202, 409]
    old.refresh_from_db()
    candidate.refresh_from_db()
    assert old.is_canonical and old.deleted_at is None
    assert not candidate.is_canonical and candidate.deleted_at is None
    assert Job.objects.count() == 1


def test_concurrent_analysis_retries_create_one_replacement_run(segmentation):
    batch = start_batch(
        segmentation.asset.experiment,
        {
            "asset_ids": [str(segmentation.asset_id)],
            "segmentation_type_ids": [str(segmentation.segmentation_type_id)],
        },
    )
    item = batch.items.get()
    old_run = item.run_id
    Job.objects.filter(pk=item.job_id).update(status="FAILED")
    AnalysisRun.objects.filter(pk=old_run).update(status="FAILED")
    assert _submit_together(f"/api/analysis-batches/{batch.pk}/", {"action": "retry_failed"}) == [
        200,
        200,
    ]
    item.refresh_from_db()
    assert item.run_id != old_run
    assert AnalysisRun.objects.filter(pk=old_run, status="FAILED").exists()
    assert AnalysisRun.objects.count() == 2
    assert Job.objects.count() == 2
