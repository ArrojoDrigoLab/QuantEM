from unittest.mock import patch

import pytest
from rest_framework.test import APIClient

from quantem.finetune.models import Adapter
from quantem.jobs.models import Job
from quantem.library.models import Experiment
from quantem.segmentation.models import ImageSegmentation, ModelApplicationBatch
from quantem.segmentation.type_service import get_or_create_mitochondria_type
from quantem.testing import create_small_test_image

pytestmark = pytest.mark.django_db


def test_selected_model_and_evaluated_threshold_reach_the_worker(application):
    _, _, _, data, url = application
    with patch(
        "quantem.segmentation.api_views.bulk_runs._build_plan",
        return_value={"organelles": [], "tiles_total": 1},
    ):
        response = APIClient().post(
            url,
            {**data, "source_models": {"mito": "omniem:mito"}, "threshold": 0.73},
            format="json",
        )
    assert response.status_code == 202, response.data
    leg = Job.objects.get().payload_json["legs"][0]
    assert leg["source_model"] == "omniem:mito"
    assert leg["threshold"] == 0.73


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "unknown"])
def test_invalid_application_threshold_does_not_queue_work(application, value):
    _, _, _, data, url = application
    response = APIClient().post(url, {**data, "threshold": value}, format="json")
    assert response.status_code == 400
    assert not Job.objects.exists()


@pytest.fixture
def application():
    first = create_small_test_image("First field", width=64, height=64).asset
    second = create_small_test_image("Second field", width=64, height=64).asset
    second.experiment = first.experiment
    second.preprocess_stage = "PENDING"
    second.save()
    first.preprocess_stage = "DONE"
    first.save()
    seg = ImageSegmentation.objects.create(
        asset=first, segmentation_type=get_or_create_mitochondria_type(), status_stage="COMPLETED"
    )
    data = {"asset_ids": [str(first.id), str(second.id)], "organelles": ["mito"]}
    return first, second, seg, data, f"/api/experiments/{first.experiment_id}/model-runs/"


def test_preview_and_partial_submission_are_durable_without_unlocking(application):
    first, second, seg, data, url = application
    client = APIClient()
    with patch(
        "quantem.segmentation.api_views.bulk_runs._build_plan",
        return_value={"organelles": [], "tiles_total": 1},
    ):
        preview = client.post(url, {**data, "preview": True}, format="json")
        assert preview.status_code == 200
        assert [row["ready"] for row in preview.data["items"]] == [True, False]
        assert not Job.objects.exists()
        response = client.post(url, data, format="json")
        assert response.status_code == 202, response.data
        job = Job.objects.get()
        assert job.payload_json["force_recompute_prob_maps"] is True
        assert job.payload_json["legs"][0]["adapter_id"] == "default"
        assert job.payload_json["legs"][0]["segmentation_id"] == str(seg.id)
        duplicate = client.post(url, data, format="json")
        assert duplicate.status_code == 400
    assert ModelApplicationBatch.objects.count() == 1
    reopened = APIClient().get(f"/api/model-runs/{response.data['id']}/")
    assert reopened.data["items"][1]["status"] == "SKIPPED"
    assert reopened.data["items"][0]["job_id"] == str(job.id)
    Job.objects.filter(pk=job.id).update(status="FAILED", message="Worker stopped")
    assert APIClient().get(url).data[0]["items"][0]["status"] == "FAILED"
    retried = client.post(f"/api/jobs/{job.id}/retry/", {}, format="json")
    assert retried.status_code == 200, retried.data
    assert APIClient().get(url).data[0]["items"][0]["status"] in {"PENDING", "RETRY"}


def test_saved_model_is_explicit_and_scoped(application):
    first, _, seg, data, url = application
    adapter = Adapter.objects.create(
        experiment=first.experiment,
        segmentation_type=seg.segmentation_type,
        base_model="quantem:mito",
        status="SUCCESS",
        name="Improved",
    )
    with patch(
        "quantem.segmentation.api_views.bulk_runs._build_plan",
        return_value={"organelles": [], "tiles_total": 1},
    ):
        response = APIClient().post(url, {**data, "adapter_id": str(adapter.id)}, format="json")
    assert response.status_code == 202, response.data
    assert Job.objects.get().payload_json["legs"][0]["adapter_id"] == str(adapter.id)
    other = Experiment.objects.create(name="Unrelated experiment")
    assert (
        APIClient()
        .post(f"/api/experiments/{other.id}/model-runs/", data, format="json")
        .status_code
        == 400
    )
    assert (
        APIClient().post(url, {**data, "adapter_id": "invalid"}, format="json").status_code == 400
    )


def test_unavailable_image_is_skipped_before_queueing(application):
    _, _, _, data, url = application
    with patch(
        "quantem.segmentation.api_views.bulk_runs._build_plan", side_effect=OSError("missing")
    ):
        response = APIClient().post(url, {**data, "preview": True}, format="json")
    assert response.status_code == 200
    assert all(not row["ready"] for row in response.data["items"])
    assert not Job.objects.exists()
