from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from rest_framework.test import APIClient

from quantem.finetune.models import Adapter
from quantem.finetune.tests.fixtures import annotated_segmentation
from quantem.jobs.handlers.evaluation import handle_evaluate_model
from quantem.jobs.models import Job
from quantem.segmentation.result_reporting import InlineCancellation, InlineReporter

pytestmark = pytest.mark.django_db


@pytest.fixture
def evaluation():
    seg = annotated_segmentation("Evaluation field", with_prob=False)
    data = {
        "segmentation_type": str(seg.segmentation_type_id),
        "asset_ids": [str(seg.asset_id)],
        "base_model": "quantem:mito",
    }
    return seg, data


def test_training_overlap_is_flagged_but_permitted_and_evaluation_does_not_mutate(evaluation):
    seg, data = evaluation
    adapter = Adapter.objects.create(
        experiment=seg.asset.experiment,
        segmentation_type=seg.segmentation_type,
        base_model="quantem:mito",
        status="SUCCESS",
        training_exposure=[str(seg.asset_id)],
        calibrated_threshold=0.6,
    )
    data["adapter_id"] = str(adapter.id)
    client = APIClient()
    preview = client.post("/api/finetune/evaluations/", {**data, "preview": True}, format="json")
    assert preview.status_code == 200, preview.data
    assert preview.data["rois"][0]["training_overlap"] is True
    assert preview.data["threshold"] == 0.6
    expected = list(seg.segments.values_list("id", "geometry_wkb", "label_state"))
    response = client.post(
        "/api/finetune/evaluations/",
        {**data, "roi_ids": [row["id"] for row in preview.data["rois"]]},
        format="json",
    )
    assert response.status_code == 202, response.data
    assert response.data["results"] is None
    job = Job.objects.get()
    assert job.resource_class == "gpu"
    with (
        patch("quantem.jobs.handlers.evaluation.engine.load_model"),
        patch(
            "quantem.jobs.handlers.evaluation.engine.predict_region",
            side_effect=lambda model, image, **kw: SimpleNamespace(
                prob=np.ones(image.shape[:2]), context=None
            ),
        ),
        patch(
            "quantem.jobs.handlers.evaluation.resample.probability_to_native_uint8",
            side_effect=lambda prob, context: (prob * 255).astype(np.uint8),
        ),
    ):
        result = handle_evaluate_model(job.payload_json, InlineReporter(), InlineCancellation())
    assert result["training_overlap"] is True
    assert 0 < result["mean_dice"] < 1
    assert list(seg.segments.values_list("id", "geometry_wkb", "label_state")) == expected
    assert not seg.probability_maps.exists()
    assert Adapter.objects.count() == 1
    Job.objects.filter(pk=job.pk).update(status="SUCCESS", result_json=result)
    reopened = client.get(f"/api/finetune/evaluations/{job.id}/")
    assert reopened.data["results"]["per_roi"] == result["per_roi"]


def test_changed_annotations_refuse_stale_evaluation(evaluation):
    seg, data = evaluation
    client = APIClient()
    preview = client.post(
        "/api/finetune/evaluations/", {**data, "preview": True}, format="json"
    ).data
    response = client.post(
        "/api/finetune/evaluations/",
        {**data, "roi_ids": [row["id"] for row in preview["rois"]]},
        format="json",
    )
    assert response.status_code == 202
    seg.completed_rois.all().delete()
    with patch("quantem.jobs.handlers.evaluation.engine.load_model") as load:
        with pytest.raises(ValueError, match="annotations changed"):
            handle_evaluate_model(
                Job.objects.get().payload_json, InlineReporter(), InlineCancellation()
            )
        load.assert_not_called()


def test_default_model_preview_needs_no_stored_prediction(evaluation):
    _, data = evaluation
    response = APIClient().post(
        "/api/finetune/evaluations/", {**data, "preview": True}, format="json"
    )
    assert response.status_code == 200
    assert response.data["rois"]
    assert not response.data["rois"][0]["training_overlap"]
    assert not Job.objects.exists()
