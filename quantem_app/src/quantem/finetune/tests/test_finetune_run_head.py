"""A scoped fine-tune end to end, on the convolutional stand-in.

The released packs are bare ``state_dict``s that nothing in the app can turn
into a module yet, so this uses the same three-layer stand-in as
``test_adapt_job_head.py``. That is enough for everything this round adds: the
scope decides the crops, the fold plan decides the rounds, every round trains
from the released weights rather than from the round before it, the steps are
counted across all of them, and the cross-validated result carries per-image
rows as well as an average.

Marked ``slow``: it needs torch and runs several real (tiny) training loops.
"""

from __future__ import annotations

from unittest import mock

import pytest
from django.test import TestCase

torch = pytest.importorskip("torch", reason="fine-tuning needs torch")

from quantem.core.config import MODELS_DIR  # noqa: E402
from quantem.finetune.job import adapter_job  # noqa: E402
from quantem.finetune.models import (  # noqa: E402
    TRAINING_MODE_HOLDOUT_1,
    TRAINING_MODE_USE_ALL,
    Adapter,
)
from quantem.finetune.storage import (  # noqa: E402
    adapter_head_path,
    staged_head_path,
)
from quantem.finetune.tests.fixtures import (  # noqa: E402
    FakeCancel,
    FakeReporter,
    annotated_segmentation,
    done_roi,
)
from quantem.inference import engine  # noqa: E402
from quantem.inference.specs import MODEL_SPECS  # noqa: E402
from quantem.segmentation.type_service import (  # noqa: E402
    get_or_create_mitochondria_type,
)

pytestmark = pytest.mark.slow

BIG = 1024
BIG_ROI = (20, 20, 1000, 1000)
BIG_OBJECT = (300, 300, 600, 600)
STEPS = 2


class StandInModel(torch.nn.Module):
    """Same submodule names as a released pack, three orders of magnitude smaller."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.neck = torch.nn.Conv2d(4, 4, 1)
        self.decoder = torch.nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.decoder(self.neck(self.encoder(x)))


def _loader(pack_id: str, device: str | None = None):
    """A deterministic stand-in for ``engine.load_model``.

    Seeded, so every call returns the *same* base weights. That is what makes
    the per-round reload meaningful: without it "fresh weights each round" and
    "different weights each round" would look identical.
    """
    torch.manual_seed(1234)
    return engine.LoadedModel(spec=MODEL_SPECS[pack_id], device="cpu", module=StandInModel())


def _big(name: str, **kwargs):
    return annotated_segmentation(name, size=BIG, roi=BIG_ROI, obj=BIG_OBJECT, **kwargs)


class ScopedRunTests(TestCase):
    def setUp(self):
        self.first = _big("scoped_one.tif")
        self.second = _big("scoped_two.tif")
        self.asset_ids = [str(self.first.asset_id), str(self.second.asset_id)]
        self.adapter = Adapter.objects.create(
            base_model="quantem:mito",
            name="Scoped",
            mode="head",
            segmentation_type=get_or_create_mitochondria_type(),
        )
        self.adapter.scope_assets.set(self.asset_ids)

    def _payload(self, **overrides):
        payload = {
            "adapter_id": str(self.adapter.id),
            "segmentation_type_id": str(get_or_create_mitochondria_type().id),
            "asset_ids": self.asset_ids,
            "base_model": "quantem:mito",
            "training_mode": TRAINING_MODE_USE_ALL,
            "steps": STEPS,
            "lr": 0.01,
            "seed": 0,
            "name": "Scoped",
        }
        payload.update(overrides)
        return payload

    def _run(self, **overrides):
        reporter = FakeReporter()
        with mock.patch.object(engine, "load_model", side_effect=_loader):
            result = adapter_job(self._payload(**overrides), reporter, FakeCancel())
        return result, reporter

    # -- use all ------------------------------------------------------------

    def test_use_all_trains_on_every_area_in_one_round(self):
        result, reporter = self._run()
        assert result["rounds"] == 1
        assert result["total_steps"] == STEPS
        assert result["split_mode"] == "no-heldout"
        assert result["annotation_count"] == 2
        assert result["cv_results"] == {}
        # Both images' areas are in the training set.
        assert len(result["train_crop_names"]) == 2
        assert result["heldout_crop_names"] == []
        assert [scope.total for scope in reporter.scopes] == [STEPS]

    def test_the_head_lands_at_the_live_path_and_the_staged_one_is_gone(self):
        result, _reporter = self._run()
        live = adapter_head_path(str(self.adapter.id))
        assert live.exists()
        assert not staged_head_path(str(self.adapter.id)).exists()
        assert (MODELS_DIR / result["head_path"]).resolve() == live.resolve()
        assert Adapter.objects.get(id=self.adapter.id).head_path == result["head_path"]

    # -- hold out one -------------------------------------------------------

    def test_holding_one_image_out_gives_an_image_disjoint_score(self):
        result, _reporter = self._run(training_mode=TRAINING_MODE_HOLDOUT_1)
        assert result["rounds"] == 1
        assert result["split_mode"] == "image-disjoint"
        assert len(result["train_crop_names"]) == 1
        assert len(result["heldout_crop_names"]) == 1
        assert result["sweep"]["heldout_dice_at_calibrated"] is not None

    # -- cross-validation ---------------------------------------------------

    def test_cross_validation_reports_every_fold_the_mean_and_each_image(self):
        result, reporter = self._run(training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True)
        assert result["rounds"] == 2
        assert result["total_steps"] == 2 * STEPS

        cv = result["cv_results"]
        assert [fold["fold"] for fold in cv["folds"]] == [0, 1]
        assert sorted(str(f["held_out_asset_id"]) for f in cv["folds"]) == sorted(self.asset_ids)
        for fold in cv["folds"]:
            assert fold["n_tiles"] >= 1
            # None is allowed and means "undefined here"; a number must be a
            # number, never a zero standing in for one.
            assert fold["dice"] is None or 0.0 <= fold["dice"] <= 1.0
            assert fold["iou"] is None or 0.0 <= fold["iou"] <= 1.0

        assert set(cv["mean"]) == {"dice", "iou"}
        # Per-image results are required by R13, not optional.
        assert len(cv["per_image"]) == 2
        assert sorted(row["asset_id"] for row in cv["per_image"]) == sorted(self.asset_ids)
        assert all(row["name"] for row in cv["per_image"])

    def test_the_bar_counts_steps_across_every_round(self):
        _result, reporter = self._run(training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True)
        assert len(reporter.scopes) == 2
        assert [scope.detail["round"] for scope in reporter.scopes] == [1, 2]
        assert {scope.detail["total_rounds"] for scope in reporter.scopes} == {2}
        # Each round counts its own steps; the offset into the whole run is the
        # window's job, which the queue's reporter applies.
        assert [scope.max_done for scope in reporter.scopes] == [STEPS, STEPS]
        assert {scope.label for scope in reporter.scopes} == {"step"}

    def test_the_result_is_written_onto_the_adapter_row(self):
        self._run(training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True)
        adapter = Adapter.objects.get(id=self.adapter.id)
        assert adapter.status == "SUCCESS"
        assert adapter.cv_results["folds"]
        assert adapter.calibrated_threshold is not None

    def test_a_thin_average_says_it_is_thin(self):
        self._run(training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True)
        adapter = Adapter.objects.get(id=self.adapter.id)
        caveats = adapter.caveats()
        assert any("2 rounds" in note for note in caveats)
        assert any("weak estimate" in note for note in caveats)

    def test_the_run_endpoint_serves_the_cross_validated_result(self):
        self._run(training_mode=TRAINING_MODE_HOLDOUT_1, cv_benchmark=True)
        response = self.client.get(f"/api/finetune/runs/{self.adapter.id}/")
        assert response.status_code == 200
        body = response.json()
        assert body["cv_results"]["per_image"]
        assert body["asset_count"] == 2
        assert body["training_mode"] == TRAINING_MODE_HOLDOUT_1
        assert body["cv_benchmark"] is True


class RealQueueProgressTests(TestCase):
    """The same run, driven through the queue's own reporter.

    The fake above proves the job asks for the right scopes; only this proves
    the numbers reach a row. It is the whole point of the feature: a bar that is
    monotone across rounds, a denominator that is the whole run, and a progress
    endpoint whose percentage is derived from those two rather than from a
    separate opinion.
    """

    def setUp(self):
        self.first = _big("queued_one.tif")
        self.second = _big("queued_two.tif")
        self.asset_ids = [str(self.first.asset_id), str(self.second.asset_id)]
        self.adapter = Adapter.objects.create(
            base_model="quantem:mito",
            name="Queued",
            mode="head",
            segmentation_type=get_or_create_mitochondria_type(),
            training_mode=TRAINING_MODE_HOLDOUT_1,
            cv_benchmark=True,
        )
        self.adapter.scope_assets.set(self.asset_ids)

    def test_the_row_ends_at_the_full_step_count_over_both_rounds(self):
        from quantem.jobs.constants import JOB_TYPE_TRAIN_ORGANELLE_ADAPTER
        from quantem.jobs.models import Job
        from quantem.jobs.reporter import CancelToken, JobReporter

        payload = {
            "adapter_id": str(self.adapter.id),
            "segmentation_type_id": str(get_or_create_mitochondria_type().id),
            "asset_ids": self.asset_ids,
            "base_model": "quantem:mito",
            "training_mode": TRAINING_MODE_HOLDOUT_1,
            "cv_benchmark": True,
            "planned_rounds": 2,
            "steps": STEPS,
            "lr": 0.01,
            "name": "Queued",
        }
        job = Job.enqueue(
            job_type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER, payload=payload, max_attempts=1
        )
        # The denominator is on the row before a worker has touched it.
        assert job.progress_units_total == 2 * STEPS
        assert job.progress_unit_label == "step"
        Job.objects.filter(id=job.id).update(status="RUNNING")

        reporter = JobReporter(str(job.id))
        try:
            with mock.patch.object(engine, "load_model", side_effect=_loader):
                adapter_job(payload, reporter, CancelToken(str(job.id)))
        finally:
            reporter.deactivate()

        job.refresh_from_db()
        assert job.progress_units_done == 2 * STEPS
        assert job.progress_units_total == 2 * STEPS
        assert job.progress_detail_json["round"] == 2
        assert job.progress_detail_json["total_rounds"] == 2

        response = self.client.get(f"/api/finetune/runs/{self.adapter.id}/progress/")
        body = response.json()
        assert body["status"] == "SUCCESS"
        assert body["percent"] == 100.0
        assert (body["round"], body["total_rounds"]) == (2, 2)
        assert body["error"] == ""


class DoneRoiTrainingTests(TestCase):
    """A ROI ticked as done reaches the trainer, not only the counter."""

    def test_a_run_scoped_to_a_done_roi_alone_trains_on_it(self):
        segmentation = _big("done_only.tif", with_roi=False)
        done_roi(segmentation, (20, 20, 980, 980))
        adapter = Adapter.objects.create(
            base_model="quantem:mito",
            name="From a ticked region",
            mode="head",
            segmentation_type=get_or_create_mitochondria_type(),
        )
        payload = {
            "adapter_id": str(adapter.id),
            "segmentation_type_id": str(get_or_create_mitochondria_type().id),
            "asset_ids": [str(segmentation.asset_id)],
            "base_model": "quantem:mito",
            "training_mode": TRAINING_MODE_USE_ALL,
            "steps": STEPS,
            "lr": 0.01,
        }
        with mock.patch.object(engine, "load_model", side_effect=_loader):
            result = adapter_job(payload, FakeReporter(), FakeCancel())
        assert result["annotation_count"] == 1
        assert result["tile_count"] >= 1
        assert result["steps"] == STEPS


class FailedOverwriteTests(TestCase):
    """A failed overwrite must leave the previous weights loadable."""

    def test_the_live_head_survives_a_run_that_dies_while_saving(self):
        segmentation = _big("overwrite.tif")
        adapter = Adapter.objects.create(
            base_model="quantem:mito",
            name="Kept",
            mode="head",
            segmentation_type=get_or_create_mitochondria_type(),
            status="SUCCESS",
        )
        live = adapter_head_path(str(adapter.id))
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_bytes(b"the version already in use")
        adapter.head_path = f"adapters/{adapter.id}/head.pt"
        adapter.save(update_fields=["head_path"])

        payload = {
            "adapter_id": str(adapter.id),
            "segmentation_type_id": str(get_or_create_mitochondria_type().id),
            "asset_ids": [str(segmentation.asset_id)],
            "base_model": "quantem:mito",
            "training_mode": TRAINING_MODE_USE_ALL,
            "steps": STEPS,
            "lr": 0.01,
            "overwrite": True,
        }
        boom = RuntimeError("the disk went away")
        with (
            mock.patch.object(engine, "load_model", side_effect=_loader),
            mock.patch("quantem.finetune.job.save_head", side_effect=boom),
        ):
            with pytest.raises(RuntimeError):
                adapter_job(payload, FakeReporter(), FakeCancel())

        assert live.read_bytes() == b"the version already in use"
        assert not staged_head_path(str(adapter.id)).exists()
        adapter.refresh_from_db()
        assert adapter.status == "FAILED"
        assert "untouched and still in use" in adapter.error
        assert adapter.head_path == f"adapters/{adapter.id}/head.pt"
