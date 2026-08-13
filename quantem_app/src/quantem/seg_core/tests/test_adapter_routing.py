"""An applied adapter must change the run.

The bug these cover: ``POST /api/adapters/<id>/apply/`` stamped ``applied_at``
and nothing ever read it, so a calibrated threshold and a trained head were
fitted, verified, reported to the user -- and then every subsequent
segmentation ran the released model at 0.5 anyway.

No torch and no weights here. The threshold is a property of the segmenter, and
whether the run *uses* it is a property of
:func:`quantem.seg_core.db.inference.apply_active_adapter`; both are testable
without loading a ViT.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from django.test import TestCase, TransactionTestCase

from quantem.seg_core.db.inference import apply_active_adapter
from quantem.testing import create_mitochondria_segmentation, create_small_test_image

pytestmark = pytest.mark.django_db


def _adapter(segmentation, **overrides):
    """A successful, applied adapter on ``segmentation``."""
    from django.utils import timezone

    from quantem.finetune.models import STATUS_SUCCESS, Adapter

    fields = {
        "segmentation": segmentation,
        "base_model": "quantem:mito",
        "name": "mito @ liver",
        "status": STATUS_SUCCESS,
        "mode": "threshold_only",
        "calibrated_threshold": 0.31,
        "split_mode": "image-disjoint",
        "applied_at": timezone.now(),
    }
    fields.update(overrides)
    return Adapter.objects.create(**fields)


class _RecordingSegmenter:
    """A segmenter that records what was applied to it, and nothing else."""

    supports_adapters = True
    source_model = "quantem:mito"
    name = "mito"

    def __init__(self):
        self.applied: dict | None = None

    def apply_adapter(self, **kwargs):
        self.applied = kwargs


class ApplyActiveAdapterTests(TestCase):
    def setUp(self):
        image = create_small_test_image("adapter routing")
        self.segmentation = create_mitochondria_segmentation(image)
        self.segmenter = _RecordingSegmenter()
        self.details: list[str] = []

    def _apply(self):
        return apply_active_adapter(self.segmenter, self.segmentation, self.details.append)

    def test_no_adapter_leaves_the_segmenter_alone(self):
        assert self._apply() is None
        assert self.segmenter.applied is None

    def test_an_applied_adapter_reaches_the_segmenter(self):
        adapter = _adapter(self.segmentation)
        assert self._apply() == str(adapter.id)
        assert self.segmenter.applied == {
            "adapter_id": str(adapter.id),
            "base_model": "quantem:mito",
            "calibrated_threshold": 0.31,
            "head_file": None,
        }
        # The user is told, in the job's own progress feed, that this run is not
        # the published default.
        assert any("0.31" in line for line in self.details)

    def test_an_explicit_dataset_apply_adapter_routes_to_an_unscoped_target_image(self):
        adapter = _adapter(
            self.segmentation,
            segmentation_type=self.segmentation.segmentation_type,
        )
        other_image = create_small_test_image("dataset target")
        target = create_mitochondria_segmentation(other_image)
        adapter.applied_assets.add(target.asset)
        target_segmenter = _RecordingSegmenter()
        applied = apply_active_adapter(
            target_segmenter,
            target,
            adapter_id=str(adapter.id),
        )
        assert applied == str(adapter.id)
        assert target_segmenter.applied["adapter_id"] == str(adapter.id)

    def test_an_explicit_adapter_cannot_escape_its_apply_targets(self):
        adapter = _adapter(
            self.segmentation,
            segmentation_type=self.segmentation.segmentation_type,
        )
        other_image = create_small_test_image("not a dataset target")
        target = create_mitochondria_segmentation(other_image)

        with pytest.raises(ValueError, match="did not run the released model"):
            apply_active_adapter(
                _RecordingSegmenter(),
                target,
                adapter_id=str(adapter.id),
            )

    def test_an_explicit_missing_adapter_fails_instead_of_running_the_base_model(self):
        from uuid import uuid4

        with pytest.raises(ValueError, match="did not run the released model"):
            apply_active_adapter(
                self.segmenter,
                self.segmentation,
                adapter_id=str(uuid4()),
            )

    def test_an_unapplied_adapter_is_not_used(self):
        _adapter(self.segmentation, applied_at=None)
        assert self._apply() is None
        assert self.segmenter.applied is None

    def test_a_pending_adapter_is_not_used(self):
        _adapter(self.segmentation, status="PENDING")
        assert self._apply() is None
        assert self.segmenter.applied is None

    def test_the_most_recently_applied_adapter_wins(self):
        from django.utils import timezone

        older = _adapter(self.segmentation, calibrated_threshold=0.2)
        newer = _adapter(
            self.segmentation,
            calibrated_threshold=0.7,
            applied_at=timezone.now(),
        )
        assert self._apply() == str(newer.id)
        assert self.segmenter.applied["calibrated_threshold"] == 0.7
        assert str(older.id) != str(newer.id)

    def test_an_adapter_for_a_different_model_is_refused_and_reported(self):
        # A threshold calibrated on quantem:mito says nothing about where
        # omniem:mito's foreground starts. Refusing is the point.
        _adapter(self.segmentation, base_model="omniem:mito")
        assert self._apply() is None
        assert self.segmenter.applied is None
        assert any("omniem:mito" in line for line in self.details)

    def test_a_segmenter_without_adapter_support_is_not_handed_one(self):
        segmenter = SimpleNamespace(source_model="quantem:mito")
        _adapter(self.segmentation)
        assert apply_active_adapter(segmenter, self.segmentation) is None

    def test_a_stand_in_segmentation_is_not_queried(self):
        # run_inference_for_segmentation is driven with a SimpleNamespace by the
        # cache/streaming tests; an object with no row cannot have an adapter.
        assert apply_active_adapter(self.segmenter, SimpleNamespace(id="x")) is None


class SegmenterThresholdTests(TransactionTestCase):
    """The threshold an organelle run actually binarises at.

    ``TransactionTestCase`` rather than ``TestCase`` only because importing
    :mod:`quantem.inference.segmenter` pulls in torch; nothing here touches the
    database.
    """

    def test_apply_adapter_changes_the_threshold_the_run_uses(self):
        from quantem.inference.segmenter import DinoMitoSegmenter

        segmenter = DinoMitoSegmenter(source_model="quantem:mito")
        assert segmenter.fg_threshold == 0.5  # the published default

        segmenter.apply_adapter(
            adapter_id="a-1", base_model="quantem:mito", calibrated_threshold=0.31
        )

        assert segmenter.fg_threshold == 0.31
        assert segmenter.adapter_id == "a-1"
        # And it travels with the pixels, not just the log.
        meta = segmenter.get_probability_map_metadata("DINO")
        assert meta["threshold"] == 0.31
        assert meta["default_threshold"] == 0.5
        assert meta["adapter_id"] == "a-1"

    def test_the_applied_threshold_is_what_binarises_the_map(self):
        import numpy as np

        from quantem.inference.segmenter import DinoMitoSegmenter

        prob = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        segmenter = DinoMitoSegmenter(source_model="quantem:mito")

        assert segmenter._foreground_mask(prob).sum() == 2  # >= 0.5
        segmenter.apply_adapter(
            adapter_id="a-1", base_model="quantem:mito", calibrated_threshold=0.31
        )
        assert segmenter._foreground_mask(prob).sum() == 3  # >= 0.31

    def test_an_adapter_from_another_pack_is_refused(self):
        from quantem.inference.segmenter import DinoMitoSegmenter

        segmenter = DinoMitoSegmenter(source_model="quantem:mito")
        with pytest.raises(ValueError, match="omniem:mito"):
            segmenter.apply_adapter(
                adapter_id="a-1",
                base_model="omniem:mito",
                calibrated_threshold=0.31,
            )
        assert segmenter.fg_threshold == 0.5

    def test_an_adapted_head_is_loaded_instead_of_the_released_one(self):
        from unittest.mock import patch

        from quantem.inference.segmenter import DinoMitoSegmenter

        segmenter = DinoMitoSegmenter(source_model="quantem:mito")
        segmenter.apply_adapter(
            adapter_id="a-1",
            base_model="quantem:mito",
            calibrated_threshold=0.31,
            head_file="D:/nowhere/head.pt",
        )
        with (
            patch("quantem.inference.engine.load_adapted_model") as adapted,
            patch("quantem.inference.engine.load_model") as released,
        ):
            segmenter.load_models()

        released.assert_not_called()
        assert adapted.call_args.args[0] == "quantem:mito"
        assert adapted.call_args.args[1] == Path("D:/nowhere/head.pt")
        assert adapted.call_args.kwargs["adapter_id"] == "a-1"


class AdaptedRunTests(TestCase):
    """The whole path, from an ``Adapter`` row to the pixels that get labelled.

    No weights: the model's forward is a stand-in that returns a fixed
    probability map, which is exactly the seam
    :func:`quantem.inference.engine.predict_region` documents for this. What is
    under test is not the model but the wiring -- that
    ``POST /api/adapters/<id>/apply/`` reaches
    :meth:`~quantem.inference.segmenter.DinoOrganelleSegmenter.extract_instances`
    and changes what comes out.
    """

    #: Foreground probability of the one object in the stand-in's output. It sits
    #: between the published default (0.5) and the calibrated threshold (0.31),
    #: so the object is invisible to the base model and found by the adapted one.
    OBJECT_PROB = 0.4
    CALIBRATED = 0.31

    def setUp(self):
        import numpy as np

        from quantem.inference import engine
        from quantem.inference.specs import MODEL_SPECS

        image = create_small_test_image("adapted run", width=256, height=256)
        self.segmentation = create_mitochondria_segmentation(image)

        def forward(tile: np.ndarray) -> np.ndarray:
            prob = np.full(tile.shape[:2], 0.05, dtype=np.float32)
            prob[50:150, 50:150] = self.OBJECT_PROB
            return prob

        engine.clear_model_cache()
        engine.cache_model(
            engine.LoadedModel(
                spec=MODEL_SPECS["quantem:mito"],
                device="cpu",
                module=None,
                forward=forward,
                encoder_tier="stand-in",
            )
        )
        self.addCleanup(engine.clear_model_cache)

    def _run(self) -> int:
        """Segments a full run produces. Uses the real db-aware driver."""
        from quantem.inference.segmenter import DinoMitoSegmenter
        from quantem.seg_core.db.inference import run_inference_for_segmentation

        segmenter = DinoMitoSegmenter(source_model="quantem:mito", device="cpu")
        result, image = run_inference_for_segmentation(segmenter, self.segmentation, None)
        segments = segmenter.extract_instances(
            result.prob, image, result.prob_maps, coordinate_offset=(0, 0)
        )
        self.threshold_used = segmenter.fg_threshold
        return len(segments)

    def test_without_an_adapter_the_run_uses_the_published_threshold(self):
        assert self._run() == 0
        assert self.threshold_used == 0.5

    def test_an_applied_adapter_changes_what_the_run_finds(self):
        _adapter(self.segmentation, calibrated_threshold=self.CALIBRATED)
        assert self._run() == 1
        assert self.threshold_used == self.CALIBRATED

    def test_unapplying_is_enough_to_go_back_to_the_released_model(self):
        adapter = _adapter(self.segmentation, calibrated_threshold=self.CALIBRATED)
        assert self._run() == 1
        adapter.applied_at = None
        adapter.save(update_fields=["applied_at", "updated_at"])
        assert self._run() == 0
        assert self.threshold_used == 0.5
