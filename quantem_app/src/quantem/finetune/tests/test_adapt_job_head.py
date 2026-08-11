"""The ``head`` rung, end to end, on a stand-in model.

The released packs are bare ``state_dict``s that nothing in the app can turn
into a module yet (see ``quantem/inference/README.md``), so the model here is a
three-layer convolutional stand-in with the same ``encoder``/``neck``/``decoder``
shape. That is enough to exercise everything this package owns: crops become
model-scale patches, only the head trains, the sweep runs on freshly predicted
maps, the head is saved, and the saved head is reloaded and re-scored — the
reference implementation's own final assertion.

Marked ``slow``: it needs torch, and it runs a real (tiny) training loop.
"""

from __future__ import annotations

from unittest import mock

import pytest
from django.test import TestCase

torch = pytest.importorskip("torch", reason="head adaptation needs torch")

from quantem.core.config import MODELS_DIR  # noqa: E402
from quantem.finetune.job import adapter_job  # noqa: E402 -- after the torch skip
from quantem.finetune.tests.fixtures import (  # noqa: E402
    FakeCancel,
    FakeReporter,
    annotated_segmentation,
)
from quantem.inference import engine  # noqa: E402
from quantem.inference.specs import MODEL_SPECS  # noqa: E402

pytestmark = pytest.mark.slow

BIG = 1024
BIG_ROI = (20, 20, 1000, 1000)
BIG_OBJECT = (300, 300, 600, 600)


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

    Seeded so every call returns identical weights: that is what "the same
    frozen encoder blob and the same base head from the registry" means, and
    without it the reload check would be comparing two different models.
    """
    torch.manual_seed(1234)
    return engine.LoadedModel(spec=MODEL_SPECS[pack_id], device="cpu", module=StandInModel())


def _big_segmentation(name: str):
    return annotated_segmentation(name, size=BIG, roi=BIG_ROI, obj=BIG_OBJECT)


class HeadModeJobTests(TestCase):
    def test_head_mode_trains_the_head_calibrates_and_verifies_the_saved_file(self):
        first = _big_segmentation("Head image one")
        _big_segmentation("Head image two")

        payload = {
            "segmentation_id": str(first.id),
            "base_model": "quantem:mito",
            "mode": "head",
            "steps": 6,
            "lr": 0.01,
            "seed": 0,
            "adapter_id": None,
        }
        reporter = FakeReporter()
        with mock.patch.object(engine, "load_model", side_effect=_loader):
            result = adapter_job(payload, reporter, FakeCancel())

        assert result["mode"] == "head"
        assert result["steps"] == 6
        assert result["split_mode"] == "image-disjoint"
        # 512 for a patch-16 model, straight from the reference recipe.
        assert result["tile"] == 512
        assert result["trainable_params"] > 0

        # Both sweeps are reported, so "before" and "after" are comparable.
        assert result["base_sweep"]["calibrated_threshold"] > 0
        assert result["sweep"]["calibrated_threshold"] > 0
        assert result["sweep"]["heldout_dice_at_calibrated"] is not None

        # The saved head exists and reproduces the held-out score on reload.
        # Read back from the result rather than reconstructed: a run with no
        # adapter row gets a scratch path unique to it, so that two of them at
        # once cannot verify each other's weights.
        head_file = MODELS_DIR / result["head_path"]
        try:
            assert head_file.exists()
            assert result["verified_reload"] is True
            assert result["reloaded_heldout_dice"] == pytest.approx(
                result["sweep"]["heldout_dice_at_calibrated"], abs=1e-4
            )
        finally:
            head_file.unlink(missing_ok=True)

        # The shared model cache must not be left holding an adapted head that
        # the user never applied.
        assert engine.cached_model_keys() == []

    def test_a_region_too_small_for_one_tile_is_refused_not_silently_skipped(self):
        from quantem.finetune.adapt import HeadAdaptationUnavailable

        segmentation = annotated_segmentation("Head tiny region", organelle="er")
        payload = {
            "segmentation_id": str(segmentation.id),
            "base_model": "quantem:mito",
            "mode": "head",
            "steps": 2,
        }
        with mock.patch.object(engine, "load_model", side_effect=_loader):
            with pytest.raises(HeadAdaptationUnavailable, match="No training window"):
                adapter_job(payload, FakeReporter(), FakeCancel())
