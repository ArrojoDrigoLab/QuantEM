"""Loading a released pack with a user-trained head, against real weights.

:func:`quantem.inference.engine.load_adapted_model` is the second half of what
``POST /api/adapters/<id>/apply/`` promises: the first half is the calibrated
threshold (pure arithmetic, tested in ``seg_core``), and this is the part where
a 23 MB neck + decoder has to land correctly on a 525 MB frozen encoder that was
built from a different file.

Marked ``requires_weights``: skipped unless packs have been installed with::

    python -m quantem.registry.install local --all

The adapted head used here is the *released* head round-tripped through
:func:`quantem.finetune.adapt.save_head`, so the loaded model must reproduce the
base model exactly. That is a sharper assertion than a trained head could give:
any mistake in the wiring -- wrong submodule, silently skipped load, a shared
cache entry handed back -- shows up as a difference, and there is no training
noise to hide in. A second pass with deliberately perturbed weights proves the
comparison is not vacuous.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from quantem.inference import engine
from quantem.registry import cache as registry_cache
from quantem.testing import make_em_like_array

pytestmark = pytest.mark.requires_weights

PACK_ID = "quantem:mito"
IMAGE_SIZE = 256


@pytest.fixture(autouse=True)
def _drop_models() -> Iterator[None]:
    yield
    engine.clear_model_cache()


@pytest.fixture
def base_model():
    if not registry_cache.installed(PACK_ID):
        pytest.skip(
            f"{PACK_ID} is not installed; run `python -m quantem.registry.install local --all`"
        )
    return engine.load_model(PACK_ID, device="cpu")


@pytest.fixture
def adapted_head(base_model, tmp_path) -> Path:
    """The released head, saved in the format an adaptation writes."""
    from quantem.finetune.adapt import save_head

    return save_head(base_model.module, tmp_path / "head.pt", meta={"test": True})


def _predict(model) -> np.ndarray:
    return engine.predict_region(model, make_em_like_array(IMAGE_SIZE, IMAGE_SIZE, seed=3)).prob


def test_an_adapted_head_loads_onto_the_released_encoder(base_model, adapted_head):
    adapted = engine.load_adapted_model(PACK_ID, adapted_head, "cpu", adapter_id="test-1")
    assert adapted.pack_id == PACK_ID
    assert adapted.adapter_id == "test-1"
    assert adapted.encoder_tier == base_model.encoder_tier
    # Same weights in, same probabilities out.
    np.testing.assert_allclose(_predict(adapted), _predict(base_model), atol=1e-6)


def test_the_head_is_really_applied(base_model, adapted_head):
    """Perturb the saved decoder; the output must move."""
    import torch

    payload = torch.load(str(adapted_head), map_location="cpu", weights_only=False)
    payload["decoder"] = {
        key: (value + 0.05 if value.dtype.is_floating_point else value)
        for key, value in payload["decoder"].items()
    }
    torch.save(payload, str(adapted_head))

    adapted = engine.load_adapted_model(PACK_ID, adapted_head, "cpu")
    assert not np.allclose(_predict(adapted), _predict(base_model), atol=1e-4)


def test_the_released_pack_keeps_its_own_cache_slot(base_model, adapted_head):
    """An adapted model must not be handed to a run that asked for the base.

    Mutating the cached released module in place would give every later run of
    this pack a head the user never applied -- including runs on segmentations
    the adapter has nothing to do with.
    """
    adapted = engine.load_adapted_model(PACK_ID, adapted_head, "cpu")
    assert adapted is not base_model
    assert adapted.key != base_model.key
    assert engine.load_model(PACK_ID, device="cpu") is base_model
    assert engine.load_adapted_model(PACK_ID, adapted_head, "cpu") is adapted


def test_a_rewritten_head_is_not_served_from_the_cache(base_model, adapted_head):
    """Re-running an adaptation writes the same path; the cache must notice."""
    import time

    import torch

    first = engine.load_adapted_model(PACK_ID, adapted_head, "cpu")
    payload = torch.load(str(adapted_head), map_location="cpu", weights_only=False)
    time.sleep(0.01)
    torch.save(payload, str(adapted_head))

    second = engine.load_adapted_model(PACK_ID, adapted_head, "cpu")
    assert second is not first


def test_a_missing_head_is_named(base_model, tmp_path):
    with pytest.raises(FileNotFoundError, match="cannot be applied"):
        engine.load_adapted_model(PACK_ID, tmp_path / "nope.pt", "cpu")


def test_an_unknown_pack_is_refused(adapted_head):
    with pytest.raises(ValueError, match="Unknown model pack"):
        engine.load_adapted_model("quantem:golgi", adapted_head, "cpu")
