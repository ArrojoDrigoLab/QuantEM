"""The acceptance gate: quantem-core must reproduce the original inference implementation.

``quantem-core`` is a reimplementation. It carries the same arithmetic, but the encoder is built
from timm rather than Meta's ``dinov3`` package, and every step in between was rewritten. A
faithful-looking port that is subtly different would produce plausible segmentations that are not
the ones behind the published numbers, with no error anywhere to reveal it.

So: run a fixed image through both implementations and compare the probability maps pixel by pixel.

The reference side is captured once, by ``.scratch/capture_goldens.py``, which needs Meta's
``dinov3`` package and the training harness — the dependencies this package deliberately excludes.
Its output is an ``.npz`` of reference probability maps that this test then compares against
forever, needing none of that.

Set ``QUANTEM_GOLDENS`` to the ``.npz``; skipped otherwise.

**The golden file is not distributed anywhere** — not in this repository, not on Hugging Face, not
on Zenodo. It is a development artifact: ~30 MB of intermediate numbers that exist to check this
code against its predecessor, and that nobody installing or citing the plugin needs. It lives in
the gitignored ``.scratch/`` directory on whichever machine last ran the check, and is regenerated
on demand by ``.scratch/capture_goldens.py`` against the reference environment.

That makes this test opt-in by construction: it runs where the file happens to exist, and skips
cleanly everywhere else, including CI.

Measured 2026-08-07, torch 2.13.0+cu130 / timm 1.0.28 / Quadro RTX 8000:

    omniem/mito      5.402e-08        quantem/mito      0.000e+00
    omniem/er        3.874e-07        quantem/er        0.000e+00
    omniem/nucleus   9.537e-07        quantem/nucleus   0.000e+00
    omniem/ld        4.292e-06        quantem/ld        0.000e+00

The QuantEM models are bit-identical. The OmniEM ones sit at float noise: both paths run through
timm, but the reference reaches it via a different call sequence, so the last bits differ.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantem_em.registry import REGISTRY  # noqa: E402

GOLDENS = os.environ.get("QUANTEM_GOLDENS")
TOL = 1e-4

pytestmark = [
    pytest.mark.skipif(not GOLDENS, reason="set QUANTEM_GOLDENS to the reference .npz"),
    pytest.mark.skipif(
        not os.environ.get("QUANTEM_MODEL_DIR"), reason="needs the published artifacts"
    ),
]


@pytest.fixture(scope="module")
def goldens():
    return np.load(GOLDENS)


@pytest.mark.parametrize("model_id", sorted(REGISTRY))
def test_matches_the_reference_implementation(goldens, model_id):
    from quantem_em.api import load_model
    from quantem_em.inference.predict import predict_region

    key = model_id.replace("/", "_")
    if key not in goldens:
        pytest.skip(f"{model_id} not in the golden file")
    ref = goldens[key]
    em = goldens["__image__"]

    model = load_model(model_id, device="cuda" if torch.cuda.is_available() else "cpu")
    # The fixture is already at the model's working scale, so this isolates model assembly,
    # tiling and blending -- the parts that were reimplemented.
    got, _ = predict_region(model.module, em, REGISTRY[model_id], model.device)

    delta = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    assert got.shape == ref.shape
    assert delta.max() < TOL, (
        f"{model_id} diverges from the reference by {delta.max():.3e} "
        f"(tolerance {TOL}). This means the shipped model is not the published model."
    )


@pytest.mark.parametrize(
    "model_id", ["quantem/mito", "quantem/er", "quantem/nucleus", "quantem/ld"]
)
def test_quantem_models_are_bit_identical(goldens, model_id):
    """Stricter than the gate, and a canary.

    Every QuantEM difference found so far had an exact fix -- the LayerNorm epsilon, the bf16 rope
    buffer, the reference's q/k dtype casting. Bit-identity is therefore the true expectation, and
    any drift from it means a new numerical difference has crept in, long before it grows past 1e-4.
    """
    from quantem_em.api import load_model
    from quantem_em.inference.predict import predict_region

    key = model_id.replace("/", "_")
    if key not in goldens:
        pytest.skip(f"{model_id} not in the golden file")
    model = load_model(model_id, device="cuda" if torch.cuda.is_available() else "cpu")
    got, _ = predict_region(model.module, goldens["__image__"], REGISTRY[model_id], model.device)
    assert np.array_equal(got, goldens[key]), "QuantEM parity is no longer exact"
