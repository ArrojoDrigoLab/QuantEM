"""The eight released specs must match the staged training configs exactly.

These values are the inference contract. If one drifts, the plugin stops being the paper.
Cross-checked against
``training_and_analysis/segmentation_training/configs/released_models/*.yaml`` on 2026-08-06.
"""

from __future__ import annotations

import pytest

from quantem_em import get_model_spec, list_models
from quantem_em.registry import DEFAULT_MODEL_FOR_ORGANELLE, REGISTRY

EXPECTED = {
    #  model_id            neck                decoder          adapt      canonical_nm  task
    "omniem/mito": ("naive_1x1", "affinity_mws", "lora", 8.0, "instance"),
    "quantem/mito": ("naive_1x1", "affinity_mws", "last_n", 8.0, "instance"),
    "omniem/er": ("resnet34_detail", "dpt", "lora", None, "semantic"),
    "quantem/er": ("resnet34_detail", "upernet", "full", None, "semantic"),
    "omniem/nucleus": ("naive_1x1", "affinity_mws", "lora", 25.0, "instance"),
    "quantem/nucleus": ("naive_1x1", "affinity_mws", "last_n", 25.0, "instance"),
    "omniem/ld": ("naive_1x1", "affinity_mws", "lora", 8.0, "instance"),
    "quantem/ld": ("naive_1x1", "affinity_mws", "last_n", 8.0, "instance"),
}


def test_exactly_eight_models():
    assert len(list_models()) == 8
    assert set(REGISTRY) == set(EXPECTED)


@pytest.mark.parametrize("model_id", sorted(EXPECTED))
def test_spec_matches_released_config(model_id):
    neck, decoder, adapt, nm, task = EXPECTED[model_id]
    s = get_model_spec(model_id)
    assert s.neck == neck
    assert s.decoder == decoder
    assert s.adapt == adapt
    assert s.canonical_nm == nm
    assert s.task == task
    # Shared across all eight, read off the configs.
    assert s.neck_out_channels == 256
    assert s.feature_layers == "last4"
    assert s.apply_encoder_norm is True
    assert s.num_classes == 2
    assert s.overlap == 0.25
    assert s.fg_threshold == 0.5
    assert s.instance_min_size == 16
    assert s.tile_size == 512


def test_effective_tile_and_stride():
    """518 appears in no config file -- it emerges from round_up(512, 14) at runtime."""
    q = get_model_spec("quantem/mito")
    o = get_model_spec("omniem/mito")
    assert q.encoder.patch_size == 16 and o.encoder.patch_size == 14
    assert q.effective_tile() == 512
    assert o.effective_tile() == 518
    # Reference: evaluate.py:88  stride = max(1, int(round(t * (1.0 - overlap))))
    # 512 * 0.75 = 384.0            -> 384
    # 518 * 0.75 = 388.5 exactly    -> 388, because Python rounds halves to EVEN.
    # (Plan rev 1 asserted 389 here; it was wrong. An off-by-one shifts every window and
    #  changes the Hann blend, so this is a parity-critical value.)
    assert q.stride() == 384
    assert o.stride() == 388


def test_er_runs_at_native_resolution():
    for mid in ("omniem/er", "quantem/er"):
        assert get_model_spec(mid).canonical_nm is None
        assert get_model_spec(mid).resamples is False


def test_normalisation_constants():
    q = get_model_spec("quantem/mito").encoder
    assert (q.dataset_mean, q.dataset_std) == (0.583175, 0.244468)
    assert q.encoder_mean is None  # native single channel
    o = get_model_spec("omniem/mito").encoder
    assert (o.dataset_mean, o.dataset_std) == (0.0, 1.0)  # raw [0, 1] into the encoder
    assert (o.encoder_mean, o.encoder_std) == (0.595446, 0.211906)


def test_prefix_token_counts():
    assert get_model_spec("quantem/mito").encoder.n_prefix_tokens == 5  # CLS + 4 storage
    assert get_model_spec("omniem/mito").encoder.n_prefix_tokens == 1  # CLS only


def test_artifact_sharing_reflects_actual_adaptation():
    """LoRA leaves the base untouched; last_n rewrites blocks 8-11; full replaces everything."""
    for mid in ("omniem/mito", "omniem/er", "omniem/nucleus", "omniem/ld"):
        assert get_model_spec(mid).trunk_artifact == "omniem-vitl"
    for mid in ("quantem/mito", "quantem/nucleus", "quantem/ld"):
        assert get_model_spec(mid).trunk_artifact == "quantem-vitb-trunk"
    # adapt="full" stores the entire encoder in the head -> no trunk download needed.
    assert get_model_spec("quantem/er").trunk_artifact is None


def test_ui_defaults():
    """Owner ruling 2026-08-06: QuantEM for mitochondria, OmniEM for the rest."""
    assert DEFAULT_MODEL_FOR_ORGANELLE == {
        "mito": "quantem/mito",
        "er": "omniem/er",
        "nucleus": "omniem/nucleus",
        "ld": "omniem/ld",
    }
    for mid in DEFAULT_MODEL_FOR_ORGANELLE.values():
        assert mid in REGISTRY


def test_import_does_not_pull_torch():
    """napari imports plugin top-levels at manifest discovery; torch must not ride along."""
    import subprocess
    import sys

    code = "import quantem_em, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_unknown_model_id_is_helpful():
    with pytest.raises(KeyError, match="known ids"):
        get_model_spec("quantem/golgi")
