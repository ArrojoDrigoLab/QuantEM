"""FINO meta-arch graft + upstream guide heads (requires the pinned DINOv3 FINO branch)."""

from __future__ import annotations

import pytest

dinov3 = pytest.importorskip("dinov3", reason="DINOv3 not installed; run third_party/fetch_dinov3.sh")

def test_apply_fino_grafts_installs_masked_guide_losses():
    from em_ssl.fino.meta_arch_patch import apply_fino_grafts
    from dinov3.train.guided_ssl_meta_arch import GuidedSSLMetaArch

    assert apply_fino_grafts() is True
    assert getattr(GuidedSSLMetaArch._compute_guide_losses, "_em_masked", False) is True
    # idempotent
    assert apply_fino_grafts() is True

def test_apply_em_patches_includes_fino_graft():
    import em_ssl.integration.dinov3_patch as p
    from dinov3.train.guided_ssl_meta_arch import GuidedSSLMetaArch

    p.apply_em_patches()
    assert getattr(GuidedSSLMetaArch._compute_guide_losses, "_em_masked", False) is True

def test_upstream_guide_heads_build_with_translated_dims():
    """The dims config translation emits must instantiate real upstream heads."""
    import torch
    from dinov3.train.metadata_utils import Classifier, Regressor

    from em_ssl.config.schema import ExperimentSpec, _coerce
    from em_ssl.integration import config_translation as ct

    spec = _coerce(ExperimentSpec, {
        "name": "meta_modality_minus", "framework": "dinov3",
        "crops": {"schedule": [{"global_crops_size": 512, "max_steps": 100}]},
        "data": {"shard_dir": "/d"},
        "metadata_factors": [
            {"name": "modality", "field": "modality", "type": "discrete", "guidance": "positive", "loss_weight": 0.1},
            {"name": "log_effective_nm_per_px", "field": "effective_nm_per_px", "type": "continuous",
             "guidance": "negative", "loss_weight": 0.05, "log_transform": True},
        ],
    })
    cfg = ct.translate_stage(spec, 0, "/r", 0)
    embed_dim = 768
    for g in cfg["guide"]["guides"]:
        if g["method"] == "classification":
            head = Classifier(input_dim=embed_dim, hidden_dim=list(g["hidden_dim"]), num_classes=g["n_outputs"], dropout=g["dropout"])
            out = head(torch.randn(4, embed_dim), 1.0)
            assert out.shape == (4, g["n_outputs"])
        else:
            head = Regressor(input_dim=embed_dim, hidden_dim=list(g["hidden_dim"]), n_outputs=g["n_outputs"], dropout=g["dropout"])
            out = head(torch.randn(4, embed_dim), -1.0)
            assert out.shape == (4, g["n_outputs"])
