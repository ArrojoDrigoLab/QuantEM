"""FINO config: schema coercion/validation + translation to the DINOv3 guide block."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from em_ssl.config.schema import ExperimentSpec, _coerce, load_experiment
from em_ssl.integration import config_translation as ct

def _spec(factors, framework="dinov3"):
    return _coerce(ExperimentSpec, {
        "name": "t", "framework": framework,
        "crops": {"schedule": [{"global_crops_size": 512, "local_crops_size": 112, "max_steps": 100}]},
        "data": {"shard_dir": "/d"},
        "metadata_factors": factors,
    })

def test_baseline_has_no_guide_block():
    spec = _spec([])
    assert spec.fino_enabled is False
    cfg = ct.translate_stage(spec, 0, "/r", 0)
    assert cfg["MODEL"]["META_ARCHITECTURE"] == "SSLMetaArch"
    assert "guide" not in cfg

def test_fino_translation_meta_arch_and_guide_entry():
    spec = _spec([{"name": "modality", "field": "modality", "type": "discrete", "guidance": "negative", "loss_weight": 0.1}])
    spec.validate_fino()
    cfg = ct.translate_stage(spec, 0, "/r", 0)
    assert cfg["MODEL"]["META_ARCHITECTURE"] == "GuidedSSLMetaArch"
    assert cfg["guide"]["enabled"] is True
    g = cfg["guide"]["guides"][0]
    assert g["name"] == "modality" and g["method"] == "classification" and g["grl"] is True
    assert g["n_outputs"] == spec.metadata_factors[0].n_outputs
    assert cfg["guide"]["lambda_schedule"]["type"] == "sigmoid"

def test_continuous_factor_translates_to_regression():
    spec = _spec([{"name": "log_effective_nm_per_px", "field": "effective_nm_per_px", "type": "continuous",
                   "guidance": "positive", "loss_weight": 0.05, "log_transform": True}])
    cfg = ct.translate_stage(spec, 0, "/r", 0)
    g = cfg["guide"]["guides"][0]
    assert g["method"] == "regression" and g["n_outputs"] == 1 and g["grl"] is False

def test_validate_fino_rejects_denied_field_and_wrong_framework():
    # Schema loads permissively; validate_fino() enforces the allow/deny guard and the dinov3-only
    # framework requirement.
    denied = _spec([{"name": "src", "field": "source_id", "type": "discrete", "guidance": "positive"}])
    with pytest.raises(ValueError):
        denied.validate_fino()
    wrong_fw = _spec([{"name": "modality", "field": "modality", "type": "discrete", "guidance": "positive"}], framework="other")
    with pytest.raises(ValueError):
        wrong_fw.validate_fino()

def test_encoder_manifest_fields_record_fino_provenance():
    spec = _spec([{"name": "modality", "field": "modality", "type": "discrete", "guidance": "negative", "loss_weight": 0.1}])
    emf = ct.encoder_manifest_fields(spec, 0)
    assert emf["fino_enabled"] is True
    assert emf["fino_factors"][0]["guidance"] == "negative" and emf["fino_factors"][0]["grl"] is True
    assert emf["fino_factors_fingerprint"]
    # baseline -> empty/false
    b = ct.encoder_manifest_fields(_spec([]), 0)
    assert b["fino_enabled"] is False and b["fino_factors"] == [] and b["fino_factors_fingerprint"] is None

_META_CONFIGS = sorted(
    str(p) for p in (Path(__file__).resolve().parents[1]
                     / "configs" / "metadata_conditioning").glob("*.yaml"))

@pytest.mark.parametrize("cfg_path", _META_CONFIGS)
def test_metadata_configs_load_validate_translate(cfg_path):
    spec = load_experiment(cfg_path)
    spec.validate_fino()
    if not spec.fino_enabled:
        pytest.skip("the control arm continues pretraining with conditioning off")

    spec.data.shard_dir = "/d"
    cfg = ct.translate_stage(spec, 0, "/r", 0)
    assert cfg["MODEL"]["META_ARCHITECTURE"] == "GuidedSSLMetaArch"
    assert len(cfg["guide"]["guides"]) == 1
