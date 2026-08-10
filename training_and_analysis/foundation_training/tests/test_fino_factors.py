"""FINO factor layer: allow/deny guard, vocab canonicalization, log derivation, masking."""

from __future__ import annotations

import math
import pickle

import pytest

from em_ssl.fino.factors import (
    EMTileMetadata,
    FinoFactorSpec,
    FinoRuntime,
    encode_tile_metadata,
    factor_from_dict,
    factors_from_config,
    fino_factors_fingerprint,
)

def test_allowed_objective_guard_rejects_provenance():
    for field in ("source_id", "dataset_id", "scale_band", "species_group", "prep_context", "in_situ_status"):
        with pytest.raises(ValueError):
            factors_from_config([{"name": "x", "field": field, "type": "discrete", "guidance": "positive"}])

def test_allowed_objective_guard_rejects_name_field_mismatch():
    with pytest.raises(ValueError):
        factors_from_config([{"name": "modality", "field": "organ", "type": "discrete", "guidance": "positive"}])

def test_modality_canonicalization_and_defaults():
    f = factor_from_dict({"name": "modality", "field": "modality", "type": "discrete", "guidance": "positive"})
    assert f.n_outputs == len(f.effective_classes) >= 7  # default vocab filled
    # spelling/case variants normalize to canonical
    assert f.encode_value("FIB-SEM")[0] == f.class_to_idx["FIB-SEM"]
    assert f.encode_value("fibsem")[0] == f.class_to_idx["FIB-SEM"]
    assert f.encode_value("fib_sem")[0] == f.class_to_idx["FIB-SEM"]
    # unknown value -> masked out (-1, invalid) when include_unknown is false
    enc, valid = f.encode_value("not-a-real-scope")
    assert enc == -1 and valid is False
    # missing -> masked out
    assert f.encode_value(None) == (-1, False)

def test_include_unknown_keeps_unknown_as_class():
    f = factor_from_dict(
        {"name": "modality", "field": "modality", "type": "discrete", "guidance": "positive", "include_unknown": True}
    )
    enc, valid = f.encode_value("mystery")
    assert valid is True and f.effective_classes[enc] == "unknown"

def test_continuous_log_derivation_and_standardize():
    f = FinoFactorSpec(
        name="log_effective_nm_per_px", field="effective_nm_per_px", type="continuous", guidance="negative",
        log_transform=True, standardize_mean=1.6, standardize_std=0.5,
    )
    f.validate()
    enc, valid = f.encode_value(5.0)
    assert valid is True
    assert enc == pytest.approx((math.log(5.0) - 1.6) / 0.5)
    # invalid: non-positive / non-finite / non-numeric -> masked out
    assert f.encode_value(-1.0) == (0.0, False)
    assert f.encode_value(0.0) == (0.0, False)
    assert f.encode_value(float("nan")) == (0.0, False)
    assert f.encode_value("abc") == (0.0, False)

def test_prototypical_and_bce_rejected():
    # Prototypical heads are not mask-aware -> rejected for partial EM metadata.
    with pytest.raises(ValueError):
        factors_from_config([{"name": "modality", "field": "modality", "type": "discrete",
                              "guidance": "positive", "method": "prototypical"}])
    # EM discrete factors are single-label -> multi-label BCE is a misconfiguration.
    with pytest.raises(ValueError):
        factors_from_config([{"name": "modality", "field": "modality", "type": "discrete",
                              "guidance": "positive", "use_bce": True}])

def test_continuous_zero_std_rejected():
    f = FinoFactorSpec(name="log_effective_nm_per_px", field="effective_nm_per_px", type="continuous",
                       guidance="positive", standardize_std=0.0)
    with pytest.raises(ValueError):
        f.validate()

def test_encode_tile_metadata_only_configured_factors_and_diagnostics():
    facs = factors_from_config([
        {"name": "modality", "field": "modality", "type": "discrete", "guidance": "positive"},
    ])
    md = encode_tile_metadata({"modality": "TEM", "organ": "Brain",
                               "effective_nm_per_px": 8.0, "source_id": "s9", "dataset_id": "d3"}, facs)
    assert isinstance(md, EMTileMetadata)
    assert md.modality == facs[0].class_to_idx["TEM"] and md.modality_valid is True
    # organ not configured -> stays default/invalid; diagnostics always carried
    assert md.organ == -1 and md.organ_valid is False
    assert md.source_id == "s9" and md.dataset_id == "d3"

def test_target_transform_is_picklable():
    rt = FinoRuntime(factors_from_config([
        {"name": "organ", "field": "organ", "type": "discrete", "guidance": "negative"},
    ]))
    tt = rt.target_transform()
    tt2 = pickle.loads(pickle.dumps(tt))  # must survive dataloader-worker pickling (Windows spawn)
    label, md = tt2({"organ": "Kidney"})
    assert label == () and md.organ >= 0 and md.organ_valid is True

def test_fingerprint_stable_and_sensitive():
    a = factors_from_config([{"name": "modality", "field": "modality", "type": "discrete", "guidance": "positive"}])
    b = factors_from_config([{"name": "modality", "field": "modality", "type": "discrete", "guidance": "negative"}])
    assert fino_factors_fingerprint(a) == fino_factors_fingerprint(a)
    assert fino_factors_fingerprint(a) != fino_factors_fingerprint(b)
    # crop_scale_correction is part of the factor identity (provenance must distinguish the runs).
    c = factors_from_config([{"name": "log_effective_nm_per_px", "field": "effective_nm_per_px",
                              "type": "continuous", "guidance": "positive", "log_transform": True}])
    d = factors_from_config([{"name": "log_effective_nm_per_px", "field": "effective_nm_per_px",
                              "type": "continuous", "guidance": "positive", "log_transform": True,
                              "crop_scale_correction": True}])
    assert fino_factors_fingerprint(c) != fino_factors_fingerprint(d)

def test_crop_scale_correction_scales_value_before_log():
    # A 5 nm/px tile seen through a 3x-downsampled crop is truly 15 nm/px; the regression target must
    # be log(15), i.e. log(native) + log(M_g).
    f = FinoFactorSpec(name="log_effective_nm_per_px", field="effective_nm_per_px", type="continuous",
                       guidance="positive", log_transform=True, crop_scale_correction=True)
    f.validate()
    enc, valid = f.encode_value(5.0, downsample=3.0)
    assert valid is True and enc == pytest.approx(math.log(15.0))
    assert enc == pytest.approx(math.log(5.0) + math.log(3.0))  # additive in log space
    # native crop (M=1) and the default arg leave the value unchanged.
    assert f.encode_value(5.0, downsample=1.0)[0] == pytest.approx(math.log(5.0))
    assert f.encode_value(5.0)[0] == pytest.approx(math.log(5.0))
    # correction composes with standardization.
    g = FinoFactorSpec(name="log_effective_nm_per_px", field="effective_nm_per_px", type="continuous",
                       guidance="negative", log_transform=True, crop_scale_correction=True,
                       standardize_mean=1.6, standardize_std=0.5)
    assert g.encode_value(5.0, downsample=3.0)[0] == pytest.approx((math.log(15.0) - 1.6) / 0.5)
    # invalid (masked) samples are untouched by the correction.
    assert g.encode_value(0.0, downsample=3.0) == (0.0, False)

def test_crop_scale_correction_off_ignores_downsample():
    f = FinoFactorSpec(name="log_effective_nm_per_px", field="effective_nm_per_px", type="continuous",
                       guidance="positive", log_transform=True, crop_scale_correction=False)
    assert f.encode_value(5.0, downsample=3.0)[0] == pytest.approx(math.log(5.0))

def test_crop_scale_correction_rejected_on_discrete_factor():
    f = factor_from_dict({"name": "modality", "field": "modality", "type": "discrete",
                          "guidance": "positive", "crop_scale_correction": True})
    with pytest.raises(ValueError):
        f.validate()

def test_encode_tile_metadata_applies_bridged_downsample():
    facs = factors_from_config([{"name": "log_effective_nm_per_px", "field": "effective_nm_per_px",
                                 "type": "continuous", "guidance": "positive", "log_transform": True,
                                 "crop_scale_correction": True}])
    md = encode_tile_metadata({"effective_nm_per_px": 5.0, "global_downsample": 3.0}, facs)
    assert md.log_effective_nm_per_px_valid is True
    assert md.log_effective_nm_per_px == pytest.approx(math.log(15.0))
    # absent bridged key -> no correction (native_fov off / non-FINO).
    md2 = encode_tile_metadata({"effective_nm_per_px": 5.0}, facs)
    assert md2.log_effective_nm_per_px == pytest.approx(math.log(5.0))
