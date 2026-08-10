"""FINO data path: EM dataset emits encoded metadata; upstream collate batches it (gated)."""

from __future__ import annotations

import pytest

from em_ssl.fino.factors import EMTileMetadata, FinoRuntime, factors_from_config
from em_ssl.integration import dinov3_patch
from em_ssl.integration.em_dataset import make_em_dataset

@pytest.fixture
def fino_runtime():
    facs = factors_from_config([
        {"name": "modality", "field": "modality", "type": "discrete", "guidance": "positive"},
        {"name": "organ", "field": "organ", "type": "discrete", "guidance": "negative"},
    ])
    prev = dinov3_patch.ACTIVE_FINO
    dinov3_patch.set_fino_runtime(FinoRuntime(facs))
    try:
        yield facs
    finally:
        dinov3_patch.set_fino_runtime(prev)

def test_em_dataset_emits_encoded_metadata(built_shards, fino_runtime):
    ds = make_em_dataset(
        f"EMShards:root={built_shards['shard_dir']}:prefix=em_tiles_v0:min_side=512:resampled=0:shuffle=0",
        transform=lambda x: x,
        target_transform=lambda _: (),  # FINO mode overrides this
    )
    seen = 0
    for _img, tgt in ds:
        assert isinstance(tgt, tuple) and len(tgt) == 2
        label, md = tgt
        assert label == () and isinstance(md, EMTileMetadata)
        # every sample carries a class index per discrete factor: a vocab index, or -1 when masked out
        assert md.modality >= -1 and md.organ >= -1
        seen += 1
    assert seen == built_shards["n"]

def test_baseline_mode_emits_empty_target(built_shards):
    prev = dinov3_patch.ACTIVE_FINO
    dinov3_patch.set_fino_runtime(None)
    try:
        ds = make_em_dataset(
            f"EMShards:root={built_shards['shard_dir']}:prefix=em_tiles_v0:min_side=512:resampled=0:shuffle=0",
            transform=lambda x: x,
            target_transform=lambda _: (),
        )
        _img, tgt = next(iter(ds))
        assert tgt == ()
    finally:
        dinov3_patch.set_fino_runtime(prev)

def test_dataset_bridges_realized_downsample_into_scale_label(built_shards):
    """End-to-end: the aug's per-sample global downsample reaches the FINO scale label as
    log(native_nm * M_g). Both globals share one M_g, so this single per-sample value is the exact
    target for both global crops the guide head reads. A stub transform stands in for the real aug to
    pin M_g deterministically (the realized-M math itself is covered by test_native_fov)."""
    import math

    facs = factors_from_config([{
        "name": "log_effective_nm_per_px", "field": "effective_nm_per_px", "type": "continuous",
        "guidance": "positive", "log_transform": True, "crop_scale_correction": True,
    }])
    prev = dinov3_patch.ACTIVE_FINO
    dinov3_patch.set_fino_runtime(FinoRuntime(facs))
    try:
        M_G = 3.0  # built_shards tiles are all effective_nm_per_px=5.0 -> true crop nm/px = 15.0
        ds = make_em_dataset(
            f"EMShards:root={built_shards['shard_dir']}:prefix=em_tiles_v0:min_side=512:resampled=0:shuffle=0",
            transform=lambda img: {"global_crops": [img, img], "global_downsample": M_G},
            target_transform=lambda _: (),  # overridden by ACTIVE_FINO
        )
        seen = 0
        for t, (label, md) in ds:
            assert "global_downsample" not in t  # popped before the model collate sees the crops
            assert label == () and md.log_effective_nm_per_px_valid is True
            assert md.log_effective_nm_per_px == pytest.approx(math.log(5.0 * M_G))
            seen += 1
        assert seen == built_shards["n"]
    finally:
        dinov3_patch.set_fino_runtime(prev)

def test_upstream_collate_batches_em_metadata():
    """The pinned FINO-branch collate must batch EMTileMetadata into per-field tensors."""
    pytest.importorskip("dinov3", reason="DINOv3 not installed; run third_party/fetch_dinov3.sh")
    import torch
    from dinov3.data.collate import _collate_metadata

    rows = [
        EMTileMetadata(log_effective_nm_per_px=0.1, modality=0, organ=2,
                       log_effective_nm_per_px_valid=True, modality_valid=True, organ_valid=True,
                       source_id="s0", dataset_id="d0"),
        EMTileMetadata(log_effective_nm_per_px=0.0, modality=-1, organ=1,
                       log_effective_nm_per_px_valid=False, modality_valid=False, organ_valid=True,
                       source_id="s1", dataset_id="d0"),
    ]
    batched = _collate_metadata(rows)
    assert isinstance(getattr(batched, "modality"), torch.Tensor)
    assert getattr(batched, "modality").tolist() == [0, -1]
    assert getattr(batched, "modality_valid").tolist() == [1, 0]
    assert getattr(batched, "log_effective_nm_per_px").dtype == torch.float32
    assert getattr(batched, "source_id") == ["s0", "s1"]  # strings -> python list (diagnostics)
