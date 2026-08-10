"""Tests for the native-resolution field-of-view crop (em_dino_augmentations).

The pure magnification draw is checked directly; an end-to-end pass confirms the output contract.
Requires torch/torchvision.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch/torchvision required")

from em_ssl.transforms.em_dino_augmentations import (  # noqa: E402
    EMAugmentationConfig,
    EMDataAugmentationDINO,
    native_magnification,
)

def test_magnification_endpoints():
    assert native_magnification(4.0, 3.0, 0.0) == 1.0  # u=0 -> native (1:1)
    assert native_magnification(4.0, 3.0, 1.0) == 4.0  # u=1 -> max downsample
    assert native_magnification(1.0, 3.0, 0.7) == 1.0  # downsample_max=1 -> always native

def test_magnification_bias_toward_native():
    # bias>1 maps u=0.5 below the linear (bias=1) value -> closer to native.
    biased = native_magnification(4.0, 3.0, 0.5)  # 1 + 3*0.125 = 1.375
    linear = native_magnification(4.0, 1.0, 0.5)  # 1 + 3*0.5   = 2.5
    assert biased < linear
    assert abs(biased - 1.375) < 1e-9

def test_magnification_monotonic_in_u():
    vals = [native_magnification(4.0, 3.0, u / 10) for u in range(11)]
    assert vals == sorted(vals)

def _aug(native_fov, **em_over):
    em = EMAugmentationConfig(native_fov=native_fov, **em_over)
    return EMDataAugmentationDINO(
        global_crops_scale=(0.32, 1.0), local_crops_scale=(0.05, 0.32), local_crops_number=4,
        global_crops_size=512, local_crops_size=112, mean=(0.5,), std=(0.25,), em=em, expected_channels=1,
    )

def test_rand_crop_native_is_an_exact_slice():
    import torch

    aug = _aug(native_fov=True)
    tile = torch.arange(2048 * 2048, dtype=torch.float32).reshape(1, 2048, 2048)
    native = aug._rand_crop(tile, 512, 512)  # region == out -> no resize
    assert tuple(native.shape) == (1, 512, 512)
    # exact slice: every value must exist in the source (no interpolation introduced new values)
    assert native.flatten()[0].item() == float(int(native.flatten()[0].item()))
    resized = aug._rand_crop(tile, 1024, 512)  # region != out -> resized
    assert tuple(resized.shape) == (1, 512, 512)

def test_native_fov_preserves_output_contract():
    import torch

    tile = torch.rand(1, 2048, 2048)
    out = _aug(native_fov=True, native_downsample_max=4.0, native_bias=3.0)(tile)
    assert len(out["global_crops"]) == 2
    assert all(tuple(g.shape) == (1, 512, 512) for g in out["global_crops"])
    assert len(out["local_crops"]) == 4
    assert all(tuple(c.shape) == (1, 112, 112) for c in out["local_crops"])

def test_native_fov_off_unchanged_shapes():
    import torch

    out = _aug(native_fov=False)(torch.rand(1, 2048, 2048))
    assert tuple(out["global_crops"][0].shape) == (1, 512, 512)

def test_small_and_nonsquare_tiles_clamp_to_native():
    # Crop sizes are absolute pixels clamped to the real tile H/W — small tiles (incl. barely-512 and
    # non-square) cap at their native size instead of upscaling, and still produce valid crops.
    import torch

    aug = _aug(native_fov=True, native_downsample_max=4.0, native_bias=5.0)
    for h, w in [(512, 512), (700, 900), (1300, 600)]:
        out = aug(torch.rand(1, h, w))
        assert all(tuple(g.shape) == (1, 512, 512) for g in out["global_crops"])
        assert all(tuple(c.shape) == (1, 112, 112) for c in out["local_crops"])

def test_schema_default_enables_native_fov_bias5():
    # native_fov is on (bias 5) by default for every em_ssl run, set in the schema.
    from em_ssl.config.schema import AugmentationSpec

    a = AugmentationSpec()
    assert a.native_fov is True and a.native_bias == 5.0
    # The transform-level default stays off: it is a general-purpose tool, opt-in there.
    assert EMAugmentationConfig().native_fov is False

def test_native_fov_emits_realized_global_downsample():
    # native_fov surfaces the realized global-crop downsample (both globals share it) so FINO can
    # correct the nm/px label; standard RRC does not (the two globals are independent draws).
    import torch

    out = _aug(native_fov=True, native_downsample_max=4.0, native_bias=5.0)(torch.rand(1, 2048, 2048))
    m = out["global_downsample"]
    assert isinstance(m, float) and 1.0 <= m <= 4.0 + 1e-6
    out_off = _aug(native_fov=False)(torch.rand(1, 2048, 2048))
    assert "global_downsample" not in out_off

def test_native_fov_downsample_clamps_on_small_tiles():
    # bias->0 forces a near-max drawn magnification, so a tile barely larger than the crop must report
    # a downsample clamped to (tile / crop) — the region can't exceed the tile (never upsamples).
    import torch

    aug = _aug(native_fov=True, native_downsample_max=4.0, native_bias=1e-4)
    for side in (560, 700):
        ms = [aug(torch.rand(1, side, side))["global_downsample"] for _ in range(16)]
        assert max(ms) <= side / 512 + 1e-6
    # a full 2048 tile at near-max magnification should reach ~4x.
    big = [aug(torch.rand(1, 2048, 2048))["global_downsample"] for _ in range(16)]
    assert max(big) == pytest.approx(4.0, abs=1e-6)
