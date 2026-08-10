"""EM single-channel augmentations: 1-channel loading and the DINO multi-crop format."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from em_ssl.transforms import (
    EMDataAugmentationDINO,
    to_float_chw,
)
from em_ssl.transforms.primitives import RandomDihedral

def _img(h=600, w=640, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((h, w)) * 255).astype("uint8"), mode="L")

def test_to_float_chw_single_channel():
    x = to_float_chw(_img())
    assert x.shape[0] == 1 and x.dtype == torch.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0

def test_dino_aug_format_and_channels():
    aug = EMDataAugmentationDINO(global_crops_size=224, local_crops_size=96, local_crops_number=8,
                                 mean=(0.583,), std=(0.244,))
    out = aug(_img())
    assert set(out) >= {"weak_flag", "global_crops", "global_crops_teacher", "local_crops", "offsets"}
    assert len(out["global_crops"]) == 2 and len(out["local_crops"]) == 8
    for g in out["global_crops"]:
        assert tuple(g.shape) == (1, 224, 224)  # single channel
    for l in out["local_crops"]:
        assert tuple(l.shape) == (1, 96, 96)
    # normalization applied -> not bounded to [0,1], and finite
    assert torch.isfinite(out["global_crops"][0]).all()

def test_dino_gram_crops_optional():
    aug = EMDataAugmentationDINO(global_crops_size=224, local_crops_size=96, gram_teacher_crops_size=256,
                                 mean=(0.5,), std=(0.25,))
    out = aug(_img())
    assert "gram_teacher_crops" in out
    assert tuple(out["gram_teacher_crops"][0].shape) == (1, 256, 256)

def test_dihedral_deterministic_with_generator():
    x = to_float_chw(_img())
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    d = RandomDihedral()
    assert torch.equal(d(x.clone(), generator=g1), d(x.clone(), generator=g2))

def test_normalize_rejects_channel_mismatch():
    # 3-element mean on a 1-channel tensor must raise (never silently expand to RGB).
    aug = EMDataAugmentationDINO(global_crops_size=64, local_crops_size=32, local_crops_number=1,
                                 mean=(0.4, 0.4, 0.4), std=(0.2, 0.2, 0.2))
    import pytest

    with pytest.raises(ValueError):
        aug(_img(80, 80))
