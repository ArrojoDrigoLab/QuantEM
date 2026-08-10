"""DINOv3 1-channel integration + batch contract (skipped if dinov3 is not installed).

These tests require the pinned DINOv3 to be installed (third_party/fetch_dinov3.sh). They assert
the patched SSL path builds a true 1-channel model and that the EM augmentation flows through
DINOv3's collate into the exact batch dict the training step consumes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

dinov3 = pytest.importorskip("dinov3", reason="DINOv3 not installed; run third_party/fetch_dinov3.sh")

@pytest.fixture(scope="module")
def patched():
    import em_ssl.integration.dinov3_patch as p

    p.apply_em_patches()
    return p

def test_verify_one_channel(patched):
    assert patched.verify_one_channel("vit_small") is True

def test_em_aug_builds_from_cfg(patched):
    from dinov3.configs import get_default_config

    cfg = get_default_config()
    cfg.crops.global_crops_size = 224
    cfg.crops.local_crops_size = 96
    cfg.crops.local_crops_number = 4
    cfg.crops.rgb_mean = [0.583]
    cfg.crops.rgb_std = [0.244]
    aug = patched.build_em_data_augmentation(cfg)
    img = Image.fromarray((np.random.rand(560, 560) * 255).astype("uint8"), mode="L")
    out = aug(img)
    assert len(out["global_crops"]) == 2 and tuple(out["global_crops"][0].shape) == (1, 224, 224)
    assert len(out["local_crops"]) == 4

def test_collate_batch_contract(patched):
    from dinov3.configs import get_default_config
    from dinov3.data.collate import collate_data_and_cast
    from dinov3.data.masking import MaskingGenerator

    cfg = get_default_config()
    cfg.crops.global_crops_size = 224
    cfg.crops.local_crops_size = 96
    cfg.crops.local_crops_number = 8
    cfg.crops.rgb_mean = [0.583]
    cfg.crops.rgb_std = [0.244]
    aug = patched.build_em_data_augmentation(cfg)
    imgs = [Image.fromarray((np.random.rand(560, 560) * 255).astype("uint8"), mode="L") for _ in range(2)]
    samples = [(aug(im), ()) for im in imgs]
    img_size, patch = 224, 16
    ntok = (img_size // patch) ** 2
    mg = MaskingGenerator(input_size=(img_size // patch, img_size // patch),
                          max_num_patches=0.5 * img_size // patch * img_size // patch)
    out = collate_data_and_cast(samples, mask_ratio_tuple=(0.1, 0.5), mask_probability=0.5,
                                dtype=torch.float32, n_tokens=ntok, mask_generator=mg)
    assert out["collated_global_crops"].shape == (4, 1, 224, 224)  # 2 global * B=2, 1 channel
    assert out["collated_local_crops"].shape == (16, 1, 96, 96)
    assert out["collated_masks"].shape[1] == ntok and out["collated_masks"].dtype == torch.bool
    for k in ("mask_indices_list", "masks_weight", "upperbound", "n_masked_patches"):
        assert k in out
