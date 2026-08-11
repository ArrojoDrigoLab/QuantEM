"""Training-patch extraction. Pure numpy — this half of the trainer needs no torch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from quantem.finetune.adapt import (
    IGNORE,
    AdaptConfig,
    augment,
    build_patches,
    tile_for,
)
from quantem.inference.engine import normalize_tile

MEAN, STD = 0.583175, 0.244468


@dataclass
class FakeCrop:
    name: str
    em: np.ndarray
    gt: np.ndarray
    valid: np.ndarray


def _crop(size: int = 1024, *, valid_slice=None) -> FakeCrop:
    rng = np.random.default_rng(0)
    em = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    gt = np.zeros((size, size), dtype=np.uint8)
    gt[size // 4 : size // 2, size // 4 : size // 2] = 1
    valid = np.zeros((size, size), dtype=np.uint8)
    if valid_slice is None:
        valid[:] = 1
    else:
        valid[valid_slice] = 1
    gt &= valid
    return FakeCrop(name="crop", em=em, gt=gt, valid=valid)


class TestTileSize:
    def test_patch_16_models_use_512(self):
        assert tile_for(512, 16) == 512

    def test_patch_14_models_use_518(self):
        assert tile_for(518, 14) == 518

    def test_a_tile_is_rounded_up_to_whole_patches(self):
        assert tile_for(500, 16) == 512
        assert tile_for(512, 14) == 518


class TestBuildPatches:
    def test_stride_is_half_a_tile(self):
        patches = build_patches([_crop(1024)], 512, image_mean=MEAN, image_std=STD)
        # starts at 0, 256, 512 on each axis -> 3 x 3
        assert len(patches) == 9
        assert all(image.shape == (512, 512) for image, _ in patches)

    def test_windows_barely_inside_the_completed_area_are_dropped(self):
        """The 20 % rule: a window that is almost all ``ignore`` is not a step,
        it is noise."""
        crop = _crop(1024, valid_slice=(slice(0, 200), slice(0, 200)))
        patches = build_patches([crop], 512, image_mean=MEAN, image_std=STD)
        assert patches == []

    def test_unannotated_pixels_become_ignore_not_background(self):
        crop = _crop(1024, valid_slice=(slice(0, 700), slice(0, 700)))
        patches = build_patches([crop], 512, image_mean=MEAN, image_std=STD)
        assert patches
        targets = np.concatenate([t.ravel() for _, t in patches])
        assert set(np.unique(targets)) <= {0, 1, IGNORE}
        assert (targets == IGNORE).any()

    def test_images_are_normalised_with_the_encoder_statistics(self):
        crop = _crop(512)
        ((image, _),) = build_patches([crop], 512, image_mean=MEAN, image_std=STD)
        assert image.dtype == np.float32
        np.testing.assert_allclose(image, normalize_tile(crop.em, MEAN, STD), rtol=1e-6)

    def test_a_crop_smaller_than_a_tile_is_padded_and_the_padding_is_ignored(self):
        crop = _crop(600)
        patches = build_patches([crop], 512, image_mean=MEAN, image_std=STD)
        assert patches  # 600 px of valid data clears 20 % of a 512 tile
        crop_small = _crop(200)
        assert build_patches([crop_small], 512, image_mean=MEAN, image_std=STD) == []

    def test_a_crop_with_no_pixels_loaded_is_a_clear_error(self):
        crop = _crop(512)
        crop.em = None
        with pytest.raises(ValueError, match="no EM pixels"):
            build_patches([crop], 512, image_mean=MEAN, image_std=STD)

    def test_the_valid_fraction_rule_is_configurable(self):
        crop = _crop(1024, valid_slice=(slice(0, 200), slice(0, 200)))
        patches = build_patches(
            [crop],
            512,
            image_mean=MEAN,
            image_std=STD,
            config=AdaptConfig(min_valid_fraction=0.1),
        )
        assert len(patches) == 1


class TestAugment:
    def test_image_and_target_stay_registered(self):
        """A flip applied to one and not the other silently trains on garbage."""
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        target = image.astype(np.int64)
        rng = np.random.default_rng(7)
        for _ in range(20):
            out_image, out_target = augment(image, target, rng)
            np.testing.assert_array_equal(out_image.astype(np.int64), out_target)

    def test_augmentation_is_dihedral_only(self):
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        rng = np.random.default_rng(3)
        for _ in range(20):
            out_image, _ = augment(image, image.astype(np.int64), rng)
            # Same pixels, rearranged: no interpolation, no new values.
            np.testing.assert_array_equal(np.sort(out_image.ravel()), np.sort(image.ravel()))
