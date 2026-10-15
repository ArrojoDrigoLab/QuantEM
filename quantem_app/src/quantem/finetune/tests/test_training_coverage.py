"""Regression checks for sparse supervision and training-data coverage."""

from types import SimpleNamespace

import numpy as np
import pytest

from quantem.finetune.adapt import (
    IGNORE,
    AdaptConfig,
    build_patches,
    iter_windows,
    masks_to_model_scale,
    steps_for_training_tiles,
)
from quantem.finetune.calibrate import Crop, sweep_threshold
from quantem.finetune.options import parse_options, validation_split
from quantem.inference.resample import plan_resample


def test_irregular_edges_and_a_single_pixel_are_covered():
    valid = np.ones((633, 799), dtype=np.uint8)
    covered = np.zeros_like(valid)
    for y, x in iter_windows(valid, 512):
        covered[y : y + 512, x : x + 512] = 1
    assert covered.all()
    valid[:] = 0
    valid[-1, -1] = 1
    assert list(iter_windows(valid, 512)) == [(121, 287)]


def test_downsampling_cannot_erase_a_small_complete_background_area():
    valid = np.zeros((1024, 1024), dtype=np.uint8)
    valid[507, 509] = 1
    gt = np.zeros_like(valid)
    context = plan_resample(valid.shape, 2.5, 8)
    out_gt, out_valid = masks_to_model_scale(gt, valid, context)
    assert out_valid.any()
    assert not out_gt.any()
    crop = SimpleNamespace(
        name="small", em=np.zeros(out_gt.shape, np.uint8), gt=out_gt, valid=out_valid
    )
    patches = build_patches([crop], 512, image_mean=0, image_std=1)
    assert patches
    assert all(set(np.unique(target)) == {0, IGNORE} for _, target in patches)


def test_resampling_does_not_mix_unannotated_background_into_a_label():
    valid = np.zeros((64, 64), np.uint8)
    valid[19, 27] = 1
    gt, out_valid = masks_to_model_scale(valid, valid, plan_resample(valid.shape, 1, 8))
    assert out_valid.any()
    assert np.array_equal(gt, out_valid)


def test_negative_only_threshold_calibration_keeps_default():
    c = Crop(name="negative", prob=np.zeros((8, 8)), gt=np.zeros((8, 8)), valid=np.ones((8, 8)))
    assert sweep_threshold([c]).calibrated_threshold == 0.5


def test_large_training_sets_receive_at_least_one_full_pass():
    assert steps_for_training_tiles(1300) == 1300
    assert steps_for_training_tiles(361, schedule="extended") == 7220


@pytest.mark.parametrize(
    "options",
    [
        {"lr": float("nan")},
        {"lr": -1},
        {"steps": 1.5},
        {"steps": 20001},
        {"eval_interval": 0},
        {"validation_fraction": 0.8},
        {"seed": -1},
    ],
)
def test_invalid_controls_are_rejected(options):
    with pytest.raises(ValueError):
        parse_options(options)


def test_validation_groups_whole_training_images_and_is_repeatable():
    crops = [SimpleNamespace(image_key=f"im{i // 2}", name=f"c{i}") for i in range(20)]
    train, val = validation_split(crops, 0.2, 12)
    assert len(val) == 4
    assert not {c.image_key for c in train} & {c.image_key for c in val}
    assert validation_split(crops, 0.2, 12) == (train, val)
    assert validation_split(crops[:2], 0.2, 12) == (crops[:2], [])


def test_ignored_logits_have_zero_gradient_and_empty_annotations_have_signal():
    torch = pytest.importorskip("torch")
    from quantem.finetune.adapt import head_loss

    logits = torch.zeros((1, 2, 8, 8), requires_grad=True)
    target = torch.full((8, 8), IGNORE, dtype=torch.long)
    target[3, 3] = 0
    head_loss(logits, target).backward()
    assert logits.grad[0, 1, 3, 3] > 0  # gradient descent suppresses false foreground
    assert torch.count_nonzero(logits.grad) == 2


def test_shuffled_passes_cover_every_patch_before_repeating():
    pytest.importorskip("torch")
    from quantem.finetune.adapt import train_head
    from quantem.finetune.tests.test_adapt_head import TinyModel, _patches

    result = train_head(TinyModel(), _patches(9), config=AdaptConfig(steps=9))
    assert result.patches_seen == 9
    assert result.as_dict()["coverage_fraction"] == 1


def test_early_stopping_restores_the_best_validation_checkpoint():
    torch = pytest.importorskip("torch")
    from quantem.finetune.adapt import train_head
    from quantem.finetune.tests.test_adapt_head import TinyModel, _patches

    torch.manual_seed(6)
    first = TinyModel()
    second = TinyModel()
    second.load_state_dict(first.state_dict())
    patches = _patches(2)
    # A deliberately unreachable improvement threshold fixes the first
    # evaluation as best and makes restoration directly verifiable.
    config = AdaptConfig(steps=10, eval_interval=1, patience=2, min_delta=100)
    result = train_head(first, patches, validation_patches=patches, config=config)
    train_head(second, patches, config=AdaptConfig(steps=1))
    assert result.stopped_early and result.selected_step == 1 and result.steps == 3
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])
