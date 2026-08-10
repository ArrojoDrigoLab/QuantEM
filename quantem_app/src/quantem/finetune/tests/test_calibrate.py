"""Threshold-calibration tests. Pure numpy — no torch, no database."""

from __future__ import annotations

import numpy as np
import pytest

from quantem.finetune.calibrate import (
    DEFAULT_THRESHOLDS,
    Crop,
    masked_dice,
    mean_dice,
    split_crops,
    sweep_threshold,
)

H = W = 64


def _disc(r=12, cx=32, cy=32):
    yy, xx = np.mgrid[0:H, 0:W]
    return (((yy - cy) ** 2 + (xx - cx) ** 2) < r * r).astype(np.uint8)


class TestMaskedDice:
    """A direct port of ``gk_base_eval.py::selftest`` — the model-free proof that
    the metric itself is right. The reference shipped this as an assert-driven
    function; it is a test here."""

    def test_perfect_prediction_is_one(self):
        gt = _disc()
        valid = np.ones((H, W), np.uint8)
        assert masked_dice(gt.astype(float), gt, valid, 0.5) == pytest.approx(1.0)

    def test_empty_prediction_against_foreground_is_zero(self):
        gt = _disc()
        valid = np.ones((H, W), np.uint8)
        assert masked_dice(np.zeros((H, W)), gt, valid, 0.5) == 0.0

    def test_no_foreground_anywhere_is_undefined_not_zero(self):
        valid = np.ones((H, W), np.uint8)
        assert masked_dice(np.zeros((H, W)), np.zeros((H, W)), valid, 0.5) is None

    def test_prediction_outside_the_valid_region_is_ignored(self):
        """The whole point of the completed-ROI contract: a model firing where
        the user never annotated must not be punished."""
        gt = _disc()
        pred = np.ones((H, W), dtype=float)
        valid = np.zeros((H, W), np.uint8)
        valid[gt > 0] = 1  # the annotated region is exactly the object
        assert masked_dice(pred, gt, valid, 0.5) == pytest.approx(1.0)

    def test_missing_valid_mask_means_everything_counts(self):
        gt = _disc()
        pred = np.ones((H, W), dtype=float)
        full = masked_dice(pred, gt, None, 0.5)
        assert full is not None and full < 0.5  # heavily penalised, as it should be

    def test_mean_dice_skips_undefined_crops(self):
        gt = _disc()
        valid = np.ones((H, W), np.uint8)
        good = Crop("a", gt.astype(float), gt, valid)
        empty = Crop("b", np.zeros((H, W)), np.zeros((H, W), np.uint8), valid)
        assert mean_dice([good, empty], 0.5) == pytest.approx(1.0)


def _crop(name, *, best_thr, noise_seed=0):
    """A crop whose probability map is maximally separated at ``best_thr``."""
    gt = _disc()
    rng = np.random.default_rng(noise_seed)
    prob = np.where(gt > 0, best_thr + 0.05, best_thr - 0.05)
    prob = np.clip(prob + rng.normal(0, 0.01, (H, W)), 0, 1)
    return Crop(name, prob, gt, np.ones((H, W), np.uint8))


class TestSweep:
    def test_finds_the_separating_threshold(self):
        crops = [_crop(f"c{i}", best_thr=0.7, noise_seed=i) for i in range(3)]
        r = sweep_threshold(crops)
        assert r.calibrated_threshold == pytest.approx(0.70, abs=0.051)
        assert r.train_dice_at_calibrated == pytest.approx(1.0, abs=0.02)

    def test_calibration_beats_the_default_when_the_model_is_miscalibrated(self):
        train = [_crop(f"t{i}", best_thr=0.8, noise_seed=i) for i in range(2)]
        held = [_crop(f"h{i}", best_thr=0.8, noise_seed=10 + i) for i in range(2)]
        r = sweep_threshold(train, heldout_crops=held)
        assert r.heldout_dice_at_default is not None
        assert r.improvement > 0.3

    def test_threshold_is_never_fit_on_heldout(self):
        """Held-out crops that prefer a different threshold must not move the
        chosen value at all."""
        train = [_crop(f"t{i}", best_thr=0.3, noise_seed=i) for i in range(2)]
        r_alone = sweep_threshold(train)
        held = [_crop(f"h{i}", best_thr=0.9, noise_seed=20 + i) for i in range(4)]
        r_with = sweep_threshold(train, heldout_crops=held)
        assert r_with.calibrated_threshold == r_alone.calibrated_threshold

    def test_oracle_is_a_ceiling_not_below_the_calibrated_score(self):
        train = [_crop(f"t{i}", best_thr=0.6, noise_seed=i) for i in range(2)]
        held = [_crop("h0", best_thr=0.4, noise_seed=7), _crop("h1", best_thr=0.85, noise_seed=8)]
        r = sweep_threshold(train, heldout_crops=held)
        assert r.heldout_oracle >= r.heldout_dice_at_calibrated - 1e-9

    def test_curve_covers_the_documented_sweep(self):
        r = sweep_threshold([_crop("a", best_thr=0.5)])
        assert len(r.thresholds) == len(DEFAULT_THRESHOLDS) == 19
        assert r.thresholds[0] == pytest.approx(0.05)
        assert r.thresholds[-1] == pytest.approx(0.95)

    def test_records_which_crops_were_fit_on(self):
        train = [_crop("t0", best_thr=0.5)]
        held = [_crop("h0", best_thr=0.5, noise_seed=3)]
        r = sweep_threshold(train, heldout_crops=held)
        assert r.train_crop_names == ["t0"]
        assert r.heldout_crop_names == ["h0"]
        assert set(r.per_crop) == {"t0", "h0"}

    def test_requires_at_least_one_region(self):
        with pytest.raises(ValueError, match="at least one annotated region"):
            sweep_threshold([])

    def test_all_empty_annotations_is_a_clear_error(self):
        valid = np.ones((H, W), np.uint8)
        blank = Crop("b", np.zeros((H, W)), np.zeros((H, W), np.uint8), valid)
        with pytest.raises(ValueError, match="Dice is undefined"):
            sweep_threshold([blank])


class TestSplit:
    def test_prefers_image_disjoint(self):
        crops = [_crop(n, best_thr=0.5) for n in ("a1", "a2", "b1", "b2")]
        image_of = {"a1": "imgA", "a2": "imgA", "b1": "imgB", "b2": "imgB"}
        train, held, mode = split_crops(crops, image_of=image_of)
        assert mode == "image-disjoint"
        assert {c.name for c in train} == {"a1", "a2"}
        assert {c.name for c in held} == {"b1", "b2"}

    def test_single_image_is_labelled_within_image(self):
        crops = [_crop(n, best_thr=0.5) for n in ("a1", "a2")]
        image_of = {"a1": "imgA", "a2": "imgA"}
        _, held, mode = split_crops(crops, image_of=image_of)
        assert mode == "within-image"
        assert held  # still produces a held-out set, just a weaker one

    def test_one_crop_has_no_heldout(self):
        train, held, mode = split_crops([_crop("only", best_thr=0.5)])
        assert mode == "no-heldout" and not held and len(train) == 1
