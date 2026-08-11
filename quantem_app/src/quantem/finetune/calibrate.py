"""Dice-maximising threshold calibration — the cheapest adaptation rung.

This is the capability the manuscript names explicitly:

    "During QuantEM's guided fine-tuning workflow ... user-provided ground-truth
    maps are used to sweep a range of threshold values and select the
    Dice-maximizing threshold."

It fits exactly **one scalar** — the foreground probability threshold — against
the user's own annotations. No gradients, no GPU, no torch: it is numpy over
probability maps the app has already computed. On a laptop it takes seconds,
which makes it the rung that is always available, with head training layered on
top only when the hardware allows.

Ported from ``gk_gold_seg/scripts/finetune_cv/calib.py`` with its discipline
intact:

* **The sweep is fit on the training crops only.** The held-out score is then
  *reported at that threshold* and never used to choose it. Fitting on held-out
  data would inflate the number the user is shown, which is the number they will
  quote.
* **The valid mask is honoured.** Dice is computed only where the user declared
  the region exhaustively annotated. Everything outside a completed ROI is
  ``ignore``, not background — scoring it as background punishes the model for
  finding real objects the user simply did not label.
* **The per-crop oracle is reported as a ceiling.** It is the best achievable if
  you could pick a threshold per crop using the answers, so it is not reachable
  in practice. Showing it stops a user reading the calibrated number as a
  maximum.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

#: The 19-point sweep from the reference: 0.05 to 0.95 inclusive, step 0.05.
DEFAULT_THRESHOLDS: np.ndarray = np.round(np.arange(0.05, 0.96, 0.05), 2)

#: What inference uses when nothing has been calibrated.
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Crop:
    """One annotated region: probabilities, ground truth, and where it counts.

    ``valid`` is the completed-ROI mask. Zero means "the user did not tell us
    what is here" and is excluded from the score entirely.
    """

    name: str
    prob: np.ndarray
    gt: np.ndarray
    valid: np.ndarray | None = None

    def valid_mask(self) -> np.ndarray:
        if self.valid is None:
            return np.ones(self.prob.shape, dtype=bool)
        return self.valid.astype(bool)


def masked_dice(
    prob: np.ndarray, gt: np.ndarray, valid: np.ndarray | None, threshold: float
) -> float | None:
    """Dice between a thresholded probability map and ground truth.

    Returns ``None`` when neither prediction nor truth has any foreground inside
    the valid region — that is "undefined", not zero, and averaging it as zero
    would drag a mean down for crops that contain nothing to find.
    """
    v = np.ones(prob.shape, dtype=bool) if valid is None else valid.astype(bool)
    p = (prob >= threshold) & v
    g = (gt > 0) & v
    denom = int(p.sum()) + int(g.sum())
    if denom == 0:
        return None
    return 2.0 * float((p & g).sum()) / denom


def masked_iou(
    prob: np.ndarray, gt: np.ndarray, valid: np.ndarray | None, threshold: float
) -> float | None:
    """Intersection over union, on the same terms as :func:`masked_dice`.

    Returns ``None`` on the same condition and for the same reason: with no
    predicted and no true foreground inside the valid region the ratio is 0/0,
    which is undefined and not zero. Reported beside Dice because they rank
    identically but do not read identically — a Dice of 0.80 is an IoU of 0.67,
    and a reader who has one number in mind for "good" needs to be told which.
    """
    v = np.ones(prob.shape, dtype=bool) if valid is None else valid.astype(bool)
    p = (prob >= threshold) & v
    g = (gt > 0) & v
    union = int((p | g).sum())
    if union == 0:
        return None
    return float((p & g).sum()) / union


def mean_dice(crops: Sequence[Crop], threshold: float) -> float | None:
    """Unweighted mean Dice over crops — each crop counts once, whatever its size."""
    scores = [masked_dice(c.prob, c.gt, c.valid, threshold) for c in crops]
    scores = [s for s in scores if s is not None]
    return float(np.mean(scores)) if scores else None


def mean_iou(crops: Sequence[Crop], threshold: float) -> float | None:
    """Unweighted mean IoU over crops. Undefined crops are dropped, not zeroed."""
    scores = [masked_iou(c.prob, c.gt, c.valid, threshold) for c in crops]
    scores = [s for s in scores if s is not None]
    return float(np.mean(scores)) if scores else None


@dataclass(frozen=True)
class SweepResult:
    """The full curve, plus the point chosen and how it does held out."""

    thresholds: list[float]
    train_dice: list[float | None]
    calibrated_threshold: float
    train_dice_at_calibrated: float | None
    #: Dice at the *default* 0.5, so the UI can show what calibration bought.
    train_dice_at_default: float | None
    heldout_dice_at_calibrated: float | None = None
    heldout_dice_at_default: float | None = None
    #: Best achievable with a per-crop threshold chosen using the answers.
    #: A ceiling, not a target.
    heldout_oracle: float | None = None
    per_crop: dict[str, float | None] = field(default_factory=dict)
    #: Names of the crops the threshold was fit on, so the UI can badge them.
    train_crop_names: list[str] = field(default_factory=list)
    heldout_crop_names: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float | None:
        if self.heldout_dice_at_calibrated is None or self.heldout_dice_at_default is None:
            return None
        return self.heldout_dice_at_calibrated - self.heldout_dice_at_default

    def as_dict(self) -> dict[str, object]:
        return {
            "thresholds": self.thresholds,
            "train_dice": self.train_dice,
            "calibrated_threshold": self.calibrated_threshold,
            "train_dice_at_calibrated": self.train_dice_at_calibrated,
            "train_dice_at_default": self.train_dice_at_default,
            "heldout_dice_at_calibrated": self.heldout_dice_at_calibrated,
            "heldout_dice_at_default": self.heldout_dice_at_default,
            "heldout_oracle": self.heldout_oracle,
            "improvement": self.improvement,
            "per_crop": self.per_crop,
            "train_crop_names": self.train_crop_names,
            "heldout_crop_names": self.heldout_crop_names,
        }


def sweep_threshold(
    train_crops: Sequence[Crop],
    *,
    heldout_crops: Sequence[Crop] = (),
    thresholds: np.ndarray | Sequence[float] = DEFAULT_THRESHOLDS,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> SweepResult:
    """Choose the threshold that maximises mean Dice on ``train_crops``.

    ``heldout_crops`` are only ever *scored*, never used to choose. Pass them
    when the user annotated regions on more than one image; with a single image
    the app must say the number is not image-disjoint.
    """
    if not train_crops:
        raise ValueError("threshold calibration needs at least one annotated region")

    thrs = [float(t) for t in np.asarray(thresholds, dtype=float).ravel()]
    curve = [mean_dice(train_crops, t) for t in thrs]

    scored = [(t, d) for t, d in zip(thrs, curve, strict=True) if d is not None]
    if not scored:
        raise ValueError("no annotated region contains foreground; Dice is undefined everywhere")
    best_thr, best_train = max(scored, key=lambda kv: kv[1])

    heldout_cal = mean_dice(heldout_crops, best_thr) if heldout_crops else None
    heldout_def = mean_dice(heldout_crops, default_threshold) if heldout_crops else None

    oracle = None
    if heldout_crops:
        per_crop_best = []
        for c in heldout_crops:
            scores = [masked_dice(c.prob, c.gt, c.valid, t) for t in thrs]
            scores = [s for s in scores if s is not None]
            if scores:
                per_crop_best.append(max(scores))
        oracle = float(np.mean(per_crop_best)) if per_crop_best else None

    per_crop = {
        c.name: masked_dice(c.prob, c.gt, c.valid, best_thr)
        for c in list(train_crops) + list(heldout_crops)
    }

    return SweepResult(
        thresholds=thrs,
        train_dice=curve,
        calibrated_threshold=best_thr,
        train_dice_at_calibrated=best_train,
        train_dice_at_default=mean_dice(train_crops, default_threshold),
        heldout_dice_at_calibrated=heldout_cal,
        heldout_dice_at_default=heldout_def,
        heldout_oracle=oracle,
        per_crop=per_crop,
        train_crop_names=[c.name for c in train_crops],
        heldout_crop_names=[c.name for c in heldout_crops],
    )


def split_crops(
    crops: Sequence[Crop], *, image_of: dict[str, str] | None = None
) -> tuple[list[Crop], list[Crop], str]:
    """Split annotated crops into fit and held-out sets.

    Prefers an **image-disjoint** split: crops from different source images go to
    different sides, so the held-out score measures generalisation to a new image.
    With only one annotated image that is impossible, and the caller is told so
    via the returned mode string — the UI must label which kind of number it is
    showing rather than presenting them as equivalent.
    """
    crops = list(crops)
    if len(crops) < 2:
        return crops, [], "no-heldout"

    if image_of:
        by_image: dict[str, list[Crop]] = {}
        for c in crops:
            by_image.setdefault(image_of.get(c.name, c.name), []).append(c)
        if len(by_image) >= 2:
            keys = sorted(by_image)
            cut = max(1, len(keys) // 2)
            train = [c for k in keys[:cut] for c in by_image[k]]
            held = [c for k in keys[cut:] for c in by_image[k]]
            return train, held, "image-disjoint"

    cut = max(1, len(crops) // 2)
    return crops[:cut], crops[cut:], "within-image"
