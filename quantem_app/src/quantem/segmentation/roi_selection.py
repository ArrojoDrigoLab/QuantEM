"""
ROI selection utilities for ROI-first bootstrapping.
"""

import logging
from dataclasses import dataclass

import numpy as np

from quantem.assets.task_utils import load_image_preview_array

logger = logging.getLogger(__name__)


@dataclass
class RoiSelectionResult:
    x: int
    y: int
    width: int
    height: int
    score: float


def _choose_scored_candidate(
    top_candidates: list[tuple[float, int, int]],
    *,
    seed: int | None = None,
) -> tuple[float, int, int]:
    finite_candidates = [
        candidate
        for candidate in top_candidates
        if candidate[0] > 0 and np.isfinite(candidate[0])
    ]
    if not finite_candidates:
        return top_candidates[0]

    scores = np.array([item[0] for item in finite_candidates], dtype=float)
    total_score = float(np.sum(scores))
    if not np.isfinite(total_score) or total_score <= 0:
        return finite_candidates[0]

    weights = scores / total_score
    weight_sum = float(np.sum(weights))
    if not np.all(np.isfinite(weights)) or weight_sum <= 0:
        return finite_candidates[0]

    weights = weights / weight_sum
    rng = np.random.default_rng(seed)
    choice_idx = int(rng.choice(len(finite_candidates), p=weights))
    return finite_candidates[choice_idx]


def _score_window(
    window: np.ndarray,
    *,
    black_threshold: float,
    bright_threshold: float,
    global_contrast: float,
) -> float:
    """Score a window based on content coverage, contrast, and texture."""
    if window.size == 0:
        return -1.0
    non_black = window > black_threshold
    non_black_fraction = float(np.mean(non_black))
    if non_black_fraction < 0.65:
        return -1.0

    values = window[non_black]
    if values.size < 10:
        return -1.0

    contrast = float(np.percentile(values, 95) - np.percentile(values, 5))
    variance = float(np.var(values))
    texture = np.sqrt(max(variance, 0.0))

    normalized_contrast = contrast / (global_contrast + 1e-6)
    normalized_contrast = float(np.clip(normalized_contrast, 0.0, 1.0))

    bright_fraction = float(np.mean(values >= bright_threshold))
    bright_penalty = bright_fraction * (1.0 - normalized_contrast)

    score = (non_black_fraction**2) * (contrast + 1e-6) * (1.0 - bright_penalty)
    score *= 0.5 + 0.5 * normalized_contrast
    score *= 0.5 + 0.5 * (texture / (texture + 10.0))
    return score


def select_roi_for_image(
    image,
    roi_size: int = 3000,
    preview_max_size: int = 1024,
    seed: int | None = None,
) -> RoiSelectionResult:
    """
    Select an ROI for an image using a downsampled preview.

    The method samples a grid of candidate windows and scores them based on
    non-black coverage and intensity variance.
    """
    preview = load_image_preview_array(image, max_size=preview_max_size)
    preview_height, preview_width = preview.shape

    roi_width = min(roi_size, image.width)
    roi_height = min(roi_size, image.height)

    scale_x = image.width / max(preview_width, 1)
    scale_y = image.height / max(preview_height, 1)

    roi_width_preview = max(1, int(round(roi_width / scale_x)))
    roi_height_preview = max(1, int(round(roi_height / scale_y)))

    step_x = max(1, roi_width_preview // 4)
    step_y = max(1, roi_height_preview // 4)

    black_threshold = float(np.percentile(preview, 2))
    black_threshold = max(3.0, black_threshold)
    bright_threshold = float(np.percentile(preview, 98))
    global_contrast = float(np.percentile(preview, 95) - np.percentile(preview, 5))

    best_score = -1.0
    best_x = 0
    best_y = 0

    content_y_min = 0
    content_x_min = 0
    content_y_max = preview_height - 1
    content_x_max = preview_width - 1

    non_black_mask = preview > black_threshold
    if np.any(non_black_mask):
        ys, xs = np.where(non_black_mask)
        content_y_min = int(np.min(ys))
        content_y_max = int(np.max(ys))
        content_x_min = int(np.min(xs))
        content_x_max = int(np.max(xs))

    margin_x = max(1, roi_width_preview // 8)
    margin_y = max(1, roi_height_preview // 8)

    x_start = max(0, content_x_min - margin_x)
    y_start = max(0, content_y_min - margin_y)
    x_end = min(preview_width - roi_width_preview, content_x_max + margin_x)
    y_end = min(preview_height - roi_height_preview, content_y_max + margin_y)

    if x_end < x_start:
        x_start = 0
        x_end = max(preview_width - roi_width_preview, 0)
    if y_end < y_start:
        y_start = 0
        y_end = max(preview_height - roi_height_preview, 0)

    y_positions = list(range(y_start, max(y_end + 1, 1), step_y))
    x_positions = list(range(x_start, max(x_end + 1, 1), step_x))

    if not y_positions:
        y_positions = [0]
    if not x_positions:
        x_positions = [0]

    candidates = []
    for y in y_positions:
        for x in x_positions:
            window = preview[y : y + roi_height_preview, x : x + roi_width_preview]
            score = _score_window(
                window,
                black_threshold=black_threshold,
                bright_threshold=bright_threshold,
                global_contrast=global_contrast,
            )
            if score > 0:
                candidates.append((score, x, y))
            if score > best_score:
                best_score = score
                best_x = x
                best_y = y

    if candidates:
        candidates.sort(reverse=True, key=lambda item: item[0])
        top_candidates = candidates[:5]
        best_score, best_x, best_y = _choose_scored_candidate(
            top_candidates,
            seed=seed,
        )

    if best_score < 0:
        best_x = max(0, (preview_width - roi_width_preview) // 2)
        best_y = max(0, (preview_height - roi_height_preview) // 2)
        best_score = 0.0

    x_full = int(round(best_x * scale_x))
    y_full = int(round(best_y * scale_y))

    x_full = min(max(0, x_full), max(0, image.width - roi_width))
    y_full = min(max(0, y_full), max(0, image.height - roi_height))

    logger.info(
        "Selected ROI for image %s at %s,%s (%sx%s), score=%.4f",
        image.id,
        x_full,
        y_full,
        roi_width,
        roi_height,
        best_score,
    )

    return RoiSelectionResult(
        x=x_full,
        y=y_full,
        width=roi_width,
        height=roi_height,
        score=best_score,
    )
