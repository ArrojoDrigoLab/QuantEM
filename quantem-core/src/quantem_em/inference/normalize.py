"""EM normalisation. Verbatim from ``segmentation_training/harness/dataset.py::normalize_em``."""

from __future__ import annotations

import numpy as np


def normalize_em(em_uint8: np.ndarray, mean: float, std: float) -> np.ndarray:
    """uint8 [0, 255] -> float32 scaled to [0, 1], then ``(x - mean) / std``.

    No per-tile percentile normalisation: mean/std are the encoder's EM corpus statistics, never
    ImageNet, and never recomputed per image.
    """
    x = em_uint8.astype(np.float32) / 255.0
    return (x - mean) / std
