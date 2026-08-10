"""Canonical nm/px resampling.

Ported verbatim from ``segmentation_training/dataprep/resample.py``.

This is the one step the reference inference path does **not** perform: its evaluation data was
resampled offline when the derived dataset was built, so ``predict_region``'s docstring states it
"NEVER resamples". Here it has to happen at request time, which makes it the piece of new code that
carries the most scientific risk -- hence the verbatim port and the guard band in ``predict``.

scipy.ndimage.zoom only, order=1 for EM and order=0 for labels. Deliberately **not**
``skimage.transform.resize(anti_aliasing=True)``: training used zoom with no anti-aliasing, and the
difference is a real Dice change on large downsamples.
"""

from __future__ import annotations

import numpy as np

#: Skip the resample when the factor is this close to 1.0 -- avoids needless interpolation blur.
UNIT_EPS = 1e-3


def resample_factors(src_nm_row: float, src_nm_col: float, target_nm: float) -> tuple[float, float]:
    """>1 upsamples (source coarser than target), <1 downsamples."""
    return (float(src_nm_row) / float(target_nm), float(src_nm_col) / float(target_nm))


def is_noop(factors: tuple[float, float]) -> bool:
    return abs(factors[0] - 1.0) < UNIT_EPS and abs(factors[1] - 1.0) < UNIT_EPS


def zoom_image(a: np.ndarray, factors: tuple[float, float]) -> np.ndarray:
    """Bilinear resample of a uint8 EM plane (computed in float, rounded back to uint8)."""
    from scipy import ndimage as ndi

    if is_noop(factors):
        return a
    out = ndi.zoom(a.astype(np.float32), factors, order=1, mode="nearest", grid_mode=False)
    return np.ascontiguousarray(np.clip(np.round(out), 0, 255).astype(np.uint8))


def zoom_labels(a: np.ndarray, factors: tuple[float, float]) -> np.ndarray:
    """Nearest-neighbour resample of a label / instance map -- ids never interpolate."""
    from scipy import ndimage as ndi

    if is_noop(factors):
        return a
    return np.ascontiguousarray(ndi.zoom(a, factors, order=0, mode="nearest", grid_mode=False))


def zoom_probability(a: np.ndarray, factors: tuple[float, float]) -> np.ndarray:
    """Bilinear resample of a float probability map, kept in [0, 1]."""
    from scipy import ndimage as ndi

    if is_noop(factors):
        return a
    out = ndi.zoom(a.astype(np.float32), factors, order=1, mode="nearest", grid_mode=False)
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0).astype(np.float32))
