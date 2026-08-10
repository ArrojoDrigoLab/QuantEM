"""Canonical nm/px resampling — the central step of the segmentation data pipeline.

Every source asset is resampled to a per-organelle canonical nm/px (ER -> 2 nm, mito -> 8 nm; see
``constants.CANONICAL_NM``) so that, for one organelle, a 512 window covers the same physical field
at every source in the corpus and is comparable across arms.

  factor = source_nm_per_px / target_nm_per_px   (per axis; >1 upsamples, <1 downsamples)

EM is resampled bilinearly (order=1); masks and instance-id maps are resampled nearest-neighbour
(order=0) so label values never interpolate (0/1/255 and integer ids stay exact). scipy.ndimage.zoom
only — no numpy-BLAS or skimage, so the module runs on a CPU-only machine.
"""

from __future__ import annotations

import numpy as np

_UNIT_EPS = 1e-3  # skip the resample when the factor is within this of 1.0 (no-op, avoids blur)


def resample_factors(src_nm_row: float, src_nm_col: float, target_nm: float) -> tuple[float, float]:
    return (float(src_nm_row) / float(target_nm), float(src_nm_col) / float(target_nm))


def _zoom_image(a: np.ndarray, factors: tuple[float, float]) -> np.ndarray:
    """Bilinear resample of a uint8 EM plane (compute in float, round back to uint8)."""
    from scipy import ndimage as ndi

    out = ndi.zoom(a.astype(np.float32), factors, order=1, mode="nearest", grid_mode=False)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _zoom_labels(a: np.ndarray, factors: tuple[float, float]) -> np.ndarray:
    """Nearest-neighbour resample of a label / instance map (no interpolation of ids)."""
    from scipy import ndimage as ndi

    return ndi.zoom(a, factors, order=0, mode="nearest", grid_mode=False)


def resample_arrays(em: np.ndarray, mask: np.ndarray, inst: np.ndarray | None,
                    src_nm_row: float, src_nm_col: float, target_nm: float):
    """Resample (em, mask, inst) from (src_nm_row, src_nm_col) to an isotropic ``target_nm`` grid.

    Returns (em2 uint8, mask2 uint8 {0,1,255}, inst2 int32|None, (fr, fc)). A near-unit factor is a
    no-op (returns the inputs unchanged) to avoid needless interpolation blur.
    """
    fr, fc = resample_factors(src_nm_row, src_nm_col, target_nm)
    if abs(fr - 1.0) < _UNIT_EPS and abs(fc - 1.0) < _UNIT_EPS:
        return em, mask, inst, (fr, fc)
    em2 = _zoom_image(em, (fr, fc))
    mask2 = _zoom_labels(mask, (fr, fc)).astype(np.uint8)
    inst2 = None
    if inst is not None:
        inst2 = _zoom_labels(inst.astype(np.int32), (fr, fc)).astype(np.int32)
    # em/mask/inst share input shape + factors -> identical output shape; guard defensively anyway.
    h = min(em2.shape[0], mask2.shape[0])
    w = min(em2.shape[1], mask2.shape[1])
    em2, mask2 = em2[:h, :w], mask2[:h, :w]
    if inst2 is not None:
        inst2 = inst2[:h, :w]
    return np.ascontiguousarray(em2), np.ascontiguousarray(mask2), inst2, (fr, fc)


def resolve_src_nm(sample_extra: dict) -> tuple[float | None, float | None]:
    """Read (row_nm, col_nm) recorded by the extractors; (None, None) for unknown-scale crops."""
    r = sample_extra.get("src_nm_row")
    c = sample_extra.get("src_nm_col")
    r = float(r) if r not in (None, "") else None
    c = float(c) if c not in (None, "") else None
    if r is not None and c is None:
        c = r
    if c is not None and r is None:
        r = c
    return r, c
