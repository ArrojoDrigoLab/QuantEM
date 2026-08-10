"""Get an arbitrary user image into the uint8 single-channel form the models were trained on.

The corpus rule (``normalization_scope="source"``): non-uint8 input is stretched to uint8 using the 0.1 / 99.9
percentiles of the finite, non-zero-padding pixels. uint8 passes through untouched — it is already
what training saw.
"""

from __future__ import annotations

import warnings

import numpy as np

#: Percentiles for the intensity stretch. Matches the corpus tiling rule.
LO_PCT, HI_PCT = 0.1, 99.9

#: Luminance weights for the RGB -> grey fallback (Rec. 601, matching PIL's "L" conversion).
_RGB_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_uint8(image: np.ndarray, *, invert: bool = False) -> tuple[np.ndarray, dict]:
    """Return ``(uint8 [H, W], provenance)``.

    Parameters
    ----------
    invert
        Flip contrast. **Never automatic** — the corpus policy is ``invert_policy="auto_report_only"``,
        so this is a user decision only.
    """
    a = np.asarray(image)
    info: dict = {"input_dtype": str(a.dtype), "input_shape": tuple(a.shape)}

    if a.ndim == 3:
        if a.shape[-1] in (3, 4):
            warnings.warn(
                "RGB(A) input converted to luminance; these models are single-channel EM models.",
                stacklevel=2,
            )
            a = (a[..., :3].astype(np.float32) * _RGB_WEIGHTS).sum(-1)
            info["rgb_to_luminance"] = True
        elif a.shape[0] == 1:
            a = a[0]
        else:
            raise ValueError(
                f"expected a 2-D image or an RGB(A) image, got shape {tuple(np.asarray(image).shape)}"
            )
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D image after channel handling, got {a.ndim}-D")

    if a.dtype == np.uint8:
        info["intensity_rescale"] = "none"
        out = a
    else:
        finite = np.isfinite(a)
        vals = a[finite]
        if vals.size == 0:
            raise ValueError("image has no finite pixels")
        lo, hi = np.percentile(vals.astype(np.float64), [LO_PCT, HI_PCT])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            out = np.zeros(a.shape, dtype=np.uint8)
        else:
            x = np.clip((a.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
            x[~finite] = 0.0
            out = np.round(x * 255.0).astype(np.uint8)
        info["intensity_rescale"] = f"percentile_{LO_PCT}_{HI_PCT}"
        info["intensity_range"] = (float(lo), float(hi))

    if invert:
        out = (255 - out).astype(np.uint8)
    info["inverted"] = bool(invert)
    return np.ascontiguousarray(out), info
