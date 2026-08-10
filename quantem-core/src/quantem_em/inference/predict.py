"""Tiled prediction — the function that has to match the published numbers.

``predict_region`` is a faithful port of ``segmentation_training/harness/evaluate.py::predict_region``
with the training-only branches removed (conditioning context, position-dependent aux handling for
the panoptic decoder, which we do not ship).

``predict_image`` wraps it with the two steps the reference never performs, because its evaluation
data was pre-resampled offline: resampling in to the model's training resolution, and back out to
the caller's grid.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from ..spec import ModelSpec
from .normalize import normalize_em
from .resample import resample_factors, zoom_probability
from .tiling import hann2d, pad_to_tile, round_up, stride_for, window_starts

#: Warn above this upsample factor, refuse above the hard limit without an explicit override.
#: Grounded in FIG4_CAVEATS C1: platynereis at 80 nm upsampled 3.2x to 25 nm collapsed OmniEM
#: nucleus Dice to 0.250 against NucleoNet's 0.925.
UPSAMPLE_WARN = 2.0
UPSAMPLE_MAX = 8.0
#: Warn when the resampled working side exceeds this, mirroring the corpus build cap.
MAX_SIDE_WARN = 8192


class InferenceCancelled(RuntimeError):
    """Raised when a caller's ``cancel()`` returns True between windows."""


def predict_region(
    model,
    em_uint8: np.ndarray,
    spec: ModelSpec,
    device,
    *,
    collect_aux: bool = False,
    progress=None,
    cancel=None,
) -> tuple[np.ndarray, list[np.ndarray] | None]:
    """Sliding-window foreground probability over an EM region already at the model's scale.

    Returns ``(fg [H, W] float32, aux | None)``.
    """
    import torch

    patch = spec.encoder.patch_size
    t = round_up(spec.tile_size, patch)
    em_p, (h0, w0) = pad_to_tile(np.asarray(em_uint8), t, patch)
    h, w = em_p.shape

    stride = stride_for(t, spec.overlap)
    win = hann2d(t)
    xnorm = normalize_em(em_p, spec.encoder.dataset_mean, spec.encoder.dataset_std)

    k = spec.num_classes
    acc = np.zeros((k, h, w), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)
    aux_acc: list[np.ndarray] | None = None

    ys = window_starts(h, t, stride)
    xs = window_starts(w, t, stride)
    total = len(ys) * len(xs)
    done = 0

    for y in ys:
        for x0 in xs:
            if cancel is not None and cancel():
                raise InferenceCancelled(f"cancelled after {done}/{total} windows")
            tile = np.ascontiguousarray(xnorm[y : y + t, x0 : x0 + t])
            xt = torch.from_numpy(tile)[None, None].to(device)
            with torch.no_grad():
                logits = model(xt)
                probs = torch.softmax(logits[0].float(), dim=0).cpu().numpy()
            acc[:, y : y + t, x0 : x0 + t] += probs * win[None]
            wsum[y : y + t, x0 : x0 + t] += win

            if collect_aux:
                auxs = getattr(model, "aux_logits", None) or []
                if aux_acc is None:
                    aux_acc = [np.zeros((int(a.shape[1]), h, w), dtype=np.float32) for a in auxs]
                for i, a in enumerate(auxs):
                    aux_acc[i][:, y : y + t, x0 : x0 + t] += a[0].float().cpu().numpy() * win[None]

            done += 1
            if progress is not None:
                progress(done, total)

    probs = (acc / np.maximum(wsum, 1e-6)[None])[:, :h0, :w0]
    fg = (probs[1] if k == 2 else probs[1:].max(axis=0)).astype(np.float32)

    aux_out = None
    if collect_aux and aux_acc is not None:
        wn = np.maximum(wsum[:h0, :w0], 1e-6)[None]
        aux_out = [(a[:, :h0, :w0] / wn).astype(np.float32) for a in aux_acc]
    return fg, aux_out


def plan_resample(
    spec: ModelSpec,
    shape: tuple[int, int],
    pixel_size_nm: float | tuple[float, float] | None,
    *,
    allow_extreme: bool = False,
) -> tuple[tuple[float, float] | None, dict]:
    """Decide the resample factors and report what it means, before any work happens.

    Returns ``(factors | None, info)``. ``None`` means run at native resolution — either because the
    model has no canonical scale (ER) or because the caller gave no pixel size.
    """
    info: dict = {
        "model_pixel_size_nm": spec.canonical_nm,
        "source_pixel_size_nm": None,
        "resample_factor": None,
        "resampled": False,
        "warnings": [],
    }
    if spec.canonical_nm is None:
        info["reason"] = "model runs at native resolution"
        return None, info
    if pixel_size_nm is None:
        # Owner ruling: never infer a pixel size, and never rescale without one.
        info["reason"] = "no pixel size supplied -- running at native resolution"
        info["warnings"].append(
            f"No pixel size given, so the image is used as-is. This model was trained at "
            f"{spec.canonical_nm:g} nm/px; supplying a pixel size usually improves results."
        )
        return None, info

    if np.isscalar(pixel_size_nm):
        src_r = src_c = float(pixel_size_nm)
    else:
        src_r, src_c = (float(v) for v in pixel_size_nm)
    if src_r <= 0 or src_c <= 0:
        raise ValueError(f"pixel size must be positive, got {pixel_size_nm!r}")

    factors = resample_factors(src_r, src_c, spec.canonical_nm)
    info["source_pixel_size_nm"] = (src_r, src_c)
    info["resample_factor"] = factors

    up = max(factors)
    if up > UPSAMPLE_MAX and not allow_extreme:
        raise ValueError(
            f"refusing to upsample {up:.1f}x ({src_r:g} -> {spec.canonical_nm:g} nm/px). Beyond "
            f"{UPSAMPLE_MAX:g}x this reliably produces poor segmentations. Pass "
            "allow_extreme=True to override."
        )
    if up > UPSAMPLE_WARN:
        info["warnings"].append(
            f"Upsampling {up:.1f}x to reach {spec.canonical_nm:g} nm/px. Large upsamples degrade "
            "accuracy substantially; results should be treated with caution."
        )
    out_hw = (max(1, round(shape[0] * factors[0])), max(1, round(shape[1] * factors[1])))
    info["working_shape"] = out_hw
    if max(out_hw) > MAX_SIDE_WARN:
        info["warnings"].append(
            f"Resampled working size is {out_hw[0]}x{out_hw[1]} px; this will be slow and memory-hungry."
        )
    info["resampled"] = True
    return factors, info


def predict_image(
    model,
    image: np.ndarray,
    spec: ModelSpec,
    device,
    *,
    pixel_size_nm=None,
    invert: bool = False,
    collect_aux: bool = False,
    allow_extreme_resample: bool = False,
    progress=None,
    cancel=None,
) -> tuple[np.ndarray, dict]:
    """Full path: prepare -> resample in -> tiled predict -> resample out.

    Returns ``(probability [H, W] float32 on the ORIGINAL grid, contract)``.
    """
    from .prepare import to_uint8
    from .resample import zoom_image

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    em, prep_info = to_uint8(image, invert=invert)
    timings["prepare"] = time.perf_counter() - t0

    factors, rs_info = plan_resample(
        spec, em.shape, pixel_size_nm, allow_extreme=allow_extreme_resample
    )
    for msg in rs_info["warnings"]:
        warnings.warn(msg, stacklevel=2)

    original_shape = em.shape
    t0 = time.perf_counter()
    if factors is not None:
        em = zoom_image(em, factors)
    timings["resample_in"] = time.perf_counter() - t0

    t = round_up(spec.tile_size, spec.encoder.patch_size)
    t0 = time.perf_counter()
    fg, aux = predict_region(
        model, em, spec, device, collect_aux=collect_aux, progress=progress, cancel=cancel
    )
    timings["predict"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if factors is not None and fg.shape != original_shape:
        inv = (1.0 / factors[0], 1.0 / factors[1])
        fg = zoom_probability(fg, inv)
        fg = _fit_to(fg, original_shape)
        if aux is not None:
            aux = [
                _fit_to(zoom_probability(a, (1.0, *inv)[1:]), original_shape, lead=a.shape[0])
                for a in aux
            ]
    timings["resample_out"] = time.perf_counter() - t0

    contract = {
        "model_id": spec.model_id,
        "arm_name": spec.arm_name,
        "organelle": spec.organelle,
        "family": spec.family,
        "tile_size": t,
        "patch_size": spec.encoder.patch_size,
        "overlap": spec.overlap,
        "stride": stride_for(t, spec.overlap),
        "blend": "hann2d+1e-3",
        "pad_mode": "constant",
        "task": spec.task,
        "device": str(device),
        "dtype": "float32",
        "autocast": False,
        **{k: v for k, v in rs_info.items() if k != "warnings"},
        "resample_warnings": rs_info["warnings"],
        **prep_info,
        "timings": timings,
    }
    return fg, contract


def _fit_to(a: np.ndarray, hw: tuple[int, int], lead: int | None = None) -> np.ndarray:
    """Crop or edge-pad ``a`` to exactly ``hw`` (zoom round-off can be off by a pixel)."""
    h, w = hw
    a = a[..., :h, :w]
    ph, pw = h - a.shape[-2], w - a.shape[-1]
    if ph > 0 or pw > 0:
        pad = [(0, 0)] * (a.ndim - 2) + [(0, max(ph, 0)), (0, max(pw, 0))]
        a = np.pad(a, pad, mode="edge")
    return a
