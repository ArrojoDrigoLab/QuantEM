"""Multi-scale test-time fusion — a scale ensemble applied at evaluation.

A model trained at one scale is applied to each test region resampled to several scales, and the
per-scale foreground probability maps are fused by mean, max or a weighted mean. No retraining.

The resampling is region-level rather than tile-level, so the physical field seen at each scale really
does differ: resample the whole region, evaluate it with the ordinary sliding window, resize the
probability map back, then fuse. ``harness.evaluate.predict_region`` is reused unchanged, so tiling and
blending are identical to every other arm; the outer resample and fuse are specific to this arm.

The fused map drives the semantic and semantic-connected-component scores. The true-instance
``inst_*`` score instead uses the decoder's own post-processing on the native-scale auxiliary output,
because fusing affinities or centre offsets across scales is not well defined; instance quality is
therefore reported at native scale alongside the fused semantic map.
"""

from __future__ import annotations

import numpy as np

from ...harness.evaluate import predict_region


def _zoom(arr: np.ndarray, factor, order: int) -> np.ndarray:
    """Resample ``arr`` by ``factor`` (a scalar, or a per-axis tuple). No-op if all factors ≈ 1."""
    facs = factor if isinstance(factor, (tuple, list)) else (factor, factor)
    if all(abs(float(f) - 1.0) < 1e-3 for f in facs):
        return arr
    from scipy.ndimage import zoom
    return zoom(arr, factor, order=order, mode="nearest")


def predict_region_multiscale(model, em_uint8: np.ndarray, cfg, mean: float, std: float, device, *,
                              scales=(0.75, 1.0, 1.5), fuse: str = "mean", weights=None,
                              collect_aux: bool = False):
    """Fused foreground probability over a region across ``scales`` (resample→predict→resize back→fuse).

    ``fuse``: ``mean`` (equal-weight prob average) | ``max`` (max prob; recall-favoring) | ``wmean``
    (weighted mean, ``weights`` per scale — e.g. a learned scale weighting). Returns ``fg[H,W]`` (and, if
    ``collect_aux``, the s=1.0 aux for the true-instance metric)."""
    H, W = em_uint8.shape
    probs, aux_native = [], None
    for s in scales:
        em_s = _zoom(em_uint8, s, order=1).clip(0, 255).astype(np.uint8) if abs(s - 1.0) >= 1e-3 else em_uint8
        if collect_aux and abs(s - 1.0) < 1e-3:
            prob_s, aux_native = predict_region(model, em_s, cfg, mean, std, device, collect_aux=True)
        else:
            prob_s = predict_region(model, em_s, cfg, mean, std, device)
        if prob_s.shape != (H, W):
            prob_s = _zoom(prob_s, (H / prob_s.shape[0], W / prob_s.shape[1]), order=1)
            prob_s = prob_s[:H, :W]
            if prob_s.shape != (H, W):  # zoom rounding guard
                pad = np.zeros((H, W), np.float32)
                pad[:prob_s.shape[0], :prob_s.shape[1]] = prob_s
                prob_s = pad
        probs.append(prob_s.astype(np.float32))
    stack = np.stack(probs, 0)                                   # [S, H, W]
    if fuse == "max":
        fg = stack.max(0)
    elif fuse == "wmean" and weights is not None:
        w = np.asarray(weights, np.float32)
        w = w / max(w.sum(), 1e-8)
        fg = (stack * w[:, None, None]).sum(0)
    else:
        fg = stack.mean(0)
    fg = fg.astype(np.float32)
    if collect_aux:
        # aux_native may be None if s=1.0 not in scales — fall back to a plain native pass for the instance head.
        if aux_native is None:
            _p, aux_native = predict_region(model, em_uint8, cfg, mean, std, device, collect_aux=True)
        return fg, aux_native
    return fg


def evaluate_multiscale(model, records, cfg, data_root, device, mean: float, std: float, *,
                        scales=(0.75, 1.0, 1.5), fuse: str = "mean", weights=None) -> dict:
    """Sliding-window multi-scale-fused eval over every record, plus aggregation (same scoring as
    ``harness.evaluate.evaluate_head``, via the shared predictor loop). Reports both metrics."""
    from ..common.region_eval import evaluate_with_predictor

    def predict_fn(m, em, c, mn, sd, dev, collect_aux=False):
        return predict_region_multiscale(m, em, c, mn, sd, dev, scales=scales, fuse=fuse,
                                         weights=weights, collect_aux=collect_aux)

    return evaluate_with_predictor(model, records, cfg, data_root, device, mean, std, predict_fn,
                                   extra_summary={"multiscale": {"scales": list(scales), "fuse": fuse}})
