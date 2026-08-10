"""Segmentation metrics for the probe.

Per-crop semantic / boundary / topology / probability metrics here; instance metrics (PQ / AP / VI)
live in ``instance_metrics.py``. All metrics (1) exclude IGNORE pixels first, (2) use the both-empty
exclusion policy, (3) aggregate per-crop -> per-subgroup mean -> equal-weight macro (+ worst-subgroup,
micro, bootstrap CIs).

Per-head sets:
  * ER:   dice, iou, precision, recall, boundary_f1, boundary_iou, hd95, cldice, auprc
  * mito: ...the above (minus cldice) + pq, sq, rq, ap, vi  (instance metrics merged in by
    ``eval_metrics.crop_metrics``)

Implementation uses numpy, ``scipy.ndimage`` and ``scipy.spatial``. ``clDice`` prefers ``skimage.skeletonize``; where
scikit-image is not installed a scipy morphological skeleton stands in, which gives slightly different
values; scikit-image is therefore required for clDice values that are comparable across runs.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi

from ..constants import FOREGROUND, IGNORE_INDEX

_EPS = 1e-7

SEMANTIC_KEYS = ("dice", "iou", "precision", "recall", "boundary_f1", "boundary_iou",
                 "hd95", "cldice", "auprc")
INSTANCE_KEYS = ("pq", "sq", "rq", "ap", "vi")
ALL_KEYS = SEMANTIC_KEYS + INSTANCE_KEYS
LOWER_BETTER = frozenset({"hd95", "vi"})  # for these, a lower value is better (worst = max)

def _diag(h: int, w: int) -> float:
    return float(np.sqrt(h * h + w * w))

def _boundary(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """1-pixel inner contour of a binary mask, ignore-aware (the mask/ignore frontier is not treated
    as an object edge)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~ndi.binary_erosion(mask | ~valid, border_value=0)

def _band(mask: np.ndarray, valid: np.ndarray, d: int) -> np.ndarray:
    """Outer d-px ring of ``mask`` (ignore-aware). Uses a city-block distance transform — O(n_px) —
    instead of ``binary_erosion(iterations=d)`` — O(d * n_px), which dominates evaluation cost for
    big crops where d ~= 100+. Exactly equivalent: iterated cross-erosion == taxicab distance <= d,
    with a 1-px False pad reproducing erosion's border_value=0."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    m = np.pad(mask | ~valid, 1, constant_values=False)
    dt = ndi.distance_transform_cdt(m, metric="taxicab")[1:-1, 1:-1]
    return mask & (dt <= max(int(d), 1))

def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin centerline of a binary mask (skimage if present; scipy morphological skeleton fallback)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    try:
        from skimage.morphology import skeletonize
        return np.asarray(skeletonize(mask), dtype=bool)
    except Exception:
        # Lantuejoul morphological skeleton: union over k of (erode^k - opening(erode^k)); a subset of
        # the mask, so clDice's perfect-prediction case still yields 1.0.
        skel = np.zeros_like(mask, dtype=bool)
        eroded = mask.copy()
        while eroded.any():
            opened = ndi.binary_dilation(ndi.binary_erosion(eroded))
            skel |= eroded & ~opened
            eroded = ndi.binary_erosion(eroded)
        return skel

def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Centerline Dice (Shit et al. 2021): harmonic mean of topology precision + sensitivity."""
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    if gt.sum() == 0 or pred.sum() == 0:
        return 0.0
    sp, sg = _skeletonize(pred), _skeletonize(gt)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    tprec = (sp & gt).sum() / sp.sum()  # pred skeleton inside gt
    tsens = (sg & pred).sum() / sg.sum()  # gt skeleton inside pred
    return 0.0 if (tprec + tsens) == 0 else float(2 * tprec * tsens / (tprec + tsens))

def _auprc_binned(scores: np.ndarray, pos: np.ndarray, bins: int = 256) -> float:
    """Binned area under the precision-recall curve (average precision). ``pos`` = boolean labels."""
    npos = int(pos.sum())
    if npos == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    hp, _ = np.histogram(scores[pos], edges)
    hn, _ = np.histogram(scores[~pos], edges)
    hp, hn = hp[::-1], hn[::-1]  # high score first
    tp, fp = np.cumsum(hp), np.cumsum(hn)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / npos
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))

def per_crop_metrics(pred_bin: np.ndarray, gt_label: np.ndarray, *, prob: np.ndarray | None = None,
                     organelle: str | None = None, theta_frac: float = 0.0075,
                     dilation_ratio: float = 0.02, auprc_bins: int = 256,
                     hd95_pct: float = 95.0) -> dict:
    """Per-crop semantic + boundary + topology + probability metrics. ``gt_label`` in {0,1,255}.

    ``prob`` (foreground probability, same shape) enables AUPRC; ``organelle=='er'`` enables clDice.
    """
    pred_bin = np.asarray(pred_bin).astype(bool)
    gt_label = np.asarray(gt_label)
    valid = gt_label != IGNORE_INDEX
    gt = (gt_label == FOREGROUND) & valid
    pred = pred_bin & valid
    gt_fg, pred_fg, valid_px = int(gt.sum()), int(pred.sum()), int(valid.sum())

    out = {k: None for k in SEMANTIC_KEYS}
    out.update(excluded=False, gt_fg=gt_fg, pred_fg=pred_fg, valid_px=valid_px)
    if gt_fg == 0 and pred_fg == 0:  # both-empty -> excluded from means (but counted)
        out["excluded"] = True
        return out

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    out["dice"] = (2 * tp + _EPS) / (2 * tp + fp + fn + _EPS)
    out["iou"] = (tp + _EPS) / (tp + fp + fn + _EPS)
    out["precision"] = (tp + _EPS) / (tp + fp + _EPS)
    out["recall"] = (tp + _EPS) / (tp + fn + _EPS)

    h, w = gt.shape
    diag = _diag(h, w)
    theta = theta_frac * diag
    d = max(int(round(dilation_ratio * diag)), 1)

    gb = _boundary(gt, valid) & valid
    pb = _boundary(pred, valid) & valid
    # Boundary-F1
    if gb.sum() == 0 and pb.sum() == 0:
        out["boundary_f1"] = 1.0
    elif gb.sum() == 0 or pb.sum() == 0:
        out["boundary_f1"] = 0.0
    else:
        # Distances from each pred-boundary px to the nearest gt-boundary px (and vice-versa) via a
        # nearest-neighbour query over just the boundary pixels (cKDTree). This is exactly what
        # distance_transform_edt(~gb)[pb] gives (Euclidean dist to nearest boundary px), but O(#boundary
        # px) instead of a full-crop O(16.7M px) EDT, which otherwise dominates evaluation cost.
        from scipy.spatial import cKDTree
        d_pb = cKDTree(np.argwhere(gb)).query(np.argwhere(pb), workers=1)[0]
        d_gb = cKDTree(np.argwhere(pb)).query(np.argwhere(gb), workers=1)[0]
        bp_prec = float((d_pb <= theta).mean())
        bp_rec = float((d_gb <= theta).mean())
        out["boundary_f1"] = 0.0 if (bp_prec + bp_rec) == 0 else 2 * bp_prec * bp_rec / (bp_prec + bp_rec)
        # HD95 (symmetric 95th-percentile boundary distance, px)
        out["hd95"] = float(max(np.percentile(d_pb, hd95_pct), np.percentile(d_gb, hd95_pct)))
    if out["hd95"] is None:
        # exactly one side has a boundary -> worst-case finite penalty (the image diagonal)
        out["hd95"] = 0.0 if (gb.sum() == 0 and pb.sum() == 0) else float(diag)

    # Boundary-IoU
    bg = _band(gt, valid, d) & valid
    bpd = _band(pred, valid, d) & valid
    union = int((bg | bpd).sum())
    out["boundary_iou"] = 1.0 if union == 0 else int((bg & bpd).sum()) / union

    if organelle == "er":
        out["cldice"] = cldice(pred, gt)
    if prob is not None and gt_fg > 0:
        out["auprc"] = _auprc_binned(np.asarray(prob)[valid].astype(np.float32), gt[valid], auprc_bins)

    out["theta_px"], out["dilation_px"] = theta, d
    return out

# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _present_keys(records) -> list[str]:
    have = set()
    for r in records:
        for k in ALL_KEYS:
            if r.get(k) is not None:
                have.add(k)
    return [k for k in ALL_KEYS if k in have]

def _subgroup_means(kept, keys):
    by_sub = defaultdict(list)
    for r in kept:
        by_sub[r.get("subgroup", "") or "(none)"].append(r)
    per = {}
    for sub, rs in sorted(by_sub.items()):
        row = {"n": len(rs)}
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            row[k] = float(np.mean(vals)) if vals else None
        per[sub] = row
    return per

def _macro(per_subgroup, keys):
    out = {}
    for k in keys:
        sm = [v[k] for v in per_subgroup.values() if v.get(k) is not None]
        out[k] = float(np.mean(sm)) if sm else None
    return out

def aggregate(records: list[dict], *, bootstrap_n: int = 0, seed: int = 0, ci: float = 95.0) -> dict:
    """Aggregate per-crop metric dicts: macro (balanced) + micro + per-subgroup + worst-subgroup,
    over whatever metric keys are present, with optional bootstrap CIs on the macro means."""
    kept = [r for r in records if not r.get("excluded")]
    n_excluded = sum(1 for r in records if r.get("excluded"))
    keys = _present_keys(kept)

    per_subgroup = _subgroup_means(kept, keys)
    macro = _macro(per_subgroup, keys)
    micro = {}
    for k in keys:
        vals = [r[k] for r in kept if r.get(k) is not None]
        micro[k] = float(np.mean(vals)) if vals else None

    worst = {}
    for k in keys:
        sm = [(v[k], sub) for sub, v in per_subgroup.items() if v.get(k) is not None]
        if sm:
            pick = max(sm) if k in LOWER_BETTER else min(sm)
            worst[k] = {"value": pick[0], "subgroup": pick[1]}

    summary = {
        "macro": macro, "micro": micro, "worst_subgroup": worst, "per_subgroup": per_subgroup,
        "n_crops": len(records), "n_evaluated": len(kept), "n_excluded_both_empty": n_excluded,
    }
    if bootstrap_n and kept:
        summary["macro_ci"] = _bootstrap_macro_ci(kept, keys, bootstrap_n, seed, ci)
    return summary

def _bootstrap_macro_ci(kept, keys, n_boot, seed, ci):
    """Resample crops *within each subgroup* (preserving the balanced design) -> percentile CI on the
    macro mean per metric."""
    rng = np.random.default_rng(seed)
    by_sub = defaultdict(list)
    for r in kept:
        by_sub[r.get("subgroup", "") or "(none)"].append(r)
    subs = sorted(by_sub)
    arrays = {sub: {k: np.array([r[k] for r in by_sub[sub] if r.get(k) is not None], float)
                    for k in keys} for sub in subs}
    lo_p, hi_p = (100 - ci) / 2, 100 - (100 - ci) / 2
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        for k in keys:
            sub_means = []
            for sub in subs:
                a = arrays[sub][k]
                if a.size:
                    idx = rng.integers(0, a.size, a.size)
                    sub_means.append(a[idx].mean())
            if sub_means:
                boot[k].append(float(np.mean(sub_means)))
    out = {}
    for k in keys:
        if boot[k]:
            out[k] = {"lo": float(np.percentile(boot[k], lo_p)),
                      "hi": float(np.percentile(boot[k], hi_p))}
    return out
