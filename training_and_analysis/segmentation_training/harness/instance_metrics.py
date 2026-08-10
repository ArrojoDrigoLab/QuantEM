"""Instance metrics for the mitochondria head: PQ / SQ / RQ, AP@[.5:.95], VI.

Predictions are turned into instances by a fixed, deterministic post-proc (identical across every
encoder, for fairness): threshold the foreground probability, drop objects below ``min_size``, then
connected-components label (``scipy.ndimage.label``). GT instances are the corpus instance ids where
available, else connected-components 'pseudo-instances' of the binary GT (flagged ``gt_is_instance``
upstream). numpy + scipy.ndimage only (no sklearn).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi

_AP_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # COCO-style [.5:.05:.95]


def postproc_instances(prob: np.ndarray, valid: np.ndarray, threshold: float = 0.5,
                       min_size: int = 0) -> np.ndarray:
    """Fixed semantic->instance post-proc: threshold -> remove small -> connected components."""
    binary = (np.asarray(prob) >= threshold) & valid
    lab, n = ndi.label(binary)
    if min_size > 0 and n > 0:
        sizes = np.bincount(lab.ravel())
        small = np.where(sizes < min_size)[0]
        small = small[small > 0]
        if small.size:
            lab[np.isin(lab, small)] = 0
            lab, _ = ndi.label(lab > 0)
    return lab


def _overlaps(gt: np.ndarray, pred: np.ndarray):
    """Per-id areas + foreground-foreground overlap pairs (vectorised, no dense matrix)."""
    Kp = int(pred.max()) + 1
    g = gt.ravel().astype(np.int64)
    p = pred.ravel().astype(np.int64)
    area_g = np.bincount(g)
    area_p = np.bincount(p)
    fg = (g > 0) & (p > 0)
    if fg.any():
        upair, cnt = np.unique(g[fg] * Kp + p[fg], return_counts=True)
        gi = (upair // Kp).astype(int)
        pi = (upair % Kp).astype(int)
    else:
        gi = pi = cnt = np.zeros(0, dtype=int)
    return area_g, area_p, gi, pi, cnt


def _ap_all_points(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC all-points AP from a precision-recall curve."""
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def instance_metrics(prob: np.ndarray, gt_inst: np.ndarray, valid: np.ndarray, *,
                     threshold: float = 0.5, min_size: int = 0) -> dict:
    """PQ/SQ/RQ, AP@[.5:.95], VI for one crop via the fixed threshold + connected-components labeller.

    This is the semantic-map-via-connected-components metric (decoder-agnostic, cross-arm-fair): pred
    instances come from thresholding the semantic ``prob`` and labelling connected components, not from
    any decoder's own instance head. For the true-instance metric (a decoder's native mutex-watershed /
    panoptic grouping), use ``instance_metrics_from_labels``. ``gt_inst`` = GT instance ids (0=bg/ignore)."""
    valid = np.asarray(valid)
    gt = np.asarray(gt_inst).astype(np.int64) * valid
    pred = postproc_instances(prob, valid, threshold, min_size)
    return _score_instances(gt, pred, np.asarray(prob), valid)


def instance_metrics_from_labels(pred_labels: np.ndarray, gt_inst: np.ndarray, valid: np.ndarray,
                                 prob: np.ndarray) -> dict:
    """PQ/SQ/RQ, AP, VI for one crop from a decoder's own native instance labels (the true-instance metric).

    ``pred_labels`` = the decoder's native instance-id map (mutex-watershed on affinities, or panoptic
    center/offset grouping). ``prob`` (semantic foreground) is used only for the AP confidence ordering."""
    valid = np.asarray(valid)
    gt = np.asarray(gt_inst).astype(np.int64) * valid
    pred = np.asarray(pred_labels).astype(np.int64) * valid
    return _score_instances(gt, pred, np.asarray(prob), valid)


def _score_instances(gt: np.ndarray, pred: np.ndarray, prob: np.ndarray, valid: np.ndarray) -> dict:
    """Score a predicted instance-label map ``pred`` against GT instances ``gt`` (both [H,W] int).
    ``prob`` [H,W] is the semantic foreground (AP confidence ordering only)."""
    area_g, area_p, gi, pi, inter = _overlaps(gt, pred)
    gt_ids = np.nonzero(area_g)[0]
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = np.nonzero(area_p)[0]
    pred_ids = pred_ids[pred_ids > 0]

    out = {"pq": None, "sq": None, "rq": None, "ap": None, "vi": None,
           "n_gt_inst": int(gt_ids.size), "n_pred_inst": int(pred_ids.size)}
    if gt_ids.size == 0 and pred_ids.size == 0:
        out.update(pq=1.0, sq=1.0, rq=1.0, ap=1.0, vi=0.0)
        return out

    iou = inter / np.maximum(area_g[gi] + area_p[pi] - inter, 1) if inter.size else inter.astype(float)
    iou_by_pred = defaultdict(list)
    for g_, p_, io in zip(gi, pi, iou):
        iou_by_pred[int(p_)].append((float(io), int(g_)))

    # --- PQ (unique IoU>0.5 matching) ---
    matched_g, matched_p, sum_iou = set(), set(), 0.0
    for g_, p_, io in sorted(zip(gi, pi, iou), key=lambda t: -t[2]):
        if io > 0.5 and int(g_) not in matched_g and int(p_) not in matched_p:
            matched_g.add(int(g_))
            matched_p.add(int(p_))
            sum_iou += float(io)
    tp = len(matched_g)
    fp = pred_ids.size - tp
    fn = gt_ids.size - tp
    out["sq"] = sum_iou / tp if tp else 0.0
    out["rq"] = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    out["pq"] = out["sq"] * out["rq"]

    # --- AP@[.5:.95] (per-crop, confidence = mean prob per predicted instance) ---
    prob = np.asarray(prob)
    score = {int(pid): float(prob[pred == pid].mean()) for pid in pred_ids}
    order = sorted(pred_ids.tolist(), key=lambda pid: -score[int(pid)])
    aps = []
    for t in _AP_THRESHOLDS:
        taken, tps = set(), []
        for pid in order:
            cands = sorted([(io, g_) for io, g_ in iou_by_pred[int(pid)] if io >= t and g_ not in taken],
                           reverse=True)
            if cands:
                taken.add(cands[0][1])
                tps.append(1)
            else:
                tps.append(0)
        if not order:
            aps.append(1.0 if gt_ids.size == 0 else 0.0)
            continue
        tps = np.array(tps)
        tp_c = np.cumsum(tps)
        fp_c = np.cumsum(1 - tps)
        rec = tp_c / max(gt_ids.size, 1)
        prec = tp_c / np.maximum(tp_c + fp_c, 1)
        aps.append(_ap_all_points(rec, prec) if gt_ids.size else 0.0)
    out["ap"] = float(np.mean(aps))

    # --- VI (variation of information, bits; over all valid pixels incl. background) ---
    out["vi"] = _variation_of_information(gt, pred, valid)
    return out


def _variation_of_information(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> float:
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    n = g.size
    if n == 0:
        return 0.0
    Kp = int(p.max()) + 1
    upair, cnt = np.unique(g * Kp + p, return_counts=True)
    gi = upair // Kp
    pi = upair % Kp
    ng = np.bincount(g)
    npp = np.bincount(p)
    pxy = cnt / n
    px = ng[gi] / n
    py = npp[pi] / n
    h_x_given_y = -np.sum(pxy * np.log2(pxy / py))
    h_y_given_x = -np.sum(pxy * np.log2(pxy / px))
    return float(h_x_given_y + h_y_given_x)
