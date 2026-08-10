"""Region-masked segmentation metrics: semantic (Dice etc.) + instance (PQ/AP)."""
import numpy as np


def semantic_metrics(pred_bin, gt_bin, eval_mask):
    """All quantities computed only over eval_mask == True (ignore elsewhere)."""
    m = eval_mask.astype(bool)
    p = (np.asarray(pred_bin) > 0) & m
    g = (np.asarray(gt_bin) > 0) & m
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(np.count_nonzero(~p & ~g))
    inter = tp
    union = tp + fp + fn
    psum = tp + fp
    gsum = tp + fn
    dice = (2 * inter) / (psum + gsum) if (psum + gsum) > 0 else np.nan
    iou = inter / union if union > 0 else np.nan
    prec = tp / psum if psum > 0 else np.nan
    rec = tp / gsum if gsum > 0 else np.nan
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and prec + rec > 0) else (
        0.0 if (psum + gsum) > 0 else np.nan)
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    # empty-GT / empty-pred convention: if both empty within region -> perfect (1.0)
    if gsum == 0 and psum == 0:
        dice = iou = 1.0
        prec = rec = f1 = 1.0
    return dict(dice=_r(dice), iou=_r(iou), precision=_r(prec), recall=_r(rec),
                f1=_r(f1), accuracy=_r(acc), specificity=_r(spec),
                tp=tp, fp=fp, fn=fn, tn=tn,
                gt_area=gsum, pred_area=psum, eval_area=int(np.count_nonzero(m)),
                gt_frac=_r(gsum / max(1, np.count_nonzero(m))))


def _prep_topology(p, g, cap=1024):
    """Crop to the bbox of p|g and stride-downscale to <=cap px/side so the expensive
    skeletonize/distance-transform ops stay tractable on huge eval regions."""
    m = p | g
    if not m.any():
        return p, g
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    p = p[y0:y1, x0:x1]; g = g[y0:y1, x0:x1]
    step = max(1, int(np.ceil(max(p.shape) / cap)))
    if step > 1:
        p = p[::step, ::step]; g = g[::step, ::step]
    return p, g


def cldice(pred_bin, gt_bin, eval_mask):
    """centerline Dice (topology-aware). Requires skimage; returns nan on failure."""
    try:
        from skimage.morphology import skeletonize
    except Exception:
        return np.nan
    m = eval_mask.astype(bool)
    p = ((np.asarray(pred_bin) > 0) & m)
    g = ((np.asarray(gt_bin) > 0) & m)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    p, g = _prep_topology(p, g)
    sp = skeletonize(p); sg = skeletonize(g)
    tprec = (sp & g).sum() / sp.sum() if sp.sum() > 0 else 0.0
    tsens = (sg & p).sum() / sg.sum() if sg.sum() > 0 else 0.0
    return _r(2 * tprec * tsens / (tprec + tsens) if (tprec + tsens) > 0 else 0.0)


def boundary_f1(pred_bin, gt_bin, eval_mask, tol=2):
    """Boundary F-score at pixel tolerance `tol` (BF metric)."""
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
    except Exception:
        return np.nan
    m = eval_mask.astype(bool)
    p = ((np.asarray(pred_bin) > 0) & m)
    g = ((np.asarray(gt_bin) > 0) & m)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    p, g = _prep_topology(p, g)
    def bound(x):
        return x ^ binary_erosion(x)
    pb = bound(p); gb = bound(g)
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    dg = distance_transform_edt(~gb)
    dp = distance_transform_edt(~pb)
    prec = (dg[pb] <= tol).mean()
    rec = (dp[gb] <= tol).mean()
    return _r(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)


def _match_instances(pred_inst, gt_inst, eval_mask):
    """IoU matrix between predicted and gt instances within eval_mask."""
    m = eval_mask.astype(bool)
    pred = np.where(m, pred_inst, 0)
    gt = np.where(m, gt_inst, 0)
    pids = [i for i in np.unique(pred) if i != 0]
    gids = [i for i in np.unique(gt) if i != 0]
    if not pids or not gids:
        return pids, gids, np.zeros((len(pids), len(gids)))
    pidx = {v: k for k, v in enumerate(pids)}
    gidx = {v: k for k, v in enumerate(gids)}
    parea = {v: 0 for v in pids}
    garea = {v: 0 for v in gids}
    inter = {}
    pf = pred.ravel(); gf = gt.ravel()
    # areas
    for v, c in zip(*np.unique(pred[pred > 0], return_counts=True)):
        parea[int(v)] = int(c)
    for v, c in zip(*np.unique(gt[gt > 0], return_counts=True)):
        garea[int(v)] = int(c)
    both = (pred > 0) & (gt > 0)
    keys = pred[both].astype(np.int64) * (max(gids) + 1) + gt[both].astype(np.int64)
    for k, c in zip(*np.unique(keys, return_counts=True)):
        pv = int(k // (max(gids) + 1)); gv = int(k % (max(gids) + 1))
        inter[(pv, gv)] = int(c)
    iou = np.zeros((len(pids), len(gids)))
    for (pv, gv), c in inter.items():
        u = parea[pv] + garea[gv] - c
        if u > 0:
            iou[pidx[pv], gidx[gv]] = c / u
    return pids, gids, iou


def instance_metrics(pred_inst, gt_inst, eval_mask, iou_thresholds=(0.5, 0.75)):
    """Panoptic Quality (@0.5) + detection precision/recall/F1/AP at IoU thresholds."""
    pids, gids, iou = _match_instances(pred_inst, gt_inst, eval_mask)
    out = dict(n_pred=len(pids), n_gt=len(gids))
    if len(pids) == 0 and len(gids) == 0:
        out.update(pq=1.0, sq=1.0, rq=1.0)
        for t in iou_thresholds:
            out[f"precision@{t}"] = 1.0; out[f"recall@{t}"] = 1.0
            out[f"f1@{t}"] = 1.0; out[f"ap@{t}"] = 1.0
        return out
    # greedy one-to-one matching by descending IoU
    def match_at(thr):
        pairs = []
        if iou.size:
            order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
            usedp, usedg = set(), set()
            for pi, gi in order:
                if iou[pi, gi] < thr:
                    break
                if pi in usedp or gi in usedg:
                    continue
                usedp.add(pi); usedg.add(gi); pairs.append((pi, gi, iou[pi, gi]))
        tp = len(pairs); fp = len(pids) - tp; fn = len(gids) - tp
        return tp, fp, fn, pairs
    for t in iou_thresholds:
        tp, fp, fn, pairs = match_at(t)
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec > 0) else 0.0
        ap = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan   # AP-style (no scores)
        out[f"precision@{t}"] = _r(prec); out[f"recall@{t}"] = _r(rec)
        out[f"f1@{t}"] = _r(f1); out[f"ap@{t}"] = _r(ap)
    # Panoptic Quality at 0.5
    tp, fp, fn, pairs = match_at(0.5)
    sq = np.mean([p[2] for p in pairs]) if pairs else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    out.update(pq=_r(sq * rq), sq=_r(sq), rq=_r(rq))
    return out


def _r(x):
    if x is None:
        return None
    try:
        if isinstance(x, float) and (np.isnan(x)):
            return float("nan")
        return round(float(x), 6)
    except Exception:
        return x


ALL_SEMANTIC_KEYS = ["dice", "iou", "precision", "recall", "f1", "accuracy",
                     "specificity", "cldice", "boundary_f1",
                     "tp", "fp", "fn", "tn", "gt_area", "pred_area", "eval_area", "gt_frac"]
