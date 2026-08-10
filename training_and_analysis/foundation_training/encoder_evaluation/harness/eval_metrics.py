"""Torch-free per-crop metric computation, isolated so it can run in a *spawn* ProcessPool.

The single biggest CPU cost of a mito eval is the per-crop scipy work (two full-crop city-block
distance transforms for boundary-IoU, boundary-pixel nearest-neighbour queries for boundary-F1/HD95,
plus instance PQ/AP/VI). It is pure numpy/scipy and deterministic,
so distributing one crop per pool task across cores is bit-for-bit identical to the serial path
(there is no RNG here — the bootstrap runs later, sequentially, in ``metrics.aggregate``).

Kept out of ``evaluate.py`` (which imports torch) so the spawn workers import only numpy/scipy and
start fast.
"""

from __future__ import annotations

from ..constants import IGNORE_INDEX
from .instance_metrics import instance_metrics
from .metrics import per_crop_metrics

# Stratification / identity fields copied through onto each per-crop record (for the per-image CSV
# and the macro-over-subgroups aggregation).
STRAT_FIELDS = ("dataset", "subgroup", "modality", "scale_band", "tissue_context",
                "species_group", "organelle", "split", "collection", "crop_id",
                "orientation", "plane_k")

def cfg_fields(cfg) -> dict:
    """The (picklable, torch-free) subset of cfg the metric task needs, so cfg itself is never shipped."""
    return {"fg_threshold": cfg.fg_threshold,
            "boundary_theta_frac": cfg.boundary_theta_frac,
            "boundary_dilation_ratio": cfg.boundary_dilation_ratio,
            "auprc_bins": cfg.auprc_bins,
            "hd95_pct": cfg.hd95_pct,
            "instance_min_size": cfg.instance_min_size}

def crop_metrics(prob, mask, r, cf):
    """All metrics for one crop and one head. ``cf`` is the dict from :func:`cfg_fields`."""
    organelle = r.get("organelle")
    pred = prob >= cf["fg_threshold"]
    m = per_crop_metrics(pred, mask, prob=prob, organelle=organelle,
                         theta_frac=cf["boundary_theta_frac"], dilation_ratio=cf["boundary_dilation_ratio"],
                         auprc_bins=cf["auprc_bins"], hd95_pct=cf["hd95_pct"])
    if organelle == "mito" and not m["excluded"]:
        gt_inst = r.get("_gt_inst")
        if gt_inst is not None:
            m.update(instance_metrics(prob, gt_inst, mask != IGNORE_INDEX,
                                      threshold=cf["fg_threshold"], min_size=cf["instance_min_size"]))
            m["gt_is_instance"] = r.get("gt_is_instance")
    for f in STRAT_FIELDS:
        m[f] = r.get(f)
    m["subgroup"] = r.get("subgroup", "") or "(none)"
    m["sample_id"] = r.get("sample_id")
    return m

def crop_task(payload):
    """Pool task: metrics for all heads of one crop. ``payload`` is fully picklable (arrays + dicts)."""
    probs_by_head, mask, r, cf = payload
    return {name: crop_metrics(prob, mask, r, cf) for name, prob in probs_by_head.items()}
