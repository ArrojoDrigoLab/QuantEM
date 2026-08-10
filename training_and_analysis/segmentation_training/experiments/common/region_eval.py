"""Sliding-window evaluation loop parameterised by a region predictor.

The shared core behind the multi-scale-fusion and two-scale evaluations, and any other arm whose
per-region prediction differs from the stock ``predict_region`` but whose scoring is identical.

Mirrors ``harness.evaluate.evaluate_head`` exactly — same metrics, same dual-instance path,
same stratification and aggregation. Only ``predict_fn(model, em, cfg, mean, std, device,
collect_aux) -> prob | (prob, aux)`` is swapped. Both metrics are reported.
"""

from __future__ import annotations

from ...constants import IGNORE_INDEX
from ...harness.evaluate import _STRAT_FIELDS, _cap_region
from ...harness.metrics import aggregate, per_crop_metrics


def evaluate_with_predictor(model, records, cfg, data_root, device, mean, std, predict_fn, *,
                            extra_summary: dict | None = None) -> dict:
    """Score ``records`` using ``predict_fn`` for per-region foreground prediction. Same contract as
    ``evaluate_head`` (semantic + true-instance ``inst_*``)."""
    from ...harness.dataset import load_sample
    from ...harness.instance_eval import dual_instance_metrics, has_native_instance

    model = model.to(device).eval()
    ev, organelle = cfg.eval, cfg.data.organelle
    is_instance = getattr(cfg.data, "task", "semantic") == "instance"
    has_native = is_instance and has_native_instance(model)
    max_region_px = int(getattr(ev, "max_region_px", 0) or 0)
    per_crop = []
    for r in records:
        em, mask, inst = load_sample(r, data_root)
        em, mask, inst = _cap_region(em, mask, inst, max_region_px, int(cfg.encoder.tile_size))
        aux = None
        if has_native:
            prob, aux = predict_fn(model, em, cfg, mean, std, device, collect_aux=True)
        else:
            prob = predict_fn(model, em, cfg, mean, std, device, collect_aux=False)
        pred_bin = prob >= float(ev.fg_threshold)
        m = per_crop_metrics(pred_bin, mask, prob=prob, organelle=organelle,
                             theta_frac=ev.boundary_theta_frac, dilation_ratio=ev.boundary_dilation_ratio,
                             auprc_bins=ev.auprc_bins, hd95_pct=ev.hd95_pct)
        if is_instance and not m["excluded"]:
            valid = mask != IGNORE_INDEX
            gt_inst = inst
            if gt_inst is None:
                from scipy import ndimage as ndi
                gt_inst, _ = ndi.label((mask == 1) & valid)
            m.update(dual_instance_metrics(getattr(model, "decoder", None), aux, prob, gt_inst, valid,
                                           fg_threshold=float(ev.fg_threshold),
                                           min_size=int(ev.instance_min_size)))
        for f in _STRAT_FIELDS:
            m[f] = r.get(f)
        m["subgroup"] = r.get("subgroup", "") or "(none)"
        m["sample_id"] = r.get("sample_id")
        per_crop.append(m)
    summary = aggregate(per_crop, bootstrap_n=int(ev.bootstrap_n), ci=float(ev.bootstrap_ci),
                        seed=int(cfg.optim.seed))
    if extra_summary:
        summary.update(extra_summary)
    return {"summary": summary, "per_crop": per_crop}
