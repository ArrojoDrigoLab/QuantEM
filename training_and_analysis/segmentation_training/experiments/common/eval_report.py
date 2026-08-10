"""Standardised reporting for every experiment arm.

Every arm reports:
  * both metrics side by side — semantic foreground, and true-instance (``inst_*`` from
    ``harness.instance_eval``); connected components on a semantic map is reported as its own metric
    family, not as the instance score;
  * the three-way split — ``train``, held-out-image ``test_image``, and held-out-source ``test`` —
    plus the ``test_image`` to ``test`` gap, the appearance-shift term that dominates architecture;
  * per-subgroup worst case, with the individual source rows kept explicit.

This module does not run models. Callers hand it ``{split: {"summary": ..., "per_crop": [...]}}``,
from ``harness.evaluate.evaluate_head`` or from an arm's own evaluation loop, and it assembles the
record. Standard library only, plus the harness metric keys.
"""

from __future__ import annotations

import json
from pathlib import Path

# Metric groups (kept in sync with harness.metrics.{SEMANTIC_KEYS,INSTANCE_KEYS}).
SEMANTIC_KEYS = ("dice", "iou", "precision", "recall", "boundary_f1", "boundary_iou", "hd95", "cldice", "auprc")
SEMANTIC_CC_INSTANCE = ("pq", "sq", "rq", "ap", "vi")                       # threshold+CC on the softmax
TRUE_INSTANCE_KEYS = ("inst_pq", "inst_sq", "inst_rq", "inst_ap", "inst_vi")  # the decoder's own post-processing
LOWER_BETTER = {"hd95", "vi", "inst_vi"}

# Named-source rows reported as explicit subgroups. Matched case-insensitively against a crop's
# ``dataset`` (preferred) or ``subgroup`` string, as spelled in the manifest's source names.
NAMED_SOURCES = {
    "mito": {"FAST-EM": ("fastem",), "nPOD": ("npod", "islet")},
    "er": {"nPOD": ("npod", "islet"), "FAST-EM": ("fastem",)},
}


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def named_source_metrics(per_crop: list[dict], matchers: dict[str, tuple], metrics) -> dict:
    """Per-named-source metric means computed directly from per-crop records (by ``dataset`` substring).

    Returns ``{label: {"n": k, metric: mean, ...}}``; a label with no matching crops is omitted.
    """
    out: dict = {}
    for label, subs in matchers.items():
        subs = tuple(s.lower() for s in subs)
        rows = [c for c in per_crop
                if any(s in str(c.get("dataset", "")).lower() or s in str(c.get("subgroup", "")).lower()
                       for s in subs)]
        if not rows:
            continue
        out[label] = {"n": len(rows)}
        for m in metrics:
            out[label][m] = _mean([c.get(m) for c in rows])
    return out


def _present_metrics(summary: dict) -> list[str]:
    macro = summary.get("macro", {}) if isinstance(summary, dict) else {}
    keys = list(SEMANTIC_KEYS) + list(SEMANTIC_CC_INSTANCE) + list(TRUE_INSTANCE_KEYS)
    return [k for k in keys if k in macro and macro.get(k) is not None]


def split_gap(a: dict, b: dict, metrics) -> dict:
    """``a - b`` per metric on the macro means (e.g. test_image macro minus test macro = the
    held-out-image→held-out-source generalization gap). Sign-normalized so + always = "b worse than a"."""
    ma, mb = a.get("macro", {}), b.get("macro", {})
    gap = {}
    for m in metrics:
        va, vb = ma.get(m), mb.get(m)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            gap[m] = (vb - va) if m in LOWER_BETTER else (va - vb)
    return gap


def assemble_report(arm: str, organelle: str, split_results: dict, *,
                    extra: dict | None = None) -> dict:
    """Build the standardized report record.

    Args:
        arm:            arm name (e.g. ``scale_er_2nm_omniem``).
        organelle:      ``"mito"`` | ``"er"`` (selects the FAST-EM/nPOD matchers).
        split_results:  ``{split: {"summary": <aggregate dict>, "per_crop": [<crop dicts>]}}`` for any of
                        ``train`` / ``test_image`` / ``test`` / ``val`` / ``loso``.
        extra:          arm-specific fields (e.g. TTA compute cost, scale, k, seed provenance).

    Returns a JSON-able record with per-split macro/worst-subgroup/CI, the test_image→test gap, named-source
    rows, and both metric families.
    """
    metrics_all = list(SEMANTIC_KEYS) + list(SEMANTIC_CC_INSTANCE) + list(TRUE_INSTANCE_KEYS)
    matchers = NAMED_SOURCES.get(organelle, {})
    splits: dict = {}
    for sp, res in split_results.items():
        summ = res.get("summary", {}) if isinstance(res, dict) else {}
        per_crop = res.get("per_crop", []) if isinstance(res, dict) else []
        present = _present_metrics(summ)
        splits[sp] = {
            "n_evaluated": summ.get("n_evaluated"),
            "macro": {m: summ.get("macro", {}).get(m) for m in present},
            "macro_ci": summ.get("macro_ci"),
            "worst_subgroup": summ.get("worst_subgroup"),
            "named_sources": named_source_metrics(per_crop, matchers, present),
        }
    report = {
        "arm": arm,
        "organelle": organelle,
        "metrics_reported": {"semantic": list(SEMANTIC_KEYS),
                             "instance_true": list(TRUE_INSTANCE_KEYS),
                             "instance_cc": list(SEMANTIC_CC_INSTANCE)},
        "splits": splits,
        "extra": extra or {},
    }
    # The headline generalization gap: rebalanced held-out-image → held-out-source.
    if "test_image" in split_results and "test" in split_results:
        ti = split_results["test_image"].get("summary", {})
        ts = split_results["test"].get("summary", {})
        report["test_image_to_test_gap"] = split_gap(ti, ts, _present_metrics(ti))
    return report


def headline_rows(report: dict, metrics=("dice", "cldice", "inst_pq", "pq")) -> list[dict]:
    """Flatten to one row per split for a summary table (macro values + the named-source worst)."""
    rows = []
    for sp, s in report.get("splits", {}).items():
        macro = s.get("macro", {})
        row = {"arm": report["arm"], "organelle": report["organelle"], "split": sp,
               "n": s.get("n_evaluated")}
        for m in metrics:
            row[m] = macro.get(m)
        ws = s.get("worst_subgroup", {}) or {}
        row["worst_dice"] = (ws.get("dice", {}) or {}).get("value") if isinstance(ws.get("dice"), dict) else None
        rows.append(row)
    return rows


def write_report(report: dict, out_dir: str | Path, arm: str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"report_{arm or report.get('arm', 'arm')}.json"
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return p
