"""Support-arm evaluation — precision recovery, per-source rows for FAST-EM and nPOD, a leave-one-source-out jackknife band.

The support family targets the high-recall/low-precision regime, so the success metric is precision
recovery at comparable recall, not Dice alone. Every support arm is reported against both the adapted
unconditioned baseline and the replace-combination reference, so a better combination mode has to beat
both. The ground-truth-seeded arm is a ceiling that is not achievable in deployment, and is always
shown alongside the inferred-to-ground-truth gap.

All support/conditioning deltas sit in a ~+-0.075-0.095 seed band, so a single-seed point delta is
directional rather than a significance claim. This module reports the
leave-one-source-out jackknife SD over test sources (a robustness band that needs no retraining)
alongside the point delta; the trained finalists additionally report a per-source leave-one-out delta.

Reads per-crop from a ``results_per_crop.json`` (train arm; keyed by split), from any payload holding a
``per_crop`` list, or from a TTA ``*_per_crop.json`` (a flat list). The module depends only on numpy,
so it runs on a CPU-only machine.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Subgroups always surfaced as their own rows, so per-source performance is reported explicitly.
WATCHLIST = ("FAST-EM", "nPOD", "islet ER", "human islet")


def load_per_crop(path, split: str = "test") -> list:
    """Load a per-crop list from a train-arm results_per_crop.json (keyed by split), from a payload with a
    ``per_crop`` key, or from a flat TTA list."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict) and split in d:
        return d[split]
    if isinstance(d, dict) and "per_crop" in d:
        return d["per_crop"]
    return d if isinstance(d, list) else []


def _balanced_macro(per_crop: list, key: str):
    """Balanced macro (per-subgroup mean -> equal-weight) of ``key`` over non-excluded, annotation-masked crops."""
    by_sub = defaultdict(list)
    for r in per_crop:
        if r.get("excluded"):
            continue
        v = r.get(key)
        if v is not None:
            by_sub[r.get("subgroup") or "(none)"].append(float(v))
    sub_means = [np.mean(v) for v in by_sub.values() if v]
    return float(np.mean(sub_means)) if sub_means else None


def precision_recovery(arm_pc: list, base_pc: list, ref_pc: list | None = None, organelle: str = "er") -> dict:
    """Precision recovery vs the unconditioned baseline (and vs the replace-combination reference): the
    mechanism recovers precision only if it raises precision without materially dropping recall. Reports
    dice/prec/recall for each + the deltas."""
    hl = "pq" if organelle == "mito" else "dice"

    def row(pc):
        return {hl: _balanced_macro(pc, hl), "dice": _balanced_macro(pc, "dice"),
                "precision": _balanced_macro(pc, "precision"), "recall": _balanced_macro(pc, "recall")}

    arm, base = row(arm_pc), row(base_pc)
    out = {"arm": arm, "baseline": base,
           "d_precision_vs_baseline": _sub(arm["precision"], base["precision"]),
           "d_recall_vs_baseline": _sub(arm["recall"], base["recall"]),
           "d_headline_vs_baseline": _sub(arm[hl], base[hl])}
    # Precision counts as recovered when precision rises while recall roughly holds (|d_recall| < 0.05).
    out["precision_recovered"] = bool(out["d_precision_vs_baseline"] is not None and out["d_precision_vs_baseline"] > 0
                                      and abs(out["d_recall_vs_baseline"] or 0) < 0.05)
    if ref_pc is not None:
        ref = row(ref_pc)
        out["c_replace"] = ref
        out["d_headline_vs_creplace"] = _sub(arm[hl], ref[hl])
    return out


def _sub(a, b):
    return (a - b) if (a is not None and b is not None) else None


def watchlist_rows(per_crop: list, patterns=WATCHLIST) -> dict:
    """Precision/recall/Dice for the named subgroups as their own rows."""
    by_sub = defaultdict(list)
    for r in per_crop:
        if not r.get("excluded"):
            by_sub[r.get("subgroup") or "(none)"].append(r)
    rows = {}
    for sub, rs in by_sub.items():
        if any(p.lower() in sub.lower() for p in patterns):
            rows[sub] = {"n": len(rs), "precision": _balanced_macro(rs, "precision"),
                         "recall": _balanced_macro(rs, "recall"), "dice": _balanced_macro(rs, "dice")}
    return rows


def loso_source_delta(arm_pc: list, base_pc: list, metric: str = "dice") -> dict:
    """Leave-one-source-out jackknife on the macro delta (arm - the unconditioned baseline): point delta +
    jackknife SD across test sources. A delta smaller than its SD is directional only, not a claim."""
    def by_source(pc):
        g = defaultdict(list)
        for r in pc:
            if not r.get("excluded") and r.get(metric) is not None:
                g[str(r.get("dataset"))].append(float(r[metric]))
        return g

    ga, gb = by_source(arm_pc), by_source(base_pc)
    sources = sorted(set(ga) & set(gb))
    if len(sources) < 2:
        return {"point_delta": None, "jackknife_sd": None, "n_sources": len(sources)}

    def macro_delta(keep):
        da = np.mean([np.mean(ga[s]) for s in keep if ga[s]])
        db = np.mean([np.mean(gb[s]) for s in keep if gb[s]])
        return float(da - db)

    full = macro_delta(sources)
    loo = [macro_delta([s for s in sources if s != drop]) for drop in sources]
    n = len(sources)
    jack_sd = float(np.sqrt((n - 1) / n * np.sum((np.array(loo) - np.mean(loo)) ** 2)))
    return {"point_delta": full, "jackknife_sd": jack_sd, "n_sources": n,
            "directional_only": bool(abs(full) < jack_sd)}


def report(arm_pc: list, base_pc: list, ref_pc: list | None = None, organelle: str = "er") -> dict:
    hl = "pq" if organelle == "mito" else "dice"
    return {"organelle": organelle,
            "precision_recovery": precision_recovery(arm_pc, base_pc, ref_pc, organelle),
            "loso_delta": loso_source_delta(arm_pc, base_pc, hl),
            "watchlist_arm": watchlist_rows(arm_pc), "watchlist_a0": watchlist_rows(base_pc)}


def to_markdown(rep: dict, arm_name: str = "arm") -> str:
    pr, lo = rep["precision_recovery"], rep["loso_delta"]
    hl = "pq" if rep["organelle"] == "mito" else "dice"
    L = [f"# Support-arm report - {arm_name} ({rep['organelle']})", "",
         "Success = precision recovery at comparable recall (not Dice alone). Delta is directional only if "
         "|delta| < the leave-one-source-out jackknife SD.", "",
         f"- **{hl} vs the unconditioned baseline**: {_f(pr['d_headline_vs_baseline'])}  (arm {_f(pr['arm'][hl])} vs the unconditioned baseline {_f(pr['baseline'][hl])})",
         f"- **precision vs the unconditioned baseline**: {_f(pr['d_precision_vs_baseline'])}   **recall vs the unconditioned baseline**: {_f(pr['d_recall_vs_baseline'])}"
         f"   -> precision recovered: {pr['precision_recovered']}"]
    if "d_headline_vs_creplace" in pr:
        L.append(f"- **{hl} vs the replace-combination reference**: {_f(pr['d_headline_vs_creplace'])}")
    L.append(f"- **LOSO jackknife**: delta {_f(lo['point_delta'])} +/- {_f(lo['jackknife_sd'])} over "
             f"{lo['n_sources']} sources -> {'directional only' if lo.get('directional_only') else 'holds'}")
    L += ["", "## Per-source subgroup rows", "",
          "| subgroup | n | precision (arm/the unconditioned baseline) | recall (arm/the unconditioned baseline) | dice (arm/the unconditioned baseline) |", "|---|---|---|---|---|"]
    for sub in sorted(set(rep["watchlist_arm"]) | set(rep["watchlist_a0"])):
        a, b = rep["watchlist_arm"].get(sub, {}), rep["watchlist_a0"].get(sub, {})
        L.append(f"| {sub.replace('|', '/')} | {a.get('n', b.get('n', '?'))} "
                 f"| {_f(a.get('precision'))}/{_f(b.get('precision'))} "
                 f"| {_f(a.get('recall'))}/{_f(b.get('recall'))} | {_f(a.get('dice'))}/{_f(b.get('dice'))} |")
    return "\n".join(L)


def _f(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Support-arm report: precision recovery, leave-one-source-out band, per-source rows.")
    p.add_argument("--arm", required=True, help="Arm per-crop json (TTA *_per_crop.json or results_per_crop.json).")
    p.add_argument("--base", required=True, help="Unconditioned-baseline per-crop json.")
    p.add_argument("--ref", default=None, help="Per-crop json for the replace-combination reference arm.")
    p.add_argument("--organelle", default="er", choices=["er", "mito"])
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    rep = report(load_per_crop(a.arm, a.split), load_per_crop(a.base, a.split),
                 load_per_crop(a.ref, a.split) if a.ref else None, a.organelle)
    md = to_markdown(rep, Path(a.arm).stem)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
        print(f"[support-report] -> {a.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
