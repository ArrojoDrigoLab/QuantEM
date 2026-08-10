"""Stratify the scale sweep by native composition — a confound guard.

A ranking such as 2 nm/px beating 4 nm/px is confounded whenever most sources are natively coarser than
the finer bin: the fine bin is then largely upsampled, with fabricated detail, while the coarse bin is
largely native, so the ranking may be reporting native-beats-upsampled rather than a true scale
optimum. This reports, per scale arm, what fraction of its evaluation regions were upsampled,
downsampled or near-native, taken from each record's ``resample_factor``.

It accompanies the sweep ranking, because without it a resampling artifact is indistinguishable from a
scale effect. It reads each arm's derived manifest and nothing else — no checkpoint is loaded and no
model is built, so it runs without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path


def _factor(rec: dict) -> float | None:
    rf = rec.get("resample_factor")
    if isinstance(rf, (list, tuple)) and rf:
        vals = [float(x) for x in rf if x]
        return (sum(vals) / len(vals)) if vals else None
    if isinstance(rf, (int, float)) and rf:
        return float(rf)
    return None


def native_composition(records: list[dict], *, near_tol: float = 0.05) -> dict:
    """Fraction of ``records`` upsampled / downsampled / near-native (+ native-nm distribution).

    ``near_tol`` = |factor-1| band counted as native. Native-bucket crops (resample_factor None) count as
    near-native. Also returns the mean/median native ``src_nm`` and the mean upsample factor of the
    upsampled crops (how much fabricated detail the bin carries)."""
    n = len(records)
    up = down = near = 0
    up_factors, src_nms = [], []
    for r in records:
        f = _factor(r)
        src = r.get("src_nm_col") or r.get("src_nm_row")
        if src:
            src_nms.append(float(src))
        if f is None or abs(f - 1.0) <= near_tol:
            near += 1
        elif f > 1.0:
            up += 1
            up_factors.append(f)
        else:
            down += 1
    src_nms.sort()
    return {
        "n": n,
        "frac_upsampled": (up / n) if n else None,
        "frac_downsampled": (down / n) if n else None,
        "frac_near_native": (near / n) if n else None,
        "mean_upsample_factor": (sum(up_factors) / len(up_factors)) if up_factors else None,
        "median_src_nm": (src_nms[len(src_nms) // 2] if src_nms else None),
        "mean_src_nm": (sum(src_nms) / len(src_nms)) if src_nms else None,
    }


def composition_for_arm(data_root: str | Path, group: str, split: str, bucket: str = "canonical") -> dict:
    """Native composition of one scale arm's eval split, read straight from its manifest."""
    from ...harness.dataset import load_manifest
    recs = load_manifest(data_root, group, split, bucket=bucket)
    return native_composition(recs)


def stratify_sweep(data_roots: dict, *, organelles=("er", "mito"), split: str = "test") -> list[dict]:
    """For each scale arm in ``data_roots`` (name -> data_root), read its eval-split composition. Returns
    the composition rows that accompany the sweep ranking as its confound guard."""
    rows = []
    for name, root in data_roots.items():
        org = "er" if "_er" in name or name.startswith("er") else ("mito" if "mito" in name else None)
        if org is None or (organelles and org not in organelles):
            continue
        bucket = "native" if name.endswith("native") else "canonical"
        try:
            comp = composition_for_arm(root, f"group2_{org}", split, bucket=bucket)
        except FileNotFoundError:
            comp = {"n": 0, "error": "manifest not found under this arm's data root"}
        rows.append({"arm": name, "organelle": org, "bucket": bucket, "split": split, **comp})
    return rows


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Scale-sweep native-composition confound guard.")
    p.add_argument("--data-roots", required=True,
                   help="JSON map of scale arm name to data root, written by run_scale gen-configs")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    droots = json.loads(Path(a.data_roots).read_text(encoding="utf-8"))
    rows = stratify_sweep(droots, split=a.split)

    def _pct(x):
        return f"{100 * x:.0f}" if isinstance(x, (int, float)) else "-"

    def _num(x, fmt):
        return format(x, fmt) if isinstance(x, (int, float)) else "-"

    print(f"{'arm':<20}{'n':>6}{'%up':>7}{'%down':>8}{'%near':>8}{'meanUpF':>9}{'medSrcNm':>10}")
    for r in rows:
        print(f"{r['arm']:<20}{r.get('n', 0):>6}{_pct(r.get('frac_upsampled')):>7}"
              f"{_pct(r.get('frac_downsampled')):>8}{_pct(r.get('frac_near_native')):>8}"
              f"{_num(r.get('mean_upsample_factor'), '.2f'):>9}{_num(r.get('median_src_nm'), '.1f'):>10}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
