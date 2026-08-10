"""Results aggregation across arms into one CSV + JSON + readable Markdown.

Discovers arm run dirs by the ``run.json`` marker (written by training/run_manifest.py), reads each
arm's ``results.json``, flattens the balanced-macro metrics per split, and emits a tidy table. One
row per (arm, split) over whatever splits the arm scored — ``val``, held-out-source ``test``, and,
when the manifest and config produce them, ``test_image`` and ``loso``; the Markdown table lists the
``test`` rows first. stdlib csv/json only — no pandas/matplotlib, so aggregation needs nothing beyond
the standard library and runs on a CPU-only machine.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_METRIC_COLS = ("dice", "iou", "precision", "recall", "boundary_f1", "boundary_iou", "hd95", "cldice",
                "auprc", "pq", "sq", "rq", "ap", "vi")
_ID_COLS = ("arm", "organelle", "neck", "decoder", "loss", "task", "canonical_nm", "encoder_step", "split")


def find_runs(runs_root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(runs_root).rglob("run.json"))


def _row(marker: dict, results: dict, split: str, summ: dict) -> dict:
    macro = summ.get("macro", {}) or {}
    row = {
        "arm": marker.get("name"), "organelle": marker.get("organelle"), "neck": marker.get("neck"),
        "decoder": marker.get("decoder"), "loss": "+".join(marker.get("loss", []) or []),
        "task": marker.get("task"), "canonical_nm": marker.get("canonical_nm"),
        "encoder_step": results.get("encoder_step"), "split": split,
        "n_evaluated": summ.get("n_evaluated"), "n_excluded": summ.get("n_excluded_both_empty"),
    }
    for k in _METRIC_COLS:
        v = macro.get(k)
        row[k] = round(v, 5) if isinstance(v, (int, float)) else v
    # worst-subgroup dice (robustness signal) if present
    ws = (summ.get("worst_subgroup", {}) or {}).get("dice")
    row["worst_dice"] = round(ws["value"], 5) if isinstance(ws, dict) else None
    return row


def collect(runs_root: str | Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in find_runs(runs_root):
        try:
            marker = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        res_path = run_dir / "results.json"
        if not res_path.exists():
            continue
        try:
            results = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for split, summ in (results.get("splits", {}) or {}).items():
            rows.append(_row(marker, results, split, summ))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(_ID_COLS) + ["n_evaluated", "n_excluded"] + list(_METRIC_COLS) + ["worst_dice"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = ["# Segmentation results\n", f"{len(rows)} (arm x split) rows.\n"]
    # one section per organelle; headline metrics differ (ER tubular vs mito instance)
    for organelle in sorted({r["organelle"] for r in rows if r["organelle"]}):
        er = organelle == "er"
        cols = (["arm", "neck", "decoder", "loss", "split", "dice", "cldice", "boundary_f1", "hd95"]
                if er else
                ["arm", "neck", "decoder", "loss", "split", "dice", "pq", "ap", "boundary_f1"])
        lines.append(f"\n## {organelle} (canonical "
                     f"{'2' if er else '8'} nm/px)\n")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        sub = [r for r in rows if r["organelle"] == organelle]
        # test rows first (the headline), then val
        sub.sort(key=lambda r: (r["split"] != "test", r.get("arm") or ""))
        for r in sub:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Aggregate segmentation arm results into CSV/JSON/Markdown.")
    p.add_argument("--runs-root", default="runs/segmentation_training")
    p.add_argument("--out", default=None, help="Output dir (default: <runs-root>).")
    args = p.parse_args(argv)

    rows = collect(args.runs_root)
    out = Path(args.out or args.runs_root)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "all_results.csv")
    (out / "all_results.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    write_markdown(rows, out / "summary.md")
    print(f"Aggregated {len(rows)} rows from {args.runs_root} -> {out}/all_results.csv")


if __name__ == "__main__":
    main()
