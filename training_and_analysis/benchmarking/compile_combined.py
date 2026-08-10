#!/usr/bin/env python
"""Compile the combined benchmark comparison: the internal OmniEM & QuantEM models (trained on the
benchmark train/val splits, evaluated on the held-out benchmark test split) vs the external published-model
benchmark, per organelle, as per-dataset region-masked Dice + dataset-averaged mean (identical metric: the
training harness scores only non-ignore px == the external region-masking, and subgroup==dataset so
per_subgroup Dice == the benchmark's per-dataset).

Reads the internal results.json (arm dir <results-root>/<arm>/results.json; test.per_subgroup[<dataset>].dice)
and the external tables (<ext-dir>/benchmark_<org>_dice.csv). Emits, per organelle, a combined CSV +
printed leaderboard sorted by dataset-avg mean. Runs incrementally (skips arms whose results aren't in yet).

Usage: python compile_combined.py --ext-dir <benchmark tables dir> --results-root <arm results dir>
       --out-dir <output dir>
"""
import argparse, csv, json, os, sys
from pathlib import Path

# our-dataset-key -> external benchmark display column, per organelle
DATASET2COL = {
    "mito": {"zenodo_mitoem2": "ME2-Mossy", "empiar_10982_mitonet_benchmark": "MitoNet-bench",
             "orgsegnet_plant": "PlantHunter", "deeppi_em_skeletal_muscle": "DeepPI", "deepcontact_tem": "DeepContact-TEM"},
    "er":   {"empiar_10994_hela_sbfsem": "EMPIAR-10994", "empiar_12885_aive": "AIVE",
             "empiar_13156_hela_stard3_er": "EMPIAR-13156", "deepcontact_tem": "DeepContact-TEM",
             "lab_islet_liver_er": "islet/liver-ER"},
    "ld":   {"empiar_13420_macrophage_a431": "MacrophageSEM", "deepcontact_cell": "DeepContact-cell",
             "empiar_12885_aive": "AIVE"},
    "nucleus": {"zenodo_3675220_platynereis": "Platynereis", "sbiad2822_nuclei": "NucleiNet",
                "segapp_islet_nucleus": "IsletSEM", "orgsegnet_plant": "PlantHunter"},
}
# released internal arms per organelle: (display name, arm dir under --results-root)
OUR_ARMS = {"mito": [("OmniEM (ours)", "mitochondria_omniem"), ("QuantEM (ours)", "mitochondria_quantem")],
            "er":   [("OmniEM (ours)", "er_omniem"),   ("QuantEM (ours)", "er_quantem")],
            "ld":   [("OmniEM (ours)", "ld_omniem"),   ("QuantEM (ours)", "ld_quantem")],
            "nucleus": [("OmniEM (ours)", "nucleus_omniem"), ("QuantEM (ours)", "nucleus_quantem")]}


def _dice(v):
    return float(v["dice"]) if isinstance(v, dict) else float(v)


def _is_internal_row(model_name):
    """External-table rows for the internal models (any probe/pull variant) are dropped
    on load; only the released internal arms are reported, read from results.json."""
    n = model_name.lower()
    return n.startswith("omniem") or n.startswith("quantem")


def our_row(results_root, arm, org):
    rj = Path(results_root) / arm / "results.json"
    if not rj.exists():
        return None
    r = json.load(open(rj))
    ps = r["splits"]["test"]["per_subgroup"]
    cols = DATASET2COL[org]
    row = {}
    vals = []
    for dskey, col in cols.items():
        if dskey in ps:
            d = _dice(ps[dskey]); row[col] = d; vals.append(d)
        else:
            row[col] = None
    # dataset-avg mean over the benchmark datasets we have (matches external Mean(dataset_avg))
    row["Mean(dataset_avg)"] = round(sum(vals) / len(vals), 3) if vals else None
    return row


def load_external(ext_dir, org):
    p = Path(ext_dir) / f"benchmark_{org}_dice.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    cols = [c for c in rows[0].keys() if c != "model"]
    rows = [r for r in rows if not _is_internal_row(r["model"])]
    return rows, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-dir", required=True,
                    help="directory holding the external benchmark_<org>_dice.csv tables")
    ap.add_argument("--results-root", required=True,
                    help="directory holding one <arm>/results.json per released internal arm")
    ap.add_argument("--out-dir", required=True,
                    help="output directory for the combined per-organelle CSVs")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for org in ["mito", "er", "ld", "nucleus"]:
        ext_rows, cols = load_external(a.ext_dir, org)
        table = []  # (model, {col:val})
        for er in ext_rows:
            table.append((er["model"], {c: (float(er[c]) if er[c] not in ("", None) else None) for c in cols}))
        got = 0
        for disp, arm in OUR_ARMS[org]:
            row = our_row(a.results_root, arm, org)
            if row is not None:
                table.append((disp + " *", {c: row.get(c) for c in cols})); got += 1
        # sort by Mean(dataset_avg) desc
        mkey = "Mean(dataset_avg)"
        table.sort(key=lambda t: (t[1].get(mkey) is not None, t[1].get(mkey) or -1), reverse=True)
        # write combined CSV
        outp = Path(a.out_dir) / f"benchmark_{org}_combined.csv"
        with open(outp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["model"] + cols)
            for m, d in table:
                w.writerow([m] + [("" if d.get(c) is None else f"{d[c]:.3f}") for c in cols])
        # print leaderboard
        print(f"\n===== {org.upper()}  (ours = *; {got}/2 our models in)  ->  {outp.name} =====")
        hdr = f"{'model':26s} " + " ".join(f"{c[:12]:>13s}" for c in cols)
        print(hdr)
        for m, d in table:
            print(f"{m:26s} " + " ".join((f"{d[c]:>13.3f}" if d.get(c) is not None else f"{'-':>13s}") for c in cols))


if __name__ == "__main__":
    main()
