"""Open-source organelle-segmentation model benchmark tables.

Reads the per-crop Dice table (<data>/per_crop_all.csv, written by aggregate.py) and writes
one dataset-averaged summary table per organelle into <data>/tables/:

  benchmark_<org>_dice.csv  rows = models (sorted by mean Dice desc), columns = each
                            held-out test dataset (mean Dice) + Mean(dataset_avg).
  benchmark_indomain.csv    the (organelle, model, dataset) cells that fall inside a
                            model's own training domain -> flagged / hatched in the figure.
  benchmark_mito_pq.csv     mitochondria Panoptic Quality on the instance-labelled datasets.

The per-crop table is produced by the shared harness (./harness/) + aggregate.py; the
internal models' per-dataset Dice is read from <data>/tables/benchmark_ours_dice.csv when present.
"""
import argparse, os, csv
import numpy as np
import pandas as pd

ORGS = ["mito", "er", "nucleus", "ld"]
DATASETS = {
    "mito": ["zenodo_mitoem2", "empiar_10982_mitonet_benchmark", "orgsegnet_plant",
             "deeppi_em_skeletal_muscle", "deepcontact_tem"],
    "er": ["empiar_12885_aive", "empiar_10994_hela_sbfsem", "deepcontact_tem",
           "empiar_13156_hela_stard3_er", "lab_islet_liver_er"],
    "nucleus": ["zenodo_3675220_platynereis", "sbiad2822_nuclei",
                "segapp_islet_nucleus", "orgsegnet_plant"],
    "ld": ["empiar_13420_macrophage_a431", "deepcontact_cell", "empiar_12885_aive"],
}
DS_LABEL = {
    "zenodo_mitoem2": "ME2-Mossy", "empiar_10982_mitonet_benchmark": "MitoNet-bench",
    "orgsegnet_plant": "PlantHunter", "deeppi_em_skeletal_muscle": "DeepPI",
    "deepcontact_tem": "DeepContact-TEM", "empiar_12885_aive": "AIVE",
    "empiar_10994_hela_sbfsem": "EMPIAR-10994", "empiar_13156_hela_stard3_er": "EMPIAR-13156",
    "lab_islet_liver_er": "islet/liver-ER", "zenodo_3675220_platynereis": "Platynereis",
    "sbiad2822_nuclei": "NucleiNet", "segapp_islet_nucleus": "IsletSEM",
    "empiar_13420_macrophage_a431": "MacrophageSEM", "deepcontact_cell": "DeepContact-cell",
}
MODEL_NAME = {
    "mitonet": "MitoNet (native)", "mitonet_scaled": "MitoNet (rescaled)",
    "nucleonet": "NucleoNet", "nucleonet_scaled": "NucleoNet (rescaled)",
    "lipidnet": "DropNet", "lipidnet_scaled": "DropNet (rescaled)",
    "deepcontact_mito": "DeepContact", "deepcontact_er": "DeepContact",
    "microsam_vit_l_em_organelles": "micro-sam", "incasem": "incasem",
    "orgsegnet": "OrgSegNet",
    "omniem": "OmniEM ViT-L", "quantem": "QuantEM ViT-B",
}
SCALED = {"mitonet_scaled", "nucleonet_scaled", "lipidnet_scaled"}
# (model, dataset) pairs that fall inside the model's own training domain
INDOMAIN = {("deepcontact_mito", "deepcontact_tem"), ("deepcontact_er", "deepcontact_tem"),
            ("deepcontact_er", "deepcontact_cell"), ("deepcontact_mito", "deepcontact_cell"),
            ("orgsegnet", "orgsegnet_plant"),
            # the internal models also train on PlantHunter + DeepContact
            ("omniem", "orgsegnet_plant"), ("quantem", "orgsegnet_plant"),
            ("omniem", "deepcontact_tem"), ("quantem", "deepcontact_tem"),
            ("omniem", "deepcontact_cell"), ("quantem", "deepcontact_cell")}
# only these mito datasets carry instance-level GT -> Panoptic Quality is defined there
INST_DS = ["zenodo_mitoem2", "empiar_10982_mitonet_benchmark"]


def write_mito_pq(df, data_dir):
    """Mitochondria Panoptic Quality (instance) table: PQ = SQ x RQ, plus the SQ/RQ split.
    Only the two mito datasets with instance-labelled GT; semantic-only models are absent."""
    d = df[df["organelle"] == "mito"].copy()
    for c in ("inst_pq", "inst_sq", "inst_rq"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d = d[d["dataset"].isin(INST_DS) & d["inst_pq"].notna()]
    rows = []
    for m in d["model"].unique():
        mm = d[d["model"] == m]
        per = {}
        for ds in INST_DS:
            v = mm[mm["dataset"] == ds]
            if len(v):
                per[ds] = (float(v["inst_pq"].mean()), float(v["inst_sq"].mean()),
                           float(v["inst_rq"].mean()))
        if not per:
            continue
        rows.append(dict(key=m, per=per,
                         meanpq=float(np.mean([p[0] for p in per.values()])),
                         sq=float(np.mean([p[1] for p in per.values()])),
                         rq=float(np.mean([p[2] for p in per.values()])),
                         n=int(mm["crop_id"].nunique())))
    rows.sort(key=lambda r: -r["meanpq"])
    out = os.path.join(data_dir, "benchmark_mito_pq.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [DS_LABEL[d_] for d_ in INST_DS] + ["Mean_PQ", "SQ", "RQ"])
        for r in rows:
            line = [MODEL_NAME.get(r["key"], r["key"])]
            for ds in INST_DS:
                line.append(f"{r['per'][ds][0]:.3f}" if ds in r["per"] else "")
            line += [f"{r['meanpq']:.3f}", f"{r['sq']:.3f}", f"{r['rq']:.3f}"]
            w.writerow(line)
    print(f"wrote benchmark_mito_pq.csv  ({len(rows)} model rows; instance GT on {len(INST_DS)} datasets)")


def main(data_dir):
    tables = os.path.join(data_dir, "tables")
    os.makedirs(tables, exist_ok=True)
    src = os.path.join(data_dir, "per_crop_all.csv")
    df = pd.read_csv(src)
    if "status" in df:
        df = df[df["status"] == "ok"]
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
    # the internal foundation models (aggregated per-dataset Dice from the harness results)
    ours = {}
    op = os.path.join(tables, "benchmark_ours_dice.csv")
    if os.path.exists(op):
        for r in csv.DictReader(open(op)):
            try:
                ours[(r["model"], r["organelle"], r["dataset"])] = float(r["dice"])
            except (TypeError, ValueError):
                pass
    indomain = []

    for org in ORGS:
        dss = DATASETS[org]
        sub = df[df["organelle"] == org]
        rows = []
        for m in sub["model"].unique():
            cells = {}
            for ds in dss:
                v = sub[(sub["model"] == m) & (sub["dataset"] == ds)]["dice"].dropna()
                if len(v):
                    cells[ds] = float(v.mean())
            if cells:
                rows.append(dict(key=m, cells=cells, mean=float(np.mean(list(cells.values())))))
        for om in ("omniem", "quantem"):
            cells = {ds: ours[(om, org, ds)] for ds in dss if (om, org, ds) in ours}
            if cells:
                rows.append(dict(key=om, cells=cells, mean=float(np.mean(list(cells.values())))))
        rows.sort(key=lambda r: -r["mean"])

        out = os.path.join(tables, f"benchmark_{org}_dice.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            # columns = each dataset + the per-model Mean across datasets (no dataset-mean row, no n)
            w.writerow(["model"] + [DS_LABEL[d] for d in dss] + ["Mean(dataset_avg)"])
            for r in rows:
                line = [MODEL_NAME.get(r["key"], r["key"])]
                for d in dss:
                    line.append(f"{r['cells'][d]:.3f}" if d in r["cells"] else "")
                line += [f"{r['mean']:.3f}"]
                w.writerow(line)
        print(f"wrote {os.path.basename(out)}  ({len(rows)} model rows)")

        for r in rows:
            for d in dss:
                if (r["key"], d) in INDOMAIN and d in r["cells"]:
                    indomain.append([org, MODEL_NAME.get(r["key"], r["key"]),
                                     DS_LABEL[d], f"{r['cells'][d]:.3f}"])

    with open(os.path.join(tables, "benchmark_indomain.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["organelle", "model", "dataset", "dice"])
        w.writerows(indomain)
    print(f"wrote benchmark_indomain.csv  ({len(indomain)} in-domain cells)")

    write_mito_pq(df, tables)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True,
                    help="results root holding per_crop_all.csv (+ optional "
                         "tables/benchmark_ours_dice.csv); tables are written to <data>/tables/")
    a = ap.parse_args()
    main(a.data)
