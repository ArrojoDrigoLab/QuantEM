"""Aggregate per-crop metrics into per-dataset and per-organelle summaries.
Reads <results-root>/per_crop/*.csv -> writes:
  <results-root>/per_crop_all.csv          (concatenated)
  <results-root>/per_dataset_summary.csv   (model x organelle x dataset: mean/std/median/n per metric)
  <results-root>/per_organelle_summary.csv (model x organelle: dataset-averaged + crop-pooled means)
  <results-root>/dice_matrix_<organelle>.csv (models x datasets Dice table + dataset-avg)
No pandas dependency (csv + math only)."""
import argparse, os, csv, glob, math
from collections import defaultdict, OrderedDict

BENCHMARK_ORDER = {
    "mito": ["zenodo_mitoem2", "empiar_10982_mitonet_benchmark", "orgsegnet_plant",
             "deeppi_em_skeletal_muscle", "deepcontact_tem"],
    "er": ["empiar_12885_aive", "empiar_10994_hela_sbfsem", "deepcontact_tem",
           "empiar_13156_hela_stard3_er", "lab_islet_liver_er"],
    "nucleus": ["zenodo_3675220_platynereis", "sbiad2822_nuclei",
                "segapp_islet_nucleus", "orgsegnet_plant"],
    "ld": ["empiar_13420_macrophage_a431", "deepcontact_cell", "empiar_12885_aive"],
}
DATASET_LABEL = {
    "zenodo_mitoem2": "ME2-Mossy", "empiar_10982_mitonet_benchmark": "MitoNet",
    "orgsegnet_plant": "PlantHunter", "deeppi_em_skeletal_muscle": "DeepPI",
    "deepcontact_tem": "DeepContact", "empiar_12885_aive": "AIVE",
    "empiar_10994_hela_sbfsem": "EMPIAR-10994", "empiar_13156_hela_stard3_er": "EMPIAR-13156",
    "lab_islet_liver_er": "islet/liver-ER", "zenodo_3675220_platynereis": "Platynereis",
    "sbiad2822_nuclei": "NucleiNet", "segapp_islet_nucleus": "IsletSEM",
    "empiar_13420_macrophage_a431": "MacrophageSEM", "deepcontact_cell": "DeepContact",
}


def fnum(x):
    if x in (None, "", "nan", "NaN", "None"):
        return None
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return dict(mean=None, std=None, median=None, n=0)
    n = len(v)
    mean = sum(v) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / n) if n > 1 else 0.0
    sv = sorted(v)
    med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    return dict(mean=mean, std=std, median=med, n=n)


def main(results_root):
    pc = os.path.join(results_root, "per_crop")
    out = results_root
    files = sorted(glob.glob(os.path.join(pc, "*.csv")))
    rows = []
    metric_cols = []
    for f in files:
        for r in csv.DictReader(open(f)):
            rows.append(r)
            for k in r:
                if k not in metric_cols and any(k == m or k.startswith(p) for m, p in
                    [("dice", "x"), ("iou", "x")] ) :
                    pass
    # metric columns = numeric-looking columns excluding identifiers
    idcols = {"model", "organelle", "dataset", "dataset_label", "crop_id", "coverage",
              "source_image", "status", "error", "model_type", "_variant"}
    allkeys = []
    for r in rows:
        for k in r:
            if k not in allkeys:
                allkeys.append(k)
    metric_cols = [k for k in allkeys if k not in idcols and k not in ("voxel_nm", "infer_s")]

    ok = [r for r in rows if r.get("status", "ok") == "ok"]
    # write concatenated
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "per_crop_all.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=allkeys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # group by (model, organelle, dataset)
    g = defaultdict(list)
    for r in ok:
        g[(r["model"], r["organelle"], r["dataset"])].append(r)

    # per-dataset summary
    pds_rows = []
    for (model, org, ds), rs in sorted(g.items()):
        row = OrderedDict(model=model, organelle=org, dataset=ds,
                          dataset_label=DATASET_LABEL.get(ds, ds), n_crops=len(rs))
        for mc in metric_cols:
            s = stats([fnum(r.get(mc)) for r in rs])
            row[f"{mc}_mean"] = round(s["mean"], 6) if s["mean"] is not None else ""
            row[f"{mc}_std"] = round(s["std"], 6) if s["std"] is not None else ""
            row[f"{mc}_median"] = round(s["median"], 6) if s["median"] is not None else ""
            row[f"{mc}_n"] = s["n"]
        pds_rows.append(row)
    if pds_rows:
        with open(os.path.join(out, "per_dataset_summary.csv"), "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(pds_rows[0].keys()))
            w.writeheader(); w.writerows(pds_rows)

    # per-organelle summary: dataset-averaged (unweighted over datasets) + crop-pooled
    po_rows = []
    gmo = defaultdict(lambda: defaultdict(list))   # (model,org) -> dataset -> per-dataset means
    pooled = defaultdict(lambda: defaultdict(list))  # (model,org) -> metric -> all crop vals
    for (model, org, ds), rs in g.items():
        for mc in metric_cols:
            s = stats([fnum(r.get(mc)) for r in rs])
            if s["mean"] is not None:
                gmo[(model, org)][mc].append(s["mean"])
            pooled[(model, org)][mc].extend([fnum(r.get(mc)) for r in rs])
    for (model, org), md in sorted(gmo.items()):
        row = OrderedDict(model=model, organelle=org,
                          n_datasets=len({d for (m, o, d) in g if m == model and o == org}),
                          n_crops=sum(len(rs) for (m, o, d), rs in g.items() if m == model and o == org))
        for mc in metric_cols:
            da = stats(md.get(mc, []))              # dataset-averaged
            pl = stats(pooled[(model, org)].get(mc, []))  # crop-pooled
            row[f"{mc}_datasetavg"] = round(da["mean"], 6) if da["mean"] is not None else ""
            row[f"{mc}_croppooled"] = round(pl["mean"], 6) if pl["mean"] is not None else ""
        po_rows.append(row)
    if po_rows:
        with open(os.path.join(out, "per_organelle_summary.csv"), "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(po_rows[0].keys()))
            w.writeheader(); w.writerows(po_rows)

    # dice matrix per organelle (models x datasets + dataset-avg)
    for org, dss in BENCHMARK_ORDER.items():
        models = sorted({m for (m, o, d) in g if o == org})
        if not models:
            continue
        mat = []
        header = ["model"] + [DATASET_LABEL.get(d, d) for d in dss] + ["DatasetAvg"]
        for m in models:
            r = [m]
            dvals = []
            for d in dss:
                rs = g.get((m, org, d))
                if rs:
                    s = stats([fnum(x.get("dice")) for x in rs])
                    r.append(round(s["mean"], 4) if s["mean"] is not None else "")
                    if s["mean"] is not None:
                        dvals.append(s["mean"])
                else:
                    r.append("")
            r.append(round(sum(dvals) / len(dvals), 4) if dvals else "")
            mat.append(r)
        with open(os.path.join(out, f"dice_matrix_{org}.csv"), "w", newline="") as fo:
            w = csv.writer(fo); w.writerow(header); w.writerows(mat)

    print(f"aggregated {len(rows)} rows ({len(ok)} ok) from {len(files)} files")
    print(f"models: {sorted({r['model'] for r in ok})}")
    for org in BENCHMARK_ORDER:
        ms = sorted({m for (m, o, d) in g if o == org})
        print(f"  {org}: {ms}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True,
                    help="results root: reads <root>/per_crop/*.csv, writes summaries into <root>/")
    a = ap.parse_args()
    main(a.results_root)
