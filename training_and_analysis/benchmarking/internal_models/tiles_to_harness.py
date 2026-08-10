#!/usr/bin/env python
"""Transcode a benchmark tile build (<src>/<org>/<split>/*.tif + manifest_<org>.csv, from
segmentation_dataset/benchmark_tiles/build_benchmark_tiles.py) into the
segmentation_training.harness.run_seg group format (PNG tiles + manifest.jsonl), so the
segmentation models can train on the benchmark train split and evaluate on the held-out
benchmark test split. One per-organelle root: <dst>/<org>/{train,val,test}/*.png +
<dst>/<org>/manifest.jsonl.

  er      -> the --native tile build (ER kept at each crop's source resolution)
  ld      -> the 8 nm tile build
  nucleus -> the 25 nm tile build
  mito    -> the 8 nm tile build (the benchmark mito group is instead built by
             build_mito_cem_group.py, which adds the regrid + CEM train pools)

manifest record schema matches segmentation_training build_dataset (group=benchmark_<org>,
bucket=canonical); subgroup=dataset so the harness per-subgroup macro Dice averages per test
DATASET, matching the external-model benchmark. Labels stay {0=bg,1=fg,255=ignore}; run_seg
honors ignore in loss + Dice. inst_path=null (semantic Dice; instance decoders derive
pseudo-instances from the binary GT at train time). annotation_bbox omitted ->
random-crop path (correct for these full-frame crops).

Usage: python tiles_to_harness.py --src-root <tile build dir> --org <er|mito|ld|nucleus> --dst-root <dir> [--procs N]
"""
import argparse, csv, json, os, sys
from multiprocessing import Pool
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from segmentation_training.dataprep.io import write_png_L

TARGET_NM = {"er": "native", "mito": 8.0, "ld": 8.0, "nucleus": 25.0}
_ARGS = None


def _init(src_root, org, dst_org):
    global _ARGS
    _ARGS = (src_root, org, dst_org)


def _one(row):
    src_root, org, dst_org = _ARGS
    name, split, dataset = row["name"], row["split"], row["dataset"]
    em_tif = os.path.join(src_root, org, split, name + "_em.tif")
    lb_tif = os.path.join(src_root, org, split, name + "_label.tif")
    if not (os.path.exists(em_tif) and os.path.exists(lb_tif)):
        return None
    em = np.asarray(tifffile.imread(em_tif))
    lb = np.asarray(tifffile.imread(lb_tif))
    if em.dtype != np.uint8:
        em = em.astype(np.uint8)
    lb = lb.astype(np.uint8)
    H, W = lb.shape[:2]
    outdir = os.path.join(dst_org, split)
    em_rel = f"{split}/{name}_em.png"
    mk_rel = f"{split}/{name}_mask.png"
    write_png_L(os.path.join(outdir, name + "_em.png"), em)
    write_png_L(os.path.join(outdir, name + "_mask.png"), lb)
    fg = int((lb == 1).sum()); ig = int((lb == 255).sum())
    return {
        "sample_id": name, "organelle": org, "group": f"benchmark_{org}", "bucket": "canonical",
        "scale_mode": "canonical", "split": split, "subgroup": dataset, "collection": "gt",
        "dataset": dataset, "crop_id": row.get("crop_id", ""), "modality": row.get("modality", ""),
        "scale_band": row.get("scale_band", ""), "tissue_context": row.get("tissue_context", ""),
        "species_group": row.get("species_group", ""), "coverage_tier": row.get("coverage_tier", ""),
        "canonical_nm": (None if org == "er" else TARGET_NM[org]),
        "src_nm_row": (float(row["src_voxel_nm"]) if row.get("src_voxel_nm") else None),
        "em_path": em_rel, "mask_path": mk_rel, "inst_path": None, "gt_is_instance": False,
        "height": H, "width": W, "fg_px": fg, "ignore_px": ig, "valid_px": H * W - ig,
        "target_nm": row.get("target_nm", ""), "achieved_nm": row.get("achieved_nm", ""),
        "capped": row.get("capped", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--org", required=True, choices=["er", "mito", "ld", "nucleus"])
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--min-fg-train-px", type=int, default=0,
                    help="skip TRAIN/VAL records with fewer than this many foreground px (sparse-organelle "
                         "de-emptying, e.g. LD=1); TEST is always kept in full for the benchmark.")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.src_root, f"manifest_{a.org}.csv"), encoding="utf-8")))
    if a.min_fg_train_px > 0:
        n0 = len(rows)
        rows = [r for r in rows if r["split"] == "test" or int(r.get("fg_px", 0) or 0) >= a.min_fg_train_px]
        print(f"  fg-filter (train/val >= {a.min_fg_train_px}px): kept {len(rows)}/{n0}", flush=True)
    dst_org = os.path.join(a.dst_root, a.org)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(dst_org, sp), exist_ok=True)
    nproc = a.procs or max(1, min(16, (os.cpu_count() or 4) - 2))
    print(f"transcoding {a.org}: {len(rows)} rows, {nproc} procs -> {dst_org}", flush=True)
    recs, done = [], 0
    with Pool(nproc, initializer=_init, initargs=(a.src_root, a.org, dst_org)) as pool:
        for r in pool.imap_unordered(_one, rows, chunksize=8):
            done += 1
            if r:
                recs.append(r)
            if done % 300 == 0:
                print(f"  {done}/{len(rows)} ({len(recs)} written)", flush=True)
    with open(os.path.join(dst_org, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    import collections
    bysplit = collections.Counter(r["split"] for r in recs)
    print(f"=== {a.org} DONE: {len(recs)} records {dict(bysplit)} -> {dst_org}/manifest.jsonl", flush=True)


if __name__ == "__main__":
    main()
