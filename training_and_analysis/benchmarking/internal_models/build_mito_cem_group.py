"""Transcode the benchmark mito tile build (8 nm; standard + regrid + CEM train pools) into the
segmentation_training.harness.run_seg group format (group=benchmark_mito_cem):
<out>/{train,val,test}/*.png + <out>/manifest.jsonl.

Rows come from <tile-root>/manifest_mito_regrid_cem.csv, written by
segmentation_dataset/benchmark_tiles/build_benchmark_tiles.py. Row -> source-tile directory
(branch on `kind`):
  split in (val,test)                      -> mito/<split>/
  train & kind==cem_mitolab                -> mito_cem_clean/train/
  train & kind==regrid_cell                -> mito_regrid/train/
  train & kind in (standard,openorganelle) -> mito/train/     (the original full-size tiles)
`<name>` (NOT crop_id) is the on-disk basename (already carries __r<cy>_<cx> / __z<plane>).
EM cast uint8 (no normalize), label kept {0,1,255} verbatim. Output = <split>/<name>_{em,mask}.png,
records the same 29-field schema tiles_to_harness.py emits. No rescale (tiles are already 8 nm).

Usage: python build_mito_cem_group.py --tile-root <tile build dir> --out <group dir> [--procs N]
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np
import tifffile
from PIL import Image

ORG = "mito"
CANON_NM = 8.0
_PATHS = None


def _init(tile_root, dst):
    global _PATHS
    _PATHS = (tile_root, dst)


def resolve_dir(split, kind):
    if split in ("val", "test"):
        return f"mito/{split}"
    if kind == "cem_mitolab":
        return "mito_cem_clean/train"
    if kind == "regrid_cell":
        return "mito_regrid/train"
    return "mito/train"  # standard | openorganelle originals


def write_png_L(path, arr):
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    for _ in range(4):
        try:
            Image.fromarray(a, mode="L").save(path, format="PNG")
            Image.open(path).load()  # verify not truncated
            return True
        except Exception:
            continue
    return False


def one(row):
    tile_root, dst = _PATHS
    name, split, kind, dataset = row["name"], row["split"], row["kind"], row["dataset"]
    sd = resolve_dir(split, kind)
    em_tif = os.path.join(tile_root, sd, name + "_em.tif")
    lb_tif = os.path.join(tile_root, sd, name + "_label.tif")
    if not (os.path.exists(em_tif) and os.path.exists(lb_tif)):
        return ("missing", name)
    try:
        em = np.asarray(tifffile.imread(em_tif))
        if em.dtype != np.uint8:
            em = em.astype(np.uint8)
        lb = np.asarray(tifffile.imread(lb_tif)).astype(np.uint8)
    except Exception as e:
        return ("readerr", f"{name}: {e}")
    H, W = lb.shape[:2]
    outdir = os.path.join(dst, split)
    os.makedirs(outdir, exist_ok=True)
    em_png = os.path.join(outdir, name + "_em.png")
    mk_png = os.path.join(outdir, name + "_mask.png")
    if not (write_png_L(em_png, em) and write_png_L(mk_png, lb)):
        return ("writeerr", name)
    fg = int((lb == 1).sum()); ig = int((lb == 255).sum()); valid = H * W - ig

    def g(k):
        return row.get(k, "") or ""
    src_nm = g("src_voxel_nm")
    rec = {
        "sample_id": name, "organelle": ORG, "bucket": "canonical", "scale_mode": "canonical",
        "split": split, "subgroup": dataset, "collection": "gt", "dataset": dataset,
        "crop_id": g("crop_id"), "modality": g("modality"), "scale_band": g("scale_band"),
        "tissue_context": g("tissue_context"), "species_group": g("species_group"),
        "coverage_tier": g("coverage_tier"), "canonical_nm": CANON_NM,
        "src_nm_row": (float(src_nm) if src_nm not in ("", None) else None),
        "em_path": f"{split}/{name}_em.png", "mask_path": f"{split}/{name}_mask.png",
        "inst_path": None, "gt_is_instance": False, "height": H, "width": W,
        "fg_px": fg, "ignore_px": ig, "valid_px": valid,
        "target_nm": g("target_nm"), "achieved_nm": g("achieved_nm"), "capped": g("capped"),
    }
    return ("ok", rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-root", required=True,
                    help="benchmark tile build root (contains manifest_mito_regrid_cem.csv)")
    ap.add_argument("--out", required=True, help="destination group directory")
    ap.add_argument("--procs", type=int, default=min(16, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    csv_path = os.path.join(args.tile_root, "manifest_mito_regrid_cem.csv")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"{len(rows)} rows; transcoding with {args.procs} procs -> {args.out}", flush=True)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out, sp), exist_ok=True)
    recs = []
    n = ok = miss = err = 0
    with Pool(args.procs, initializer=_init, initargs=(args.tile_root, args.out)) as p:
        for res in p.imap_unordered(one, rows, chunksize=16):
            n += 1
            if res[0] == "ok":
                ok += 1
                recs.append(res[1])
            elif res[0] == "missing":
                miss += 1
            else:
                err += 1
                if err <= 10:
                    print(f"  ERR {res}", flush=True)
            if n % 2000 == 0:
                print(f"  {n}/{len(rows)} (ok={ok} miss={miss} err={err})", flush=True)
    with open(os.path.join(args.out, "manifest.jsonl"), "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps({**r, "group": "benchmark_mito_cem"}) + "\n")
    print(f"DONE: {ok} tiles ok, {miss} missing, {err} err. manifest={len(recs)}", flush=True)
    # split sanity
    from collections import Counter
    print("  splits:", Counter(r["split"] for r in recs))


if __name__ == "__main__":
    main()
