#!/usr/bin/env python
"""Materialize the benchmark train/val/test tile sets from the final_* splits.

Three stages, run in order against the same output root:

  build     splits/final_<org>.csv + the corpus -> per-organelle standardized tiles
            (<out>/<org>/<split>/<name>_{em,label}.tif) + <out>/manifest_<org>.csv
  regrid    balanced 512 px regrid of the mitochondria TRAIN split
            -> <out>/mito_regrid/train/ + <out>/manifest_mito_regrid.csv
  add-cem   filtered CEM-MitoLab (EMPIAR-11037) tiles appended as an optional,
            train-only pool -> <out>/mito_cem_clean/train/ + <out>/manifest_mito_regrid_cem.csv

build:
  * SPLITS   = splits/final_{er,mito,nucleus,ld}.csv (test = held-out benchmark; train/val =
               per-organelle 80/20 of every other same-organelle crop). One dataset per organelle.
  * STANDARDIZE = resample every crop to the organelle target nm/px (er 4.0, mito 8.0,
               ld 8.0, nucleus 25.0). scale = src_voxel_nm / target_nm. EM: INTER_AREA
               (downscale) / INTER_LINEAR (upscale); labels: INTER_NEAREST. Max output dim
               capped at MAX_DIM=8192 px; capped crops keep the finest res that fits and are
               flagged. --native keeps each crop at its source resolution instead (no resample).
  * LABELS   = single uint8 TIF per crop: 0=background, 1=<organelle>, 255=ignore.
               ignore = (partial/sparse coverage) real-EM outside the annotation bbox; the
               zero-pad outside valid_region is removed by cropping to valid_region.
               'full' coverage crops carry no ignore. openOrganelle cubes are densely
               annotated -> fully GT, no ignore.
  * openOrganelle (jrc_*) = 3D volumes -> extract 2D planes >=400 nm apart through the
               annotated sub-volume, registered EM<->label via the crop_manifest coordinates.
  * unknown-res crops carry flagged estimates in the manifests (crop_build/patch_estimated_res.py);
               voxel_estimated is propagated to the output manifest.

regrid (mitochondria train only; val/test are frozen):
  * Grid = zero-overlap 512 floor grid over each train tile; a cell is kept only if
    valid_frac >= MIN_VALID (0.5) and fg_px >= MIN_FG_PX (256).
  * Per-source caps on TOTAL train count (current + added): each zenodo_mitoem2 ME2-* sub-volume
    600; every other dataset (incl. each jrc_* subset) 1000; orgsegnet_plant is set to 15% of the
    total dataset across all splits. ME2-Mossy is a held-out TEST source -> never regridded.

add-cem:
  * A CEM source volume is kept iff its directory name does not start with "jrc_" and contains
    none of: wei2020_mitoem, kasthuri, guay, openorganelle (name-overlap with benchmark test/val
    pools or previously used sources) nor bleck, cremi, perez (flagged benchmark-adjacent).
  * Masks: CEM instance (uint16) -> semantic 0/1 uint8; ignore_px = 0. Not resampled (CEM
    carries no reliable scale); scale/target_nm fields left blank.
  * Every kept tile: dataset='cem_mitolab', kind='cem_mitolab', split='train',
    split_role='train_cem_optional', subgroup = the CEM source-volume directory.

Usage:
  python build_benchmark_tiles.py build   --out <root> [--org er|mito|ld|nucleus|both]
                                          [--native] [--datasets ds1,ds2] [--limit N] [--dry-run]
  python build_benchmark_tiles.py regrid  --out <root> [--write]
  python build_benchmark_tiles.py add-cem --out <root> --cem-zip <cem_mitolab.zip>
                                          --cem-xlsx <cem_mitolab_metadata.xlsx>
The corpus root comes from SEG_CORPUS_ROOT or --corpus-root.
"""
import csv, json, os, sys, glob, math, argparse, collections, zipfile, io, re, random
from multiprocessing import Pool
import numpy as np
import tifffile
import cv2

ROOT = os.environ.get("SEG_CORPUS_ROOT", ".")
TARGET_NM = {"er": 4.0, "mito": 8.0, "ld": 8.0, "nucleus": 25.0}
MAX_DIM = 8192
Z_SPACING_NM = 400.0          # openOrganelle plane sampling
# ROI-limited datasets whose manifest coverage_tier is 'full' (real-EM fraction=1.0) but where
# only a small annotation window is actually labelled -> outside annotation_bbox is ignore,
# not background. Only webknossos_fastem_mito matches this pattern.
ROI_LIMITED = {"webknossos_fastem_mito"}

# ------------------------------------------------------------------ small IO helpers
def imread(p):
    return np.asarray(tifffile.imread(p))

def to_2d_em(a):
    a = np.asarray(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[-1] in (3, 4) and a.shape[-1] < a.shape[0]:   # RGB(A) -> luma ch0 (EM is grey)
            return a[..., 0]
        return a[a.shape[0] // 2]                                # unexpected z-stack -> mid plane
    return a.reshape(a.shape[-2], a.shape[-1])

def to_2d_label(a):
    a = np.asarray(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[-1] in (3, 4) and a.shape[-1] < a.shape[0]:   # RGB mask -> any-nonzero across ch
            return a.max(axis=-1)
        return a[a.shape[0] // 2]
    return a.reshape(a.shape[-2], a.shape[-1])

def ensure_u8_em(a):
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return ((a - lo) * (255.0 / (hi - lo))).round().astype(np.uint8)

def resize_scale(arr, scale, interp):
    h, w = arr.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    if (nh, nw) == (h, w):
        return arr
    return cv2.resize(arr, (nw, nh), interpolation=interp)

def capped_scale(h, w, scale):
    """Clamp so max output dim <= MAX_DIM. Returns (scale, capped)."""
    if max(h, w) * scale <= MAX_DIM:
        return scale, False
    return MAX_DIM / float(max(h, w)), True

# ------------------------------------------------------------------ label encoding
def _org_keys(org, op):
    """Keys in organelles_present belonging to `org`, for multi-key organelles. Returns None for
    er/mito (they use the original single-entry path). Matchers:
      LD  -> lipid_droplet / ld / lds / lipid_droplet_{above,below,touching}_nucleus (union all)
      nuc -> nucleus / nuclei / nuc  (excludes nuclear_envelope, nucleolus)."""
    if org == "ld":
        match = lambda k: ("lipid" in k) or k in ("ld", "lds")
    elif org == "nucleus":
        match = lambda k: k in ("nucleus", "nuclei", "nuc")
    else:
        return None
    return [k for k in op if match(k.lower())]


def organelle_mask(label2d, crop_rec, org):
    """Boolean foreground mask for `org` from a standard 2D label plane."""
    op = crop_rec.get("organelles_present", {}) or {}
    # LD / nucleus: union the recorded values of ALL matching keys (a crop can annotate several
    # positional lipid-droplet sub-classes / a nucleus under nucleus|nuclei|nuc).
    mk = _org_keys(org, op)
    if mk is not None:
        vals = set()
        for k in mk:
            v = op[k]
            if isinstance(v, dict) and v.get("value") is not None:
                try:
                    vals.add(int(v["value"]))
                except (TypeError, ValueError):
                    pass
        if vals:
            return np.isin(label2d, list(vals))
        return label2d != 0   # single-organelle binary label, no recorded value
    # find the matching organelle entry (canonical names already match 'er'/'mito')
    entry = op.get(org)
    if entry is None:
        for k, v in op.items():
            if k.lower().startswith(org):
                entry = v; break
    uniq = None
    if isinstance(entry, dict) and "value" in entry and entry["value"] is not None:
        val = int(entry["value"])
        uniq = np.unique(label2d)
        if val in uniq:
            return label2d == val
        # value recorded but absent at this scale/plane -> fall through to nonzero if binary
    # instance / binary encoding: nonzero == the (single) organelle
    if uniq is None:
        uniq = np.unique(label2d)
    nz = uniq[uniq != 0]
    if len(nz) <= 1:
        return label2d != 0
    # multi-value label but no usable 'value' -> ambiguous; treat nonzero (report upstream)
    return label2d != 0

# ------------------------------------------------------------------ standard crop
def process_standard(ds, cid, crop_rec, org, target, native=False):
    em_p = os.path.join(ROOT, ds, crop_rec["em_file"])
    lb_p = os.path.join(ROOT, ds, crop_rec["label_file"])
    if not (os.path.exists(em_p) and os.path.exists(lb_p)):
        return None, "missing_file"
    em = ensure_u8_em(to_2d_em(imread(em_p)))
    lab = to_2d_label(imread(lb_p))
    H, W = em.shape
    if lab.shape != (H, W):
        lab = cv2.resize(lab.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
    fg = organelle_mask(lab, crop_rec, org)

    vr = crop_rec.get("valid_region_in_canvas_xyxy") or [0, 0, W, H]
    x0, y0, x1, y1 = [int(round(v)) for v in vr]
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, "empty_valid_region"
    tier = (crop_rec.get("coverage_tier") or "").lower()

    # ignore mask in canvas coords: sparse/partial (or known ROI-limited) -> outside annotation bbox.
    # 'full'-tier crops are image-complete (whole valid_region is GT) EXCEPT ROI_LIMITED datasets.
    ignore = np.zeros((H, W), bool)
    ab = crop_rec.get("annotation_bbox_in_canvas_xyxy")
    roi_limited = tier in ("sparse", "partial") or ds in ROI_LIMITED
    if roi_limited and ab and len(ab) == 4:
        ax0, ay0, ax1, ay1 = [int(round(v)) for v in ab]
        m = np.ones((H, W), bool)
        m[max(0, ay0):min(H, ay1), max(0, ax0):min(W, ax1)] = False
        ignore = m

    em_c = em[y0:y1, x0:x1]
    fg_c = fg[y0:y1, x0:x1]
    ig_c = ignore[y0:y1, x0:x1]
    label3 = np.zeros(em_c.shape, np.uint8)
    label3[fg_c] = 1
    label3[ig_c] = 255

    src = float(crop_rec["_src_voxel"])
    h, w = em_c.shape
    if native:
        scale, capped = 1.0, False          # native: keep source resolution, no resample
    else:
        scale = src / target
        scale, capped = capped_scale(h, w, scale)
    em_out = resize_scale(em_c, scale, cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    lab_out = resize_scale(label3, scale, cv2.INTER_NEAREST)
    achieved = src / scale
    return [(em_out, lab_out, {"kind": "standard", "plane": "", "capped": capped,
                               "achieved_nm": round(achieved, 3), "scale": round(scale, 5)})], None

# ------------------------------------------------------------------ openOrganelle crop
def pick_planes(plane_idx, z0, z1, orig_res_z):
    """Indices of planes within [z0,z1) subsampled to >= Z_SPACING_NM apart."""
    cand = [(i, plane_idx[i]) for i in range(len(plane_idx)) if z0 <= plane_idx[i] < z1]
    out, last = [], None
    for i, z in cand:
        if last is None or abs(z - last) * orig_res_z >= Z_SPACING_NM:
            out.append((i, z)); last = z
    return out

def process_openorganelle(cid, org, target, native=False):
    cdir = os.path.join(ROOT, "openOrganelle", cid.replace("/", os.sep))
    cmp_ = os.path.join(cdir, "crop_manifest.json")
    raw_p = os.path.join(cdir, "raw_xy.tif")
    seg_p = os.path.join(cdir, "seg_%s.tif" % org)
    if not (os.path.exists(cmp_) and os.path.exists(raw_p) and os.path.exists(seg_p)):
        return None, "missing_oo_file"
    cm = json.load(open(cmp_, encoding="utf-8"))
    rx = cm.get("raw_xy") or {}
    if rx.get("tile_axes_rows_cols") != ["y", "x"]:
        return None, "oo_unexpected_orientation"
    orig_res = cm["original_image"]["resolution_nm_zyx"]      # [rz,ry,rx]
    win = rx["window_bbox_in_original_voxels_emS0"]           # {y:[..],x:[..]}
    planes = rx["plane_indices_emS0"]
    seg_rec = next((s for s in cm.get("segmentations", []) if s.get("class_name") == org), None)
    if seg_rec is None:
        return None, "oo_no_seg_record"
    loc = seg_rec["location_in_original_image_voxels_emS0"]   # {z,y,x} original voxels
    seg_res = seg_rec["resolution_nm_zyx"]                    # [sz,sy,sx]

    raw = imread(raw_p)                                       # (P, Y, X)
    seg = imread(seg_p)                                       # (Zs, Ys, Xs)
    if raw.ndim != 3 or seg.ndim != 3:
        return None, "oo_bad_dims"
    ry0 = int(loc["y"][0] - win["y"][0]); ry1 = int(loc["y"][1] - win["y"][0])
    rx0 = int(loc["x"][0] - win["x"][0]); rx1 = int(loc["x"][1] - win["x"][0])
    Hraw, Wraw = raw.shape[1], raw.shape[2]
    ry0, ry1 = max(0, ry0), min(Hraw, ry1); rx0, rx1 = max(0, rx0), min(Wraw, rx1)
    if ry1 - ry0 < 2 or rx1 - rx0 < 2:
        return None, "oo_empty_bbox"

    picks = pick_planes(planes, loc["z"][0], loc["z"][1], orig_res[0])
    if not picks:
        return None, "oo_no_planes_in_z"
    bh, bw = ry1 - ry0, rx1 - rx0
    if native:
        scale, capped = 1.0, False                            # native: raw x-res, no resample
    else:
        scale = orig_res[2] / target                          # raw x-res -> target
        scale, capped = capped_scale(bh, bw, scale)
    out = []
    Zs = seg.shape[0]
    for i, zorig in picks:
        seg_z = int(round((zorig - loc["z"][0]) * orig_res[0] / seg_res[0]))
        seg_z = min(max(seg_z, 0), Zs - 1)
        mask_native = (seg[seg_z] != 0).astype(np.uint8)
        mask_bbox = cv2.resize(mask_native, (bw, bh), interpolation=cv2.INTER_NEAREST)
        em = ensure_u8_em(raw[i][ry0:ry1, rx0:rx1])
        em_out = resize_scale(em, scale, cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        lab_out = resize_scale(mask_bbox, scale, cv2.INTER_NEAREST)   # 0/1 fully GT
        out.append((em_out, lab_out, {"kind": "openorganelle", "plane": zorig, "capped": capped,
                                      "achieved_nm": round(orig_res[2] / scale, 3),
                                      "scale": round(scale, 5)}))
    return out, None

# ------------------------------------------------------------------ parallel worker
_MAN = None   # (dataset, crop_id) -> manifest crop record  (standard datasets)
_META = None  # (dataset, crop_id) -> crops_metadata row

def _init_worker():
    """Build the read-only indices once per worker process (spawn-safe)."""
    global _MAN, _META
    _MAN = {}
    for mf in glob.glob(os.path.join(ROOT, "*", "manifest.json")):
        folder = os.path.basename(os.path.dirname(mf))
        if folder == "openOrganelle":
            continue
        try:
            m = json.load(open(mf, encoding="utf-8"))
        except Exception:
            continue
        for c in m.get("crops", []):
            _MAN[(folder, c.get("crop_id"))] = c
    _META = {(r["dataset"], r["crop_id"]): r
             for r in csv.DictReader(open(os.path.join(ROOT, "crops_metadata.csv")))}


def _work(task):
    """Process one split-row: write its output image(s), return (manifest_rows, err_key)."""
    org, r, target, outdir, dry, native = task
    ds, cid, split = r["dataset"], r["crop_id"], r["split"]
    m = _META.get((ds, cid), {})
    try:
        src = float(m.get("voxel_x_nm") or 0)
    except ValueError:
        src = 0.0
    vest = (m.get("voxel_estimated") == "True")
    is_oo = (r["collection"] == "openOrganelle") or ("/" in cid)
    try:
        if is_oo:
            res, err = process_openorganelle(cid, org, target, native)
        else:
            crop_rec = _MAN.get((ds, cid))
            if crop_rec is None:
                res, err = None, "no_manifest_rec"
            elif src <= 0:
                res, err = None, "no_voxel"
            else:
                crop_rec["_src_voxel"] = src
                res, err = process_standard(ds, cid, crop_rec, org, target, native)
    except Exception as e:
        return [], "exc:%s" % type(e).__name__
    if err:
        return [], err
    base = cid.replace("/", "__") if "/" in cid else "%s__%s" % (ds, cid)
    out_rows, small = [], 0
    for (em_out, lab_out, info) in res:
        if min(lab_out.shape[:2]) < 16:
            small += 1
            continue
        name = base + ("__z%s" % info["plane"] if info["plane"] != "" else "")
        fg_px = int((lab_out == 1).sum()); ig_px = int((lab_out == 255).sum())
        if not dry:
            tifffile.imwrite(os.path.join(outdir, split, name + "_em.tif"), em_out)
            tifffile.imwrite(os.path.join(outdir, split, name + "_label.tif"), lab_out)
        out_rows.append({
            "organelle": org, "split": split, "dataset": ds, "crop_id": cid,
            "name": name, "kind": info["kind"], "plane": info["plane"],
            "source_image": (m.get("image_path") or ""),
            "src_voxel_nm": src, "voxel_estimated": vest,
            "target_nm": ("native" if native else target),
            "achieved_nm": info["achieved_nm"], "capped": info["capped"],
            "scale": info["scale"], "out_h": lab_out.shape[0], "out_w": lab_out.shape[1],
            "fg_px": fg_px, "ignore_px": ig_px,
            "coverage_tier": (m.get("coverage_tier") or ""),
            "modality": (m.get("modality") or ""), "scale_band": (m.get("scale_band") or ""),
            "tissue_context": (m.get("tissue_context") or ""),
            "species_group": (m.get("species_group") or ""),
            "subgroup": (r.get("subgroup") or ""), "split_role": (r.get("split_role") or ""),
        })
    return out_rows, ("too_small" if small and not out_rows else None)


# ------------------------------------------------------------------ stage: build
def build_org(org, out_root, args, pool):
    target = TARGET_NM[org]
    rows = list(csv.DictReader(open(os.path.join(ROOT, "splits", "final_%s.csv" % org))))
    if args.datasets:
        keep = set(args.datasets.split(","))
        rows = [r for r in rows if r["dataset"] in keep]
    if args.limit:
        rows = rows[:args.limit]
    outdir = os.path.join(out_root, org)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(outdir, sp), exist_ok=True)
    tasks = [(org, r, target, outdir, args.dry_run, args.native) for r in rows]
    man_rows, errors, n_img = [], collections.Counter(), collections.Counter()
    done = 0
    for out_rows, err in pool.imap_unordered(_work, tasks, chunksize=4):
        done += 1
        if err:
            errors[err] += 1
        for mr in out_rows:
            man_rows.append(mr); n_img[mr["split"]] += 1
        if args.verbose and done % 300 == 0:
            print("  [%s] %d/%d rows, %d images" % (org, done, len(rows), sum(n_img.values())), flush=True)
    if not args.dry_run and man_rows:
        cols = list(man_rows[0].keys())
        with open(os.path.join(out_root, "manifest_%s.csv" % org), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(man_rows)
    print("\n=== %s ===" % org.upper())
    print("  split rows: %d  ->  output images: %s (total %d)" % (
        len(rows), dict(n_img), sum(n_img.values())))
    print("  crops with no output (by reason): %s" % (dict(errors) or "none"))
    print("  capped(>%dpx): %d  estimated-res images: %d" % (
        MAX_DIM, sum(1 for x in man_rows if x["capped"]),
        sum(1 for x in man_rows if x["voxel_estimated"])))
    if man_rows:
        ach = sorted(x["achieved_nm"] for x in man_rows)
        dims = [max(x["out_h"], x["out_w"]) for x in man_rows]
        print("  achieved_nm: min=%.3f median=%.3f max=%.3f | max output dim=%d px" % (
            ach[0], ach[len(ach) // 2], ach[-1], max(dims)))
    return man_rows, errors, n_img


def cmd_build(args):
    global ROOT
    if args.corpus_root:
        os.environ["SEG_CORPUS_ROOT"] = args.corpus_root   # inherited by spawned workers
        ROOT = args.corpus_root
    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
    nproc = args.procs or max(1, min(20, (os.cpu_count() or 4) - 2))
    print("building%s with %d worker processes -> %s" % (
        " [NATIVE resolution]" if args.native else "", nproc, args.out), flush=True)
    orgs = ["er", "mito"] if args.org == "both" else [args.org]
    with Pool(processes=nproc, initializer=_init_worker) as pool:
        for org in orgs:
            build_org(org, args.out, args, pool)


# ------------------------------------------------------------------ stage: regrid
TILE = 512
MIN_VALID = 0.5
MIN_FG_PX = 256
CAP_OTHER = 1000
CAP_ME2 = 600
PLANT = "orgsegnet_plant"
PLANT_SHARE = 0.15
SEED = 1234

ME2_RE = re.compile(r"^(ME2-[A-Za-z0-9]+)")


def source_key(row):
    """MitoEM -> per ME2-* sub-source; everything else -> its dataset."""
    if row["dataset"] == "zenodo_mitoem2":
        m = ME2_RE.match(row["crop_id"])
        return "zenodo_mitoem2/" + (m.group(1) if m else "UNK")
    return row["dataset"]


def cap_for(key):
    return CAP_ME2 if key.startswith("zenodo_mitoem2/") else CAP_OTHER


def floor_starts(dim):
    return [i * TILE for i in range(dim // TILE)]


def compute_supply(rows, tile_dir):
    per_src = {}
    for r in rows:
        if r["split"] != "train":
            continue
        h, w = int(r["out_h"]), int(r["out_w"])
        if (h // TILE) * (w // TILE) < 2:
            continue
        lab_path = os.path.join(tile_dir, r["name"] + "_label.tif")
        if not os.path.exists(lab_path):
            continue
        lab = tifffile.imread(lab_path)
        if lab.ndim != 2:
            lab = np.squeeze(lab)
        H, W = lab.shape
        key = source_key(r)
        for cy in floor_starts(H):
            for cx in floor_starts(W):
                cell = lab[cy:cy + TILE, cx:cx + TILE]
                ig = int((cell == 255).sum())
                valid = TILE * TILE - ig
                fg = int(((cell != 0) & (cell != 255)).sum())
                if valid < MIN_VALID * TILE * TILE or fg < MIN_FG_PX:
                    continue
                per_src.setdefault(key, []).append(
                    {"parent": r, "cy": cy, "cx": cx, "fg": fg, "ignore": ig})
    return per_src


def cmd_regrid(args):
    src_manifest = os.path.join(args.out, "manifest_mito.csv")
    tile_dir = os.path.join(args.out, "mito", "train")
    out_dir = os.path.join(args.out, "mito_regrid", "train")
    out_manifest = os.path.join(args.out, "manifest_mito_regrid.csv")

    with open(src_manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0].keys())

    # per-source-key counts by split
    cur_train, cur_total = {}, {}
    val_total = test_total = 0
    for r in rows:
        k = source_key(r)
        cur_total[k] = cur_total.get(k, 0) + 1
        if r["split"] == "train":
            cur_train[k] = cur_train.get(k, 0) + 1
        elif r["split"] == "val":
            val_total += 1
        elif r["split"] == "test":
            test_total += 1

    supply = compute_supply(rows, tile_dir)
    S = {k: len(v) for k, v in supply.items()}

    # allocate non-plant: added = min(supply, cap - cur_train)
    add = {}
    for k in set(list(cur_train) + list(S)):
        if k == PLANT:
            continue
        ct = cur_train.get(k, 0)
        add[k] = max(0, min(S.get(k, 0), cap_for(k) - ct))

    # plant = 15% of TOTAL (all splits): solve X = plant train final
    NB = sum(cur_train.get(k, 0) + add.get(k, 0) for k in set(list(cur_train) + list(add)) if k != PLANT)
    VT = val_total + test_total
    plant_valtest = cur_total.get(PLANT, 0) - cur_train.get(PLANT, 0)
    X = (PLANT_SHARE * (NB + VT) - plant_valtest) / (1 - PLANT_SHARE)
    plant_add = int(round(X)) - cur_train.get(PLANT, 0)
    plant_add = max(0, min(plant_add, S.get(PLANT, 0)))
    add[PLANT] = plant_add

    # totals
    new_train = sum(cur_train.get(k, 0) + add.get(k, 0) for k in set(list(cur_train) + list(add)))
    grand = new_train + val_total + test_total
    added_total = sum(add.values())

    print(f"current: train {sum(cur_train.values())} | val {val_total} | test {test_total} "
          f"| total {len(rows)}")
    print(f"added (train only): {added_total}")
    print(f"NEW: train {new_train} | val {val_total} | test {test_total} | total {grand}\n")
    print(f"{'source':30s} {'cap':>5s} {'curTr':>6s} {'supp':>6s} {'add':>5s} {'finTr':>6s} {'tot%':>6s}")
    def keyf(k):
        return -(cur_total.get(k, 0) + add.get(k, 0))
    for k in sorted(set(list(cur_total) + list(add)), key=keyf):
        fin_tr = cur_train.get(k, 0) + add.get(k, 0)
        fin_tot = cur_total.get(k, 0) + add.get(k, 0)
        capv = "15%" if k == PLANT else str(cap_for(k))
        share = 100 * fin_tot / grand
        note = " HELDOUT" if (k == "zenodo_mitoem2/ME2-Mossy") else ""
        print(f"{k:30s} {capv:>5s} {cur_train.get(k,0):>6d} {S.get(k,0):>6d} "
              f"{add.get(k,0):>5d} {fin_tr:>6d} {share:>5.1f}%{note}")

    plant_tot = cur_total.get(PLANT, 0) + add.get(PLANT, 0)
    print(f"\nplant final: {plant_tot} = {100*plant_tot/grand:.1f}% of total (target 15%)"
          f" | {100*(cur_train.get(PLANT,0)+add.get(PLANT,0))/new_train:.1f}% of train")
    maxk = max(set(list(cur_total)+list(add)), key=lambda k: cur_total.get(k,0)+add.get(k,0))
    print(f"largest source: {maxk} = {100*(cur_total.get(maxk,0)+add.get(maxk,0))/grand:.1f}% of total")
    # cap compliance
    viol = [(k, cur_train.get(k,0)+add.get(k,0)) for k in add
            if k != PLANT and cur_train.get(k,0)+add.get(k,0) > cap_for(k)]
    print("cap violations:", viol if viol else "none")

    if not args.write:
        print("\n[dry-run] no tiles written.")
        return

    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(SEED)
    by_parent = {}
    for k, cells in supply.items():
        n = add.get(k, 0)
        if n <= 0:
            continue
        chosen = cells if n >= len(cells) else rng.sample(cells, n)
        for c in chosen:
            by_parent.setdefault(c["parent"]["name"], []).append(c)

    regrid_rows = []
    for name, cells in by_parent.items():
        em = tifffile.imread(os.path.join(tile_dir, name + "_em.tif"))
        lab = tifffile.imread(os.path.join(tile_dir, name + "_label.tif"))
        if em.ndim != 2: em = np.squeeze(em)
        if lab.ndim != 2: lab = np.squeeze(lab)
        for c in cells:
            p = c["parent"]; cy, cx = c["cy"], c["cx"]
            cid = f"{name}__r{cy}_{cx}"
            tifffile.imwrite(os.path.join(out_dir, cid + "_em.tif"), em[cy:cy+TILE, cx:cx+TILE])
            tifffile.imwrite(os.path.join(out_dir, cid + "_label.tif"), lab[cy:cy+TILE, cx:cx+TILE])
            nr = dict(p)
            nr["crop_id"] = p["crop_id"] + f"_r{cy}_{cx}"
            nr["name"] = cid
            nr["kind"] = "regrid_cell"
            nr["source_image"] = f"mito/train/{name}_label.tif"
            nr["out_h"] = TILE; nr["out_w"] = TILE
            nr["fg_px"] = c["fg"]; nr["ignore_px"] = c["ignore"]
            regrid_rows.append(nr)

    # full new dataset manifest = all original rows + regrid rows
    with open(out_manifest, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=header)
        wcsv.writeheader()
        wcsv.writerows(rows)
        wcsv.writerows(regrid_rows)
    print(f"\nWROTE {len(regrid_rows)} regrid tiles -> {out_dir}")
    print(f"WROTE full dataset manifest ({len(rows)+len(regrid_rows)} rows) -> {out_manifest}")


# ------------------------------------------------------------------ stage: add-cem
EXCLUDE = ("wei2020_mitoem", "kasthuri", "guay", "openorganelle")
FLAG = ("bleck", "cremi", "perez")

def is_clean(d):
    l = d.lower()
    if l.startswith("jrc_"):
        return False
    if any(k in l for k in EXCLUDE):
        return False
    if any(k in l for k in FLAG):
        return False
    return True

def tier(fg, area):
    f = fg / max(area, 1)
    return "sparse" if f < 0.02 else ("partial" if f < 0.1 else "full")


def cmd_add_cem(args):
    import pandas as pd

    regrid_manifest = os.path.join(args.out, "manifest_mito_regrid.csv")
    out_dir = os.path.join(args.out, "mito_cem_clean", "train")
    out_manifest = os.path.join(args.out, "manifest_mito_regrid_cem.csv")

    # metadata enrichment: Sample UID -> (organism, imaging, tissue)
    md = pd.read_excel(args.cem_xlsx, sheet_name="MetaData")
    meta = {}
    for _, r in md.iterrows():
        uid = str(r.get("Sample UID", "")).strip()
        if uid and uid != "nan":
            meta[uid] = (
                str(r.get("Organism", "") or "").strip(),
                str(r.get("Imaging Method", "") or "").strip(),
                str(r.get("Coarse Anatomical Origin", "") or r.get("Cell Type", "") or "").strip(),
            )

    z = zipfile.ZipFile(args.cem_zip)
    imgs = [n for n in z.namelist() if "/images/" in n and n.lower().endswith((".tif", ".tiff"))]
    clean_imgs = [n for n in imgs if is_clean(n.split("/")[1])]
    print(f"total image tiles: {len(imgs)} | clean: {len(clean_imgs)}")

    # header from the regrid manifest
    with open(regrid_manifest, newline="") as f:
        header = next(csv.reader(f))
    assert header[2] == "dataset" and header[23] == "subgroup", header

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    n_written = 0
    n_nomask = 0
    for i, imgname in enumerate(clean_imgs):
        d = imgname.split("/")[1]
        maskname = imgname.replace("/images/", "/masks/")
        try:
            img = tifffile.imread(io.BytesIO(z.read(imgname)))
        except Exception:
            continue
        try:
            m = tifffile.imread(io.BytesIO(z.read(maskname)))
        except KeyError:
            n_nomask += 1
            continue
        except Exception:
            n_nomask += 1
            continue
        if img.ndim != 2:
            img = np.squeeze(img)
        if m.ndim != 2:
            m = np.squeeze(m)
        H, W = img.shape[:2]
        lab = (m > 0).astype(np.uint8)  # instance -> semantic 0/1
        fg = int(lab.sum())
        org, imaging, tissue = meta.get(d, ("", "", ""))

        cid = f"cem_{i:05d}"
        name = f"cem_mitolab__{cid}"
        tifffile.imwrite(os.path.join(out_dir, name + "_em.tif"), img.astype(np.uint8, copy=False))
        tifffile.imwrite(os.path.join(out_dir, name + "_label.tif"), lab)
        n_written += 1

        row = {k: "" for k in header}
        row.update({
            "organelle": "mito", "split": "train", "dataset": "cem_mitolab",
            "crop_id": cid, "name": name, "kind": "cem_mitolab", "plane": "",
            "source_image": imgname,
            "out_h": H, "out_w": W, "fg_px": fg, "ignore_px": 0,
            "coverage_tier": tier(fg, H * W),
            "modality": imaging, "tissue_context": tissue, "species_group": org,
            "subgroup": d, "split_role": "train_cem_optional",
        })
        rows.append(row)
        if (i + 1) % 3000 == 0:
            print(f"  {i+1}/{len(clean_imgs)} processed, {n_written} written ...", flush=True)

    print(f"\nwritten: {n_written} tiles | tiles missing a mask (skipped): {n_nomask}")

    # combined manifest = regrid + cem rows
    with open(regrid_manifest, newline="") as f:
        orig = list(csv.DictReader(f))
    with open(out_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(orig)
        w.writerows(rows)
    print(f"WROTE {out_manifest}: {len(orig)} core + {len(rows)} cem = {len(orig)+len(rows)} rows")
    print(f"distinct clean CEM source volumes: {len(set(r['subgroup'] for r in rows))}")


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)

    b = sub.add_parser("build", help="final_* splits + corpus -> standardized tiles + manifests")
    b.add_argument("--out", required=True)
    b.add_argument("--corpus-root", default="")
    b.add_argument("--org", choices=["er", "mito", "ld", "nucleus", "both"], default="both")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--datasets", default="")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--verbose", action="store_true")
    b.add_argument("--procs", type=int, default=0)
    b.add_argument("--native", action="store_true",
                   help="keep each crop at its source resolution (scale=1, no resample, no cap); "
                        "same final_* splits/labels/ignore.")
    b.set_defaults(fn=cmd_build)

    g = sub.add_parser("regrid", help="balanced 512 px regrid of the mito train split")
    g.add_argument("--out", required=True)
    g.add_argument("--write", action="store_true", help="write tiles (default: dry-run report)")
    g.set_defaults(fn=cmd_regrid)

    c = sub.add_parser("add-cem", help="append the filtered CEM-MitoLab train pool")
    c.add_argument("--out", required=True)
    c.add_argument("--cem-zip", required=True, help="EMPIAR-11037 cem_mitolab.zip")
    c.add_argument("--cem-xlsx", required=True, help="EMPIAR-11037 cem_mitolab_metadata.xlsx")
    c.set_defaults(fn=cmd_add_cem)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
