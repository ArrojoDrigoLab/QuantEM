"""Finalize the OpenOrganelle crop dataset:
 1. validate every crop has its raw.tif + listed seg files (non-empty)
 2. derive the 'all' class-ID legend for ER/mito/LD empirically from the data
 3. write top-level manifest.json (meta + legend + summary + all crop entries)
 4. write README.md
 5. print a final tally
"""
import os, json, glob, numpy as np, tifffile

ROOT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "openOrganelle")
mans = sorted(glob.glob(os.path.join(ROOT, "*", "*", "crop_manifest.json")))

# ---------- 1. validate + aggregate ----------
crops = []
problems = []
per_ds = {}
total_bytes = 0
total_tiles = 0
seg_arrays = 0
tiles_xy = tiles_xz = 0
for mf in mans:
    m = json.load(open(mf))
    cdir = os.path.dirname(mf)
    ds = m["dataset"]
    per_ds.setdefault(ds, {"crops": 0, "tiles_xy": 0, "tiles_xz": 0, "bytes": 0})
    # raw tiles (both orientations)
    n_xy = n_xz = 0
    for key in ("raw_xy", "raw_xz"):
        e = m.get(key)
        if not e:
            problems.append(f"{m['crop_id']}: missing {key} entry")
            continue
        p = os.path.join(cdir, e["file"])
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            problems.append(f"{m['crop_id']}: missing/empty {e['file']}")
        n = len(e.get("plane_indices_emS0") or [])
        if key == "raw_xy": n_xy = n
        else: n_xz = n
    # segs
    for s in m["segmentations"]:
        sp = os.path.join(cdir, s["file"])
        if not (os.path.exists(sp) and os.path.getsize(sp) > 0):
            problems.append(f"{m['crop_id']}: missing/empty {s['file']}")
        seg_arrays += 1
    cbytes = sum(os.path.getsize(os.path.join(cdir, f)) for f in os.listdir(cdir))
    total_bytes += cbytes; total_tiles += n_xy + n_xz; tiles_xy += n_xy; tiles_xz += n_xz
    per_ds[ds]["crops"] += 1; per_ds[ds]["tiles_xy"] += n_xy; per_ds[ds]["tiles_xz"] += n_xz
    per_ds[ds]["bytes"] += cbytes
    m["_dir"] = os.path.relpath(cdir, ROOT)
    crops.append(m)

# ---------- 2. derive 'all' class-id legend (ER/mito/LD), voxel-weighted ----------
# accumulate per-organelle id voxel counts across sampled crops, then keep ids
# covering >=1% of that organelle's labeled voxels (drops boundary-bleed noise).
counts = {"mito": {}, "ld": {}, "er": {}}
sampled = 0
cand = sorted(crops, key=lambda c: os.path.getsize(os.path.join(ROOT, c["_dir"], "seg_all.tif"))
              if os.path.exists(os.path.join(ROOT, c["_dir"], "seg_all.tif")) else 1e18)
for c in cand:
    if sampled >= 20:
        break
    cdir = os.path.join(ROOT, c["_dir"])
    ap = os.path.join(cdir, "seg_all.tif")
    if not os.path.exists(ap):
        continue
    try:
        allm = tifffile.imread(ap)
    except Exception:
        continue
    got = False
    for org in ("mito", "ld", "er"):
        sp = os.path.join(cdir, f"seg_{org}.tif")
        if not os.path.exists(sp):
            continue
        seg = tifffile.imread(sp)
        if seg.shape != allm.shape:
            continue
        vals, cnts = np.unique(allm[seg > 0], return_counts=True)
        for v, n in zip(vals, cnts):
            if v == 0:
                continue
            counts[org][int(v)] = counts[org].get(int(v), 0) + int(n)
            got = True
    if got:
        sampled += 1

legend = {}
legend_detail = {}
for org, cmap in counts.items():
    tot = sum(cmap.values()) or 1
    kept = {i: round(n / tot, 4) for i, n in cmap.items() if n / tot >= 0.01}
    legend[org] = sorted(kept)
    legend_detail[org] = dict(sorted(kept.items(), key=lambda kv: -kv[1]))

# ---------- 3. top-level manifest.json ----------
manifest = {
    "meta": {
        "name": "OpenOrganelle ER/mito/LD crop dataset",
        "source": "OpenOrganelle / CellMap COSEM (s3://janelia-cosem-datasets)",
        "description": "Per-annotation-crop raw FIB-SEM tiles + ground-truth segmentations "
                       "for crops containing ER, mitochondria, or lipid droplets, across 18 datasets.",
        "generation_spec": {
            "scope": "crops whose label set contains any of ER / mito / LD",
            "raw_tiles": "EM s0 native resolution; per z-plane a window of min(4096,dim) per axis, "
                         "centered on the annotation (real EM context), clamped to volume bounds",
            "z_sampling": "z-planes >=200 nm apart (subsampled); see raw_crop.z_plane_indices_emS0",
            "segmentations": "Option A: 'all' (combined semantic class-id map) + 'mito' + 'ld' + 'er' "
                             "where present, at their own native s0 resolution, full crop (not z-subsampled)",
            "resolution_policy": "NO resampling: raw kept at EM s0 nm, segs kept at their native nm. "
                                 "raw and seg resolutions may differ (segs are often 2x finer).",
            "dtypes": "raw=native EM dtype (uint8); seg=as provided (mito/ld instance ids, er semantic 0/1, "
                      "all=class-id map). uint64 instance arrays downcast to uint32 losslessly.",
        },
        "all_map_class_ids_for_targets": {
            "note": "The combined 'all' map's class-id NUMBERING VARIES by dataset/annotation version "
                    "(COSEM has multiple ontology versions). Use the per-organelle seg_mito/seg_ld/seg_er "
                    "files as the authoritative masks for the targets; treat 'all' ids as dataset-specific. "
                    "Below: ids occupying >=1% of each organelle's voxels across sampled crops, with the "
                    "voxel fraction per id (shows the scheme spread, e.g. some datasets use whole-organelle "
                    "ids like mito=50, ld=44; older ones use membrane/lumen 3/4, 14/15).",
            "mito_ids": legend["mito"], "ld_ids": legend["ld"], "er_ids": legend["er"],
            "id_voxel_fractions": legend_detail,
            "legacy_membrane_lumen_scheme": {"mito_mem": 3, "mito_lum": 4, "mito_ribo": 5,
                                  "ld_mem": 14, "ld_lum": 15, "er_mem": 16, "er_lum": 17,
                                  "eres_mem": 18, "eres_lum": 19},
        },
        "alignment": "Each seg file gives physical_origin_nm_zyx + resolution_nm_zyx; the raw tile gives "
                     "its window_bbox_in_original_voxels_emS0 + resolution. Align via physical nm coords. "
                     "segmented_area.annotation_bbox_in_raw_crop_px places the labeled region within the tile.",
    },
    "summary": {
        "datasets": len(per_ds),
        "crops": len(crops),
        "raw_tiles_total": total_tiles,
        "raw_tiles_xy": tiles_xy,
        "raw_tiles_xz": tiles_xz,
        "seg_label_arrays_total": seg_arrays,
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1e9, 2),
        "per_dataset": {k: v for k, v in sorted(per_ds.items())},
        "validation_problems": problems,
    },
    "crops": crops,
}
json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w"), indent=1)

# ---------- 4. README ----------
readme = f"""# OpenOrganelle ER / mito / LD crop dataset

Raw FIB-SEM tiles + ground-truth segmentations for every OpenOrganelle / CellMap COSEM
annotation crop that contains **ER, mitochondria, or lipid droplets**, across 18 datasets.

## Contents
- **{len(crops)} crops** (folders) from **{len(per_ds)} source volumes**
- **{total_tiles} raw 4096x4096-capped EM tiles** = **{tiles_xy} X-Y** + **{tiles_xz} X-Z** (two orientations per crop)
- **{seg_arrays} segmentation label volumes** (`all` + `mito`/`ld`/`er` where present), shared by both orientations
- **{round(total_bytes/1e9,2)} GB** on disk

## Two tile orientations per crop
Every crop has BOTH:
- **`raw_xy.tif`** — X-Y planes (rows=Y, cols=X) sampled along **Z** at >=200 nm.
- **`raw_xz.tif`** — X-Z planes (rows=Z, cols=X) sampled along **Y** at >=200 nm.

This doubles usable 2-D data and is lossless because FIB-SEM is isotropic. Tile sizes differ by
orientation and dataset: a tile is full 4096x4096 only where both in-plane axes are >=4096 px;
otherwise it's clamped to the (real-EM) volume extent (no zero-padding). Each `raw_*` manifest
entry gives its own `shape_planes_rows_cols`, `window_bbox_in_original_voxels_emS0`,
`plane_indices_emS0`, `tile_axes_rows_cols`, and `annotation_bbox_in_tile_px`.

## Layout
```
<dataset>/<crop>/
  raw_xy.tif         # EM, native s0 res, X-Y planes (n_z, y, x), z-planes >=200nm apart
  raw_xz.tif         # EM, native s0 res, X-Z planes (n_y, z, x), y-planes >=200nm apart
                     # both: in-plane window = min(4096,dim), centered on annotation (real EM)
  seg_all.tif        # combined COSEM class-id map (semantic), crop native res
  seg_mito.tif       # mitochondria  (instance ids)   - if present
  seg_ld.tif         # lipid droplets (instance ids)  - if present
  seg_er.tif         # ER (semantic 0/1)              - if present
  crop_manifest.json # full provenance for this crop
manifest.json        # top-level: meta + legend + summary + all crop entries
```

## Resolution
Nothing was resampled. The **raw** is at the EM `s0` resolution (e.g. 4 or 8 nm); the
**segmentations** are at their own native resolution, which is **often 2x finer** than the
EM (COSEM stores labels on a refined grid). Each file's `resolution_nm_zyx` and
`physical_origin_nm_zyx` are in the manifest — align raw<->seg via physical nm coordinates.

## Segmentation semantics
- `mito`, `ld` = **instance** segmentation (each object has a unique id). **Use these for the targets.**
- `er` = **semantic** (0=absent, 1=present, 255=unknown). **Use this for ER.**
- `all` = combined **semantic class-id** map covering every annotated organelle; its id
  **numbering varies by dataset/annotation version** -- some volumes use whole-organelle ids
  (observed mito~50, ld~44), others a membrane/lumen split (mito_mem=3/lum=4, ld_mem=14/lum=15,
  er_mem=16/lum=17). Decoding `all` per-dataset is required for the other organelles; for
  ER/mito/LD just use the dedicated `seg_mito/seg_ld/seg_er` files. Per-id voxel fractions are in
  `manifest.json -> meta.all_map_class_ids_for_targets`.

## The 4096 window & "segmented area"
Each raw tile is up to 4096x4096 centered on the crop, padded with **real surrounding EM**
(clamped where the volume is smaller than 4096). Only a sub-region is actually annotated;
`segmented_area.annotation_bbox_in_raw_crop_px` in each crop_manifest gives that region.

## Provenance
Every crop_manifest records the source S3 URL, the original volume id/shape/resolution, and
the exact voxel bbox the tile was taken from. Source bucket: `s3://janelia-cosem-datasets`.
"""
open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8").write(readme)

# ---------- 5. tally ----------
print(f"crops={len(crops)} tiles={total_tiles} seg_arrays={seg_arrays} disk={total_bytes/1e9:.2f}GB")
print(f"legend (>=1% voxels): mito={legend['mito']} ld={legend['ld']} er={legend['er']}")
print(f"  detail: {legend_detail}")
print(f"validation problems: {len(problems)}")
for p in problems[:20]:
    print("  ", p)
print(f"  tiles: XY={tiles_xy} + XZ={tiles_xz} = {total_tiles}")
print("\nper dataset:")
for k, v in sorted(per_ds.items()):
    print(f"  {k:<28} crops={v['crops']:>3} XY={v['tiles_xy']:>4} XZ={v['tiles_xz']:>4} {v['bytes']/1e9:5.2f}GB")
print("\nwrote manifest.json + README.md")
