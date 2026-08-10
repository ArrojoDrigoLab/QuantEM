"""Zenodo 3675220 — Platynereis EM training data (manual/ariadne+Paintera proofread).
HDF5 blocks: volumes/raw (uint8) + volumes/labels/segmentation (instance).
Extract organelle structures: nuclei (nucleus) + cilia (cilium). Skip membrane/cuticle (non-organelle).
"""
import os, sys, urllib.request, shutil, zipfile, glob
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "zenodo_3675220_platynereis")
WORK = os.path.join(OUT, "_work")
REC = "https://zenodo.org/api/records/3675220/files"
# (zip, organelle, voxel xyz nm, label_key, ignore_value)
# nuclei: volumes/labels holds three datasets; the single-channel instance segmentation
#   'nucleus_instance_labels' (int32: id>0 = nucleus, 0 = background, -1 = unlabeled/ignore)
#   is the ground truth ('nucleus_binary_labels' is a (2,Z,Y,X) channel pair whose -1 ignore
#   channel would read un-annotated planes as foreground).
# cilia: single 'segmentation' dataset (uint64), no ignore value.
SRC = [("nuclei.zip", "nucleus", (80, 80, 100), "nucleus_instance_labels", -1),
       ("cilia.zip", "cilium", (20, 20, 25), None, None)]

META = {
    "name": "zenodo_3675220_platynereis",
    "source_repo": "Zenodo", "accession": "3675220",
    "doi": "10.5281/zenodo.3675220", "license": "CC-BY-4.0",
    "paper": "Vergara et al., Cell 2021 (PMC8445025) — Platynereis whole-body atlas",
    "gt_provenance": "CNN training data; nuclei ariadne.ai-seeded + proofread (NOT hand-drawn). "
                     "GT = volumes/labels/nucleus_instance_labels (instance IDs). Separate expert-"
                     "annotated set = Zenodo 3690727.",
    "modality": "SBEM (3D)", "dimensionality": "3D",
    "label_encoding": "instance (per structure; nonzero = unique id)",
    "organelle_classes": {"nuclei.zip": "nucleus", "cilia.zip": "cilium"},
    "z_rule": "nuclei z=100 nm -> every 4th; cilia z=25 nm -> every 16th",
    "alignment": "1:1 (raw+labels co-stored per HDF5 block)",
    "source_url": "https://zenodo.org/records/3675220",
    "notes": "membrane.zip/cuticle.zip skipped (cell membrane/cuticle are non-organelle). "
             "nuclei: from nucleus_instance_labels; each z-plane is cropped to its annotated (non -1) "
             "region and all-ignore planes are dropped, so every crop is fully labelled (no ignore-as-"
             "background). cilia: volumes/labels/segmentation. Per-block native XY <4096 -> padded.",
}


def dl(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, path)
    return path


def safe_label(a):
    if a.dtype == np.uint64 and int(a.max()) < 2**32:
        return a.astype(np.uint32)
    if a.dtype == np.int64:
        return a.astype(np.uint32)
    return a


def main():
    os.makedirs(WORK, exist_ok=True)
    ds = sc.Dataset(OUT, META)
    for zipname, organelle, vox, label_key, ignore_val in SRC:
        zp = dl(f"{REC}/{zipname}/content", os.path.join(WORK, zipname))
        ext = os.path.join(WORK, zipname[:-4])
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(ext)
        h5s = sorted(glob.glob(os.path.join(ext, "**", "*.h5"), recursive=True))
        step = sc.zstep_for_spacing(vox[2])
        vx = {"x": vox[0], "y": vox[1], "z": vox[2]}
        print(f"  {zipname}: {len(h5s)} blocks, step{step}, label_key={label_key}, ignore={ignore_val}")
        for hp in h5s:
            with h5py.File(hp, "r") as f:
                if "volumes/raw" not in f or "volumes/labels" not in f:
                    print(f"    {os.path.basename(hp)}: unexpected keys {list(f.keys())}"); continue
                raw = f["volumes/raw"][...]
                lgrp = f["volumes/labels"]
                if label_key and label_key in lgrp:
                    lk = label_key
                else:
                    lk = "segmentation" if "segmentation" in lgrp else list(lgrp.keys())[0]
                lab = lgrp[lk][...]
            if lab.ndim == raw.ndim + 1 and lab.shape[1:] == raw.shape:
                lab = lab[0]  # (2,Z,Y,X) fallback -> channel 0 (not used for the instance key)
            if raw.shape != lab.shape:
                print(f"    {os.path.basename(hp)}: shape {raw.shape} vs {lab.shape}"); continue
            Z = raw.shape[0]
            blk = os.path.splitext(os.path.basename(hp))[0]
            kept = skipped = 0
            for zi in range(0, Z, step):
                rp, lp = raw[zi], lab[zi]
                # Keep a plane only if it has a real organelle instance (id>0). This drops
                # the un-annotated / all-ignore planes.
                if not (lp > 0).any():
                    skipped += 1; continue
                ox, oy = 0, 0
                if ignore_val is not None:
                    # crop to the annotated (non-ignore) region so the crop is fully labelled
                    ys, xs = np.where(lp != ignore_val)
                    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                    rp = rp[y0:y1, x0:x1]
                    lp = lp[y0:y1, x0:x1].copy()
                    lp[lp < 0] = 0             # residual ignore -> background
                    ox, oy = x0, y0
                lp = safe_label(lp)
                n0 = ds._n
                ds.add_plane(rp, lp, source_image=f"{zipname}:{blk}",
                             source_shape_xy=(raw.shape[2], raw.shape[1]), z_index=int(zi),
                             z_physical_nm=float(zi * vox[2]), voxel_size_nm=vx,
                             label_kind="instance_single", organelle_name=organelle,
                             subdir=organelle, id_prefix=f"{organelle}_", origin_xy=(ox, oy))
                kept += (ds._n > n0)
            print(f"    {blk}: kept {kept}, skipped {skipped}")
        shutil.rmtree(ext, ignore_errors=True)
        # keep the source zip in _work for reproducibility
    path, n = ds.write_manifest()
    print(f"DONE: {n} crops -> {path}")
    # keep _work (source zips) for reproducibility


if __name__ == "__main__":
    main()
