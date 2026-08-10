"""Zenodo 17635006 — MitoEM 2.0, 8 vEM volumes, mitochondria instance labels
(model-seeded + expert proofread in VAST/Neuroglancer + consensus review).
nnU-Net NIfTI: Dataset00X_*.zip -> imagesTr/<case>_0000.nii.gz + labelsTr/<case>.nii.gz.
EM float32 -> uint8 full-range; labels uint16 instance.
"""
import os, sys, json, argparse, urllib.request, shutil, zipfile, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc
from readers import read_nii_gz, to_uint8_fullrange

REC = "https://zenodo.org/api/records/17635006"
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "zenodo_mitoem2")
WORK = os.path.join(OUT, "_work")


def vox_for(name):
    n = name.lower()
    if any(k in n for k in ("mossy", "pyra", "stem")):
        return {"x": 8.0, "y": 8.0, "z": 30.0}
    return {"x": 16.0, "y": 16.0, "z": 16.0}


META = {
    "name": "zenodo_mitoem2",
    "source_repo": "Zenodo", "accession": "17635006",
    "doi": "10.5281/zenodo.17635006", "license": "CC-BY-4.0",
    "paper": "Liu & Peng, MitoEM 2.0, bioRxiv 2025.11.12.687478",
    "gt_provenance": "model-seeded + expert proofread (VAST/Neuroglancer) + multi-annotator consensus review",
    "modality": "volume-EM (FIB/ssSEM/SBF), 8 volumes", "dimensionality": "3D",
    "label_encoding": "instance (mitochondria, uint16; nonzero = unique id)",
    "organelle_classes": {"nonzero": "mitochondria"},
    "z_rule": "16 nm iso -> every 25th; 8x8x30 -> every 14th plane",
    "alignment": "1:1 (nnU-Net imagesTr/labelsTr aligned)",
    "source_url": "https://zenodo.org/records/17635006",
    "notes": "EM stored float32 -> converted to uint8 full-range. Most volumes <4096 XY (padded); ME2-Pyra is 4096^2.",
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


def proc_zip(ds, key, url):
    name = key[:-4]  # strip .zip
    vox = vox_for(name)
    z_nm = vox["z"]
    zp = dl(url, os.path.join(WORK, key))
    ext = os.path.join(WORK, name)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(ext)
    imgs = sorted(glob.glob(os.path.join(ext, "**", "imagesTr", "*.nii.gz"), recursive=True))
    step = sc.zstep_for_spacing(z_nm)
    for ip in imgs:
        case = os.path.basename(ip).replace("_0000.nii.gz", "")
        lps = glob.glob(os.path.join(os.path.dirname(os.path.dirname(ip)), "labelsTr", case + ".nii.gz"))
        if not lps:
            print(f"    no label for {case}"); continue
        em_arr, _, _ = read_nii_gz(ip)     # [X,Y,Z]
        lb_arr, _, _ = read_nii_gz(lps[0])
        if em_arr.shape != lb_arr.shape:
            print(f"    shape mismatch {case}: {em_arr.shape} {lb_arr.shape}"); continue
        X, Y, Z = em_arr.shape
        kept = list(range(0, Z, step))
        print(f"    {name}/{case}: XYZ={em_arr.shape} step{step} -> {len(kept)} planes")
        for zi in kept:
            em = to_uint8_fullrange(em_arr[:, :, zi].T)   # -> (Y,X)
            lb = lb_arr[:, :, zi].T.astype(lb_arr.dtype)
            ds.add_plane(em, lb, source_image=f"{name}/{case}", source_shape_xy=(X, Y),
                         z_index=int(zi), z_physical_nm=float(zi * z_nm), voxel_size_nm=vox,
                         label_kind="instance_single", organelle_name="mitochondria",
                         subdir=name, id_prefix=f"{name.split('_')[-1]}_")
    shutil.rmtree(ext, ignore_errors=True)
    os.remove(zp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")   # substring filter e.g. Stem
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    d = json.load(urllib.request.urlopen(REC))
    zips = [(f["key"], f["links"]["self"]) for f in d["files"] if f["key"].lower().endswith(".zip")]
    if a.only:
        zips = [(k, u) for k, u in zips if a.only.lower() in k.lower()]
    ds = sc.Dataset(OUT, META, fresh=not a.only)  # keep prior when running a single volume
    for k, u in sorted(zips):
        proc_zip(ds, k, u)
    path, n = ds.write_manifest()
    print(f"DONE: {n} crops -> {path}")
    if not a.only:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
