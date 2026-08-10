"""BioStudies S-BIAD2822 — nuclei instance segmentation (NucleoNet + expert proofread).
Files/id_N/ : img_<hash>.tif + lbl_<hash>.tif.  id_1..id_10 = 2D AT; id_11,id_12 = 3D FIB-SEM (18 nm iso).
Nuclei = instance labels.
"""
import os, sys, re, urllib.request, shutil
import numpy as np
import tifffile
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

BASE = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/822/S-BIAD2822/Files"
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "sbiad2822_nuclei")
WORK = os.path.join(OUT, "_work")
# id -> (xy_nm, is_3d)
IDVOX = {1: (5, False), 2: (5, False), 3: (5, False), 4: (5, False), 5: (5, False),
         6: (10, False), 7: (10, False), 8: (10, False), 9: (10, False), 10: (10, False),
         11: (18, True), 12: (18, True)}

META = {
    "name": "sbiad2822_nuclei",
    "source_repo": "BioStudies/BioImage Archive", "accession": "S-BIAD2822",
    "doi": "S-BIAD2822", "license": "CC-BY-4.0",
    "paper": "NucleoNet/DropNet, bioRxiv 2026.04.02.713930 (Narayan, FNL/NCI)",
    "gt_provenance": "NucleoNet predictions + expert manual proofreading ('fully proofread', confidence 5), empanada-napari",
    "modality": "AT (2D) + FIB-SEM (3D)",
    "label_encoding": "instance (nucleus; nonzero = unique id)",
    "organelle_classes": {"nonzero": "nucleus"},
    "z_rule": "3D (id_11/12) 18 nm iso -> every 23rd plane",
    "alignment": "1:1 (lbl shares img hash per id folder)",
    "source_url": BASE,
    "notes": "Edge-touching nuclei removed in id_1-7,9 (border crops may show unlabeled nuclei). 2D images large -> grid-tiled; 3D small -> centered/padded.",
}


def dl(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, path)
    return path


def listing(url):
    html = urllib.request.urlopen(url, timeout=60).read().decode()
    return re.findall(r'href="([^"?/][^"]*)"', html)


def g2(a):
    return a[..., 0] if (a.ndim == 3 and a.shape[-1] in (3, 4)) else a


def main():
    os.makedirs(WORK, exist_ok=True)
    ds = sc.Dataset(OUT, META)
    for i in range(1, 13):
        xy_nm, is3d = IDVOX[i]
        files = listing(f"{BASE}/id_{i}/")
        img = next((f for f in files if f.startswith("img_") and f.lower().endswith((".tif", ".tiff"))), None)
        lbl = next((f for f in files if f.startswith("lbl_") and f.lower().endswith((".tif", ".tiff"))), None)
        if not img or not lbl:
            print(f"  id_{i}: missing pair {files}"); continue
        em = tifffile.imread(dl(f"{BASE}/id_{i}/{img}", os.path.join(WORK, f"id{i}_{img}")))
        lb = tifffile.imread(dl(f"{BASE}/id_{i}/{lbl}", os.path.join(WORK, f"id{i}_{lbl}")))
        em, lb = g2(em), g2(lb)
        vox = {"x": xy_nm, "y": xy_nm, "z": xy_nm if is3d else None}
        if is3d and em.ndim == 3:
            step = sc.zstep_for_spacing(xy_nm)
            kept = list(range(0, em.shape[0], step))
            print(f"  id_{i} 3D {em.shape} step{step} -> {len(kept)} planes")
            for zi in kept:
                ds.add_plane(em[zi], lb[zi], source_image=f"id_{i}/{img}",
                             source_shape_xy=(em.shape[2], em.shape[1]), z_index=int(zi),
                             z_physical_nm=float(zi * xy_nm), voxel_size_nm=vox,
                             label_kind="instance_single", organelle_name="nucleus",
                             subdir=f"id_{i}", id_prefix=f"id{i}_")
        else:
            if em.ndim != 2:
                print(f"  id_{i} unexpected ndim {em.ndim}");
            print(f"  id_{i} 2D {em.shape}")
            ds.add_plane(em, lb, source_image=f"id_{i}/{img}",
                         source_shape_xy=(em.shape[1], em.shape[0]), z_index=None,
                         z_physical_nm=None, voxel_size_nm=vox,
                         label_kind="instance_single", organelle_name="nucleus",
                         subdir=f"id_{i}", id_prefix=f"id{i}_")
        for f in (f"id{i}_{img}", f"id{i}_{lbl}"):
            try: os.remove(os.path.join(WORK, f))
            except OSError: pass
    path, n = ds.write_manifest()
    print(f"DONE: {n} crops -> {path}")
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
