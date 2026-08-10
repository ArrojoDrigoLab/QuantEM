"""EMPIAR-11746 — U2-OS FIB-SEM, MIB human segmentation (8 semantic classes).
Per-plane TIFFs: Images/U2OS_cell_NNNN.tif + Labels/Labels_U2OS_cell_NNNN.tif
(labels cover 0001-0874). z=5 nm -> sample every 80th plane => only fetch those.
Class map (documented, order per EMPIAR labelset; value 8 = unclassified, not organelle).
"""
import os, sys, urllib.request, shutil
import numpy as np
import tifffile
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/11746/data"
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "empiar_11746_u2os_fibsem")
WORK = os.path.join(OUT, "_work")
VOX = {"x": 2.5, "y": 2.5, "z": 5.0}
N_LABELED = 874
CLASS_MAP = {1: "nuclear_envelope", 2: "lipid_droplet", 3: "lysosome", 4: "mitochondria",
             5: "golgi", 6: "endoplasmic_reticulum", 7: "peroxisome", 8: "unclassified"}
ORG_VALUES = {1, 2, 3, 4, 5, 6, 7}  # all but 'unclassified'

META = {
    "name": "empiar_11746_u2os_fibsem",
    "source_repo": "EMPIAR", "accession": "EMPIAR-11746",
    "doi": "10.6019/EMPIAR-11746", "license": "CC0",
    "paper": "Czymmek/Belevich/Jokitalo, Nat Cell Biol 2024 (10.1038/s41556-024-01381-3)",
    "gt_provenance": "MIB human segmentation (manual brush+interpolation for lyso/LD/perox; DeepMIB-from-human-GT for NE/Golgi/ER/mito); separate human GT on Zenodo 10.5281/zenodo.10043461",
    "modality": "FIB-SEM (3D)", "dimensionality": "3D",
    "voxel_size_nm": VOX, "em_bit_depth": 8,
    "label_encoding": "semantic indexed 1-8",
    "organelle_classes": {str(k): v for k, v in CLASS_MAP.items() if k in ORG_VALUES},
    "non_organelle_classes": {"8": "unclassified"},
    "z_rule": "every 80th plane (5 nm -> >=400 nm)",
    "alignment": "1:1 (MIB labels on aligned EM stack); labels cover planes 0001-0874",
    "source_url": BASE,
    "notes": "Class index->name mapping is documented order, not header-verified; XY 3394x1385 -> centered/zero-padded. Sparse/partial coverage expected.",
}


def dl(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, path)
    return path


def g2(a):
    return a[..., 0] if (a.ndim == 3 and a.shape[-1] in (3, 4)) else a


def main():
    os.makedirs(WORK, exist_ok=True)
    ds = sc.Dataset(OUT, META)
    step = sc.zstep_for_spacing(VOX["z"])
    kept = list(range(1, N_LABELED + 1, step))
    print(f"z step {step} -> planes {kept}")
    for idx in kept:
        em_u = f"{BASE}/Images/U2OS_cell_{idx:04d}.tif"
        lb_u = f"{BASE}/Labels/Labels_U2OS_cell_{idx:04d}.tif"
        try:
            em = g2(tifffile.imread(dl(em_u, os.path.join(WORK, f"em_{idx:04d}.tif"))))
            lb = g2(tifffile.imread(dl(lb_u, os.path.join(WORK, f"lb_{idx:04d}.tif"))))
        except Exception as ex:
            print(f"  plane {idx}: {ex}"); continue
        if em.shape != lb.shape:
            print(f"  plane {idx} shape mismatch {em.shape} {lb.shape}"); continue
        Y, X = em.shape
        ds.add_plane(em, lb, source_image=f"Images/U2OS_cell_{idx:04d}.tif",
                     source_shape_xy=(X, Y), z_index=idx, z_physical_nm=float(idx * VOX["z"]),
                     voxel_size_nm=VOX, label_kind="semantic", class_map=CLASS_MAP,
                     organelle_values=ORG_VALUES, id_prefix="u2os_")
        os.remove(os.path.join(WORK, f"em_{idx:04d}.tif"))
        os.remove(os.path.join(WORK, f"lb_{idx:04d}.tif"))
    path, n = ds.write_manifest()
    print(f"DONE: {n} crops -> {path}")
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
