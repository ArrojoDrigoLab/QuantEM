"""EMPIAR-13420 — macrophage/A431 SBF/FIB-SEM, MIB human segmentation.
Per-volume: dataset/TIFs/<vol>_NNNN.tif (EM) + labels/<Labels_*_TIFs>/...NNNN.tif (indexed labels).
Per-volume index->class map (semantic). Nuclear-pore landmark files skipped.
z-subsample at file level (fetch only kept slices).
"""
import os, sys, re, urllib.request, shutil
import numpy as np
import tifffile
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

B = "https://ftp.ebi.ac.uk/empiar/world_availability/13420/data"
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "empiar_13420_macrophage_a431")
WORK = os.path.join(OUT, "_work")

# (group, volume, voxel, {value:class})  all classes here are nucleus/LD/ER variants = organelles
VOLS = [
    ("FIB", "240729_HumanMP_TNFa_1U", (6, 6, 12), {1: "nucleus", 2: "lipid_droplet"}),
    ("FIB", "240801_HumanMP_Control", (6, 6, 12),
     {1: "mitochondria", 2: "golgi", 3: "endoplasmic_reticulum", 4: "nucleus", 5: "lipid_droplet", 6: "er_other_cells"}),
    ("SBFSEM", "231106_A431_CLEM_CM1_3J", (15, 15, 30),
     {1: "lipid_droplet_above_nucleus", 2: "lipid_droplet_below_nucleus", 3: "nucleus"}),
    ("SBFSEM", "240306_Microphages_OA-Chol_7T", (15, 15, 30),
     {1: "nucleus", 2: "lipid_droplet_touching_nucleus"}),
    ("SBFSEM", "240307_Microphages_OA-Chol_AV-1", (10, 10, 30),
     {1: "nucleus", 2: "lipid_droplet_touching_nucleus"}),
    ("SBFSEM", "240307_Microphages_OA-Chol_AV-3", (15, 15, 30),
     {1: "nucleus", 2: "lipid_droplet_touching_nucleus"}),
    ("SBFSEM", "240424_Microphages_no_load", (10, 10, 30),
     {1: "nuclei", 2: "lipid_droplet_below_nucleus", 3: "lipid_droplet_above_nucleus"}),
]

META = {
    "name": "empiar_13420_macrophage_a431",
    "source_repo": "EMPIAR", "accession": "EMPIAR-13420",
    "doi": "10.6019/EMPIAR-13420", "license": "CC0",
    "paper": "Szkalisity et al., EMBO J 2025 (10.1038/s44318-025-00423-2)",
    "gt_provenance": "MIB human segmentation (manual + semi-automatic graph-cut); nuclear-pore CNN human-corrected (landmarks, not extracted here)",
    "modality": "FIB-SEM + SBF-SEM (3D)", "dimensionality": "3D",
    "label_encoding": "semantic, per-volume index->class map (see crops/voxel & notes)",
    "organelle_classes": "nucleus / lipid-droplet / ER / mito / Golgi variants (all in-scope organelles)",
    "z_rule": "FIB 12 nm -> every 34th; SBF 30 nm -> every 14th plane",
    "alignment": "1:1 (MIB labels on EM grid); labels are sparse/NE-focal",
    "source_url": B,
    "notes": "Labels sparse (NE-proximal). Nuclear pores are landmark points (.ann/.landmarkAscii) - skipped. Per-volume class maps embedded.",
}


def dl(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, path)
    return path


def listing(url):
    try:
        html = urllib.request.urlopen(url, timeout=60).read().decode()
    except Exception:
        return []
    return re.findall(r'href="([^"?/][^"]*)"', html)


def g2(a):
    return a[..., 0] if (a.ndim == 3 and a.shape[-1] in (3, 4)) else a


def idx_of(fn):
    m = re.search(r'_(\d+)\.tiff?$', fn)
    return int(m.group(1)) if m else None


def main():
    os.makedirs(WORK, exist_ok=True)
    ds = sc.Dataset(OUT, META)
    for grp, vol, vox, cmap in VOLS:
        vd = f"{B}/{grp}/{vol}"
        em_files = {idx_of(f): f for f in listing(f"{vd}/dataset/TIFs/") if idx_of(f) is not None}
        labdir = next((d for d in listing(f"{vd}/labels/") if re.search(r'TIFs?/$', d)), None)
        if not labdir:
            print(f"  {vol}: no label TIFs dir"); continue
        lb_files = {idx_of(f): f for f in listing(f"{vd}/labels/{labdir}") if idx_of(f) is not None}
        common = sorted(set(em_files) & set(lb_files))
        if not common:
            print(f"  {vol}: no common indices (em {len(em_files)} lb {len(lb_files)})"); continue
        step = sc.zstep_for_spacing(vox[2])
        kept = common[::step]
        vx = {"x": vox[0], "y": vox[1], "z": vox[2]}
        orgvals = set(cmap)  # all classes are organelles here
        print(f"  {vol}: em{len(em_files)} lb{len(lb_files)} common{len(common)} step{step} -> {len(kept)} planes")
        for i in kept:
            em_u = f"{vd}/dataset/TIFs/{em_files[i]}"
            lb_u = f"{vd}/labels/{labdir}{lb_files[i]}"
            try:
                em = g2(tifffile.imread(dl(em_u, os.path.join(WORK, f"{vol}_em_{i}.tif"))))
                lb = g2(tifffile.imread(dl(lb_u, os.path.join(WORK, f"{vol}_lb_{i}.tif"))))
            except Exception as ex:
                print(f"    idx {i}: {ex}"); continue
            if em.shape != lb.shape:
                print(f"    idx {i} shape {em.shape} vs {lb.shape}"); continue
            Y, X = em.shape
            ds.add_plane(em, lb, source_image=f"{grp}/{vol}/dataset/TIFs/{em_files[i]}",
                         source_shape_xy=(X, Y), z_index=i, z_physical_nm=float(i * vox[2]),
                         voxel_size_nm=vx, label_kind="semantic", class_map=cmap,
                         organelle_values=orgvals, subdir=vol, id_prefix=f"{vol[:6]}_")
            for f in (f"{vol}_em_{i}.tif", f"{vol}_lb_{i}.tif"):
                try: os.remove(os.path.join(WORK, f))
                except OSError: pass
    path, n = ds.write_manifest()
    print(f"DONE: {n} crops -> {path}")
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
