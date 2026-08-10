"""EMPIAR-10994 — HeLa SBF-SEM, MIB manual segmentation (Belevich/Jokitalo).
Per-slice TIF EM (02_processed_dataset) + per-slice TIF multi-class labels
(03_models/OrganelleModel_TIF) + readme.txt class map. Two cells (Control + KD).
"""
import os, sys, re, urllib.request, shutil
import numpy as np
import tifffile
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

B = "https://ftp.ebi.ac.uk/empiar/world_availability/10994/data"
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "empiar_10994_hela_sbfsem")
WORK = os.path.join(OUT, "_work")
CELLS = [("160621_HeLa_Control", (15.19, 15.19, 30.0)),
         ("161107_HeLa_REEP3-4_KD_LBR", (14.82, 14.82, 30.0))]

META = {
    "name": "empiar_10994_hela_sbfsem",
    "source_repo": "EMPIAR", "accession": "EMPIAR-10994",
    "doi": "10.6019/EMPIAR-10994", "license": "CC0",
    "paper": "Kumar et al., MBoC 2019 (PMC6724692) — REEP3/4 ER morphology",
    "gt_provenance": "manual MIB segmentation (depositors = MIB authors)",
    "modality": "SBF-SEM (3D)", "dimensionality": "3D",
    "label_encoding": "semantic multi-class (per-cell readme map)",
    "organelle_classes": "mito / ER (+subtypes) / chromosomes / centrioles / lipid droplets (per cell)",
    "z_rule": "z=30 nm -> every 14th plane",
    "alignment": "1:1 (labels on 02_processed_dataset grid; NOT raw .dm4)",
    "source_url": B,
    "notes": "Exterior(0)=background. XY ~1453-1600 -> centered/padded.",
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
        return re.findall(r'href="([^"?/][^"]*)"', urllib.request.urlopen(url, timeout=60).read().decode())
    except Exception:
        return []


def idx_of(fn):
    m = re.search(r'_(\d+)\.tiff?$', fn)
    return int(m.group(1)) if m else None


def parse_readme(txt):
    cmap = {}
    for ln in txt.splitlines():
        m = re.match(r'\s*(\d+)\s*->\s*(.+?)\s*$', ln)
        if m:
            cmap[int(m.group(1))] = m.group(2).strip().lower().replace(" ", "_")
    return cmap


def g2(a):
    return a[..., 0] if (a.ndim == 3 and a.shape[-1] in (3, 4)) else a


def main():
    os.makedirs(WORK, exist_ok=True)
    ds = sc.Dataset(OUT, META)
    for cell, vox in CELLS:
        emdir = f"{B}/{cell}/02_processed_dataset/"
        mdir = f"{B}/{cell}/03_models/"
        lbsub = next((d for d in listing(mdir) if re.search(r'_TIF/?$', d)
                      and not d.lower().startswith("celloutline")), None)
        lbdir = mdir + lbsub
        readme = urllib.request.urlopen(f"{mdir}readme.txt", timeout=60).read().decode(errors="ignore")
        cmap = parse_readme(readme)
        cmap.pop(0, None)  # drop Exterior/background
        org_vals = set(cmap)  # all labeled structures in-scope
        em_files = {idx_of(f): f for f in listing(emdir) if f.lower().endswith((".tif", ".tiff")) and idx_of(f)}
        lb_files = {idx_of(f): f for f in listing(lbdir) if f.lower().endswith((".tif", ".tiff")) and idx_of(f)}
        common = sorted(set(em_files) & set(lb_files))
        step = sc.zstep_for_spacing(vox[2])
        kept = common[::step]
        vx = {"x": vox[0], "y": vox[1], "z": vox[2]}
        print(f"  {cell}: map={cmap} em{len(em_files)} lb{len(lb_files)} common{len(common)} step{step} -> {len(kept)}")
        for i in kept:
            try:
                em = g2(tifffile.imread(dl(emdir + em_files[i], os.path.join(WORK, f"{cell}_em_{i}.tif"))))
                lb = g2(tifffile.imread(dl(lbdir + lb_files[i], os.path.join(WORK, f"{cell}_lb_{i}.tif"))))
            except Exception as ex:
                print(f"    idx {i}: {ex}"); continue
            if em.shape != lb.shape:
                print(f"    idx {i} shape {em.shape} vs {lb.shape}"); continue
            Y, X = em.shape
            ds.add_plane(em, lb, source_image=f"{cell}/02_processed_dataset/{em_files[i]}",
                         source_shape_xy=(X, Y), z_index=i, z_physical_nm=float(i * vox[2]),
                         voxel_size_nm=vx, label_kind="semantic", class_map=cmap,
                         organelle_values=org_vals, subdir=cell, id_prefix=f"{cell[:6]}_")
            for f in (f"{cell}_em_{i}.tif", f"{cell}_lb_{i}.tif"):
                try: os.remove(os.path.join(WORK, f))
                except OSError: pass
    path, n = ds.write_manifest()
    print(f"DONE: {n} crops -> {path}")
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
