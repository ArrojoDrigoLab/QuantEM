"""EMPIAR-10791 (Parlakgul/Arruda, Nature 2022) — MANUAL ER-family eval subset -> GT collection.
The dense full-volume ER is DL prediction (excluded); only the deposited MANUAL annotation eval
patches are used. Each eval = 25 small (150x150) slices + binary manual mask. Classes kept:
ER, ER sheets, ER tubules (the sheet/tubule sub-classes). Mouse liver FIB-SEM, 8 nm. <4096 -> centered+pad.
EM = OpenOrganelle jrc_mus-liver-4 (no COSEM GT), which is not in the openOrganelle set here; no overlap."""
import os, sys, glob, zipfile, io
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop

ROOT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "_work", "empiar10791", "manual", "Manual annotation and Prediction Slices")
OUT  = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "empiar_10791_liver_er_manual")

# class detection from raw.zip basename (order matters: sheets/tubules before bare ER)
def classify(b):
    bl = b.lower()
    if "er_sheets" in bl: return "er_sheets"
    if "er_tubules" in bl: return "er_tubules"
    if "_er evaluation" in bl: return "er"
    return None   # skip lipid/mito here (ER focus)

meta = {
    "name": "empiar_10791_liver_er_manual",
    "source_repo": "EMPIAR", "accession": "EMPIAR-10791",
    "doi": "10.6019/EMPIAR-10791", "publication_doi": "10.1038/s41586-022-04488-5",
    "license": "CC0",
    "paper": "Parlakgul & Arruda et al., Regulation of liver subcellular architecture, Nature 2022",
    "gt_provenance": "manual annotation eval subset (deposited 'Segmentation Evaluation'); dense full-volume ER is DL prediction and is excluded",
    "modality": "FIB-SEM (3D)", "dimensionality": "3D",
    "voxel_size_nm": {"x": 8, "y": 8, "z": 8},
    "label_encoding": "semantic per-class binary (er / er_sheets / er_tubules)",
    "organelle_classes": {"1": "(per-folder ER class)"},
    "tissue": "liver", "species": "mouse",
    "alignment": "manual mask 1:1 to raw eval slices",
    "source_url": "https://www.ebi.ac.uk/empiar/EMPIAR-10791/",
    "notes": "Manual ER-family eval patches only (150x150, centered+zero-padded to 4096). ER 12 evals + ER_Sheets 4 + ER_Tubules 4, 25 slices each, 4 liver datasets (1857/6461/6464/9430). EM=jrc_mus-liver-4 (not in the openOrganelle set here; no duplication). LD/mito manual patches present in source but skipped (ER focus).",
}

def members(z):
    return {os.path.basename(m): m for m in z.namelist()
            if m.lower().endswith((".tif", ".tiff"))
            and "__MACOSX" not in m and not os.path.basename(m).startswith("._")}

ds = seg_crop.Dataset(OUT, meta, fresh=True)
raws = sorted(glob.glob(os.path.join(ROOT, "**", "*raw.zip"), recursive=True))
for rz in raws:
    cls = classify(os.path.basename(rz))
    if cls is None:
        continue
    mz = rz.replace(" raw.zip", " manual annotation.zip")
    if not os.path.exists(mz):
        continue
    dataset = os.path.basename(os.path.dirname(rz)).split()[0]  # 1857/6461/6464/9430
    zr, zm = zipfile.ZipFile(rz), zipfile.ZipFile(mz)
    rmem, mmem = members(zr), members(zm)
    keys = sorted(set(rmem) & set(mmem))
    if not keys:  # fallback: pair by sorted order
        rk, mks = sorted(rmem), sorted(mmem)
        keys = rk
        mmem = {rk[i]: mmem[mks[i]] for i in range(min(len(rk), len(mks)))}
    start = len(ds.crops)
    for k in keys:
        em = np.array(Image.open(io.BytesIO(zr.read(rmem[k]))))
        mk = np.array(Image.open(io.BytesIO(zm.read(mmem[k]))))
        if em.ndim == 3: em = em[..., 0]
        if mk.ndim == 3: mk = mk[..., 0]
        lb = (mk > 0).astype(np.uint8)
        if lb.shape != em.shape or not lb.any():
            continue
        ds.add_plane(em, lb, source_image=f"{dataset}/{os.path.basename(rz)}/{k}",
                     source_shape_xy=(em.shape[1], em.shape[0]),
                     z_index=int("".join(filter(str.isdigit, k)) or 0),
                     voxel_size_nm={"x": 8, "y": 8, "z": 8},
                     label_kind="semantic", class_map={1: cls},
                     organelle_values={1}, subdir=f"{dataset}_{cls}", id_prefix=f"{cls[:3]}_")
    for c in ds.crops[start:]:
        c["er_class"] = cls; c["liver_dataset"] = dataset
    zr.close(); zm.close()

path, n = ds.write_manifest()
from collections import Counter
print(f"wrote {n} crops -> {path}")
print("by ER class:", Counter(c.get("er_class") for c in ds.crops))
