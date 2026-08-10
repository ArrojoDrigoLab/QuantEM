"""EMPIAR-13156 (Alpy lab STARD3, EMBO J 2026) — HeLa FIB-SEM manual ER (+ endosomes, mito).
3 conditions (Ctrl/HeLaWT 8nm, STARD3_WT 5nm, FA_YA 8nm). EM = per-slice ROI tifs in <C>/em/;
masks = full-volume multipage ER.tif/Mitochondria.tif/Endosomes.tif (page i <-> slice i+1).
Input layout: <corpus root>/_work/empiar13156/<condition>/{em/*.tif, ER.tif, Mitochondria.tif, Endosomes.tif}.
Pair by slice NUMBER (robust to missing slices). Sample >=200nm. <4096 -> centered+pad. CC0.
Dedup: distinct HeLa STARD3 FIB-SEM; not in collection."""
import os, sys, glob, re
import numpy as np, tifffile
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop

WORK = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "_work", "empiar13156")
OUT  = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "empiar_13156_hela_stard3_er")
CONDS = [("HeLaWT_Raw", "Ctrl", 8), ("HeLaSTARD3_WT_Raw", "STARD3_WT", 5), ("HeLaSTARD3_FA_YA_Raw", "STARD3_FA_YA", 8)]
MASKS = [("ER.tif", 1, "er"), ("Endosomes.tif", 2, "endosome"), ("Mitochondria.tif", 3, "mitochondria")]
CLASS_MAP = {1: "er", 2: "endosome", 3: "mitochondria"}
ORG = {1, 2, 3}

meta = {
    "name": "empiar_13156_hela_stard3_er",
    "source_repo": "EMPIAR", "accession": "EMPIAR-13156",
    "doi": "10.6019/EMPIAR-13156", "publication_doi": "10.1038/s44318-026-00705-3",
    "license": "CC0",
    "paper": "Alpy lab, STARD3 ER-endosome contacts, EMBO J 2026",
    "gt_provenance": "manual/curated organelle masks (ER, endosomes, mitochondria) on HeLa FIB-SEM",
    "modality": "FIB-SEM (3D)", "dimensionality": "3D",
    "label_encoding": "semantic multi-class: 1=ER 2=endosome 3=mitochondria",
    "organelle_classes": {"1": "er", "2": "endosome", "3": "mitochondria"},
    "tissue": "cultured_cell", "species": "human",
    "alignment": "masks 1:1 to EM ROI slices (page i <-> slice i+1)",
    "source_url": "https://www.ebi.ac.uk/empiar/EMPIAR-13156/",
    "notes": "3 conditions (Ctrl 8nm, STARD3_WT 5nm, STARD3_FA_YA 8nm). Sampled >=200nm by slice number. ER + endosomes + mito; ER is the priority. 1188x2768-ish ROI centered+zero-padded.",
}

def num_of(fn):
    m = re.findall(r"(\d+)", os.path.basename(fn))
    return int(m[-1]) if m else -1

ds = seg_crop.Dataset(OUT, meta, fresh=True)
bad = []
for cdir, cond, xy in CONDS:
    base = os.path.join(WORK, cdir)
    ems = sorted(glob.glob(os.path.join(base, "em", "*.tif")), key=num_of)
    if not ems:
        print(f"  {cond}: no EM, skip"); continue
    mfiles = {m: tifffile.TiffFile(os.path.join(base, m[0])) for m in MASKS if os.path.exists(os.path.join(base, m[0]))}
    npages = len(next(iter(mfiles.values())).pages) if mfiles else 0
    stride = max(1, round(200 / xy))   # slices
    print(f"  {cond}: {len(ems)} EM, {npages} mask pages, xy={xy}nm, stride={stride}")
    start = len(ds.crops)
    for ef in ems:
        n = num_of(ef); pg = n - 1
        if pg < 0 or pg >= npages or (pg % stride) != 0:
            continue
        try:
            em = tifffile.imread(ef)
        except Exception:
            bad.append(ef); continue
        if em.ndim == 3: em = em[..., 0]
        # read mask pages; masks may be a top-left ROI crop of a larger full-frame EM
        mdata = {}
        for m in MASKS:
            if m not in mfiles: continue
            try:
                mk = mfiles[m].pages[pg].asarray()
            except Exception:
                continue
            if mk.ndim == 3: mk = mk[..., 0]
            mdata[m] = mk
        if not mdata:
            continue
        mh, mw = next(iter(mdata.values())).shape
        if em.shape[0] >= mh and em.shape[1] >= mw:
            em = em[:mh, :mw]            # crop EM to mask ROI (top-left)
        else:
            continue
        lb = np.zeros((mh, mw), np.uint8)
        for m, mk in mdata.items():
            if mk.shape == (mh, mw):
                lb[mk > 0] = m[1]
        if not lb.any():
            continue
        ds.add_plane(em, lb, source_image=f"{cond}/{os.path.basename(ef)}",
                     source_shape_xy=(em.shape[1], em.shape[0]),
                     z_index=n, z_physical_nm=n * xy,
                     voxel_size_nm={"x": xy, "y": xy, "z": xy},
                     label_kind="semantic", class_map=CLASS_MAP, organelle_values=ORG,
                     subdir=cond, id_prefix=f"{cond[:4].lower()}_")
    for c in ds.crops[start:]:
        c["condition"] = cond
    for mf in mfiles.values():
        mf.close()

path, n = ds.write_manifest()
from collections import Counter
print(f"wrote {n} crops -> {path}")
print("by condition:", Counter(c.get("condition") for c in ds.crops))
fr = Counter()
for c in ds.crops:
    for k in c["organelles_present"]: fr[k] += 1
print("organelle crop counts:", dict(fr))
print(f"unreadable EM slices skipped: {len(bad)}")
if bad:
    open(os.path.join(WORK, "_13156_bad.txt"), "w").write("\n".join(bad))
