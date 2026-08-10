"""Guay et al. 2021 human-platelet SBF-SEM -> GT crop collection.
Input layout: <corpus root>/_work/guay/rgba/platelet-em/{images,labels-semantic}/<n>-*.tif.
Adds platelet mito + SECRETORY GRANULES (alpha, dense body, dense core) + canalicular.
images = uint16 (->8-bit full-range); semantic labels = RGBA color-coded -> class index.
SBF-SEM 10x10x40nm, human platelets. 800x800 <4096 -> centered+pad. Train(50)+eval(24) volumes.
License: US-gov (NIBIB) work, public-release (no formal license; treated as ~public domain)."""
import os, sys
import numpy as np, tifffile
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop

B = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "_work", "guay", "rgba", "platelet-em")
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "guay_platelet")
STRIDE = 2   # 2 * 40nm = 80nm (dense; granules are rare/valuable)
COLOR2IDX = {(0,40,255):1, (0,212,255):2, (124,255,121):3, (255,229,0):4, (255,70,0):5, (127,0,0):6}
CLASS_MAP = {1:"cell", 2:"mitochondria", 3:"alpha_granule", 4:"canalicular_vessel",
             5:"dense_granule", 6:"dense_granule_core"}
ORG = {2,3,4,5,6}   # cell(1) = non-organelle outline

meta = {
    "name": "guay_platelet",
    "source_repo": "leapmanlab/dense-cell (Guay et al. 2021)",
    "accession": "no DOI (Dropbox/GitHub, leapmanlab)",
    "doi": "no DOI",
    "publication_doi": "10.1038/s41598-021-81590-0",
    "license": "US-gov (NIBIB) work, cleared for public release; no formal license (treated as ~public domain)",
    "paper": "Guay et al., Dense cellular segmentation for EM using 2D-3D neural network ensembles, Sci Rep 2021",
    "gt_provenance": "manual dense segmentation (human platelets); semantic + instance labels",
    "modality": "SBF-SEM (3D)", "dimensionality": "3D",
    "voxel_size_nm": {"x": 10, "y": 10, "z": 40},
    "label_encoding": "semantic multi-class: 1=cell 2=mito 3=alpha_granule 4=canalicular 5=dense_granule 6=dense_granule_core",
    "organelle_classes": {"2":"mitochondria","3":"alpha_granule","4":"canalicular_vessel","5":"dense_granule","6":"dense_granule_core"},
    "tissue": "platelet (blood)", "species": "human",
    "alignment": "labels 1:1 voxel-registered to EM",
    "source_url": "https://leapmanlab.github.io/dense-cell/",
    "notes": "EM uint16->8-bit per-volume full-range. Semantic RGBA->class index (Cell blue / Mito cyan / Alpha green / Canalicular yellow / DenseBody red / DenseCore maroon). Sampled every 2 planes (80nm). Adds platelet mito + alpha/dense SECRETORY GRANULES (a secretory-granule type; not islet). Train(50)/eval(24) per Guay; test vol (121) not in this RGBA bundle.",
}

def to8(vol):
    lo, hi = float(vol.min()), float(vol.max())
    if hi <= lo: return np.zeros(vol.shape, np.uint8)
    return ((vol.astype(np.float32) - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)

def rgba_to_idx(lab):  # (H,W,4) -> (H,W) 0-6
    rgb = lab[..., :3]
    out = np.zeros(rgb.shape[:2], np.uint8)
    for col, idx in COLOR2IDX.items():
        out[(rgb == np.array(col, np.uint8)).all(-1)] = idx
    return out

ds = seg_crop.Dataset(OUT, meta, fresh=True)
for n, split in (("50", "train"), ("24", "eval")):
    img = to8(tifffile.imread(f"{B}/images/{n}-images.tif"))          # (Z,800,800)
    lab = tifffile.imread(f"{B}/labels-semantic/{n}-semantic.tif")     # (Z,800,800,4)
    Z = img.shape[0]
    print(f"{split}: {Z} slices img{img.shape} lab{lab.shape}")
    start = len(ds.crops)
    for z in range(0, Z, STRIDE):
        em = img[z]
        lb = rgba_to_idx(lab[z])
        if not np.isin(lb, list(ORG)).any():
            continue
        ds.add_plane(em, lb, source_image=f"{split}/{n}-images.tif[z={z}]",
                     source_shape_xy=(em.shape[1], em.shape[0]),
                     z_index=z, z_physical_nm=z*40,
                     voxel_size_nm={"x":10,"y":10,"z":40},
                     label_kind="semantic", class_map=CLASS_MAP, organelle_values=ORG,
                     subdir=split, id_prefix=f"g{split[0]}_")
    for c in ds.crops[start:]:
        c["split"] = split

path, k = ds.write_manifest()
from collections import Counter
print(f"wrote {k} crops -> {path}")
print("by split:", Counter(c["split"] for c in ds.crops))
fr = Counter()
for c in ds.crops:
    for kk in c["organelles_present"]: fr[kk] += 1
print("class crop counts:", dict(fr))
