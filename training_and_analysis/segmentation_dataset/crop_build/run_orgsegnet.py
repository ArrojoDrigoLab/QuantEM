"""OrgSegNet / Plantorgan Hunter (Nature Plants 2023, CC BY 4.0) -> GT crop collection.

Plant TEM, 19 species. Masks = single-channel index PNG: 0=bg, 1=chloroplast,
2=mitochondria, 3=vacuole, 4=nucleus. Preserves the authors' OFFICIAL split
(splits/{train,val,test}.txt = 541/180/181).

DATA (download required — SciDB is login-gated; fetch via browser/account):
  https://doi.org/10.11922/sciencedb.01335  ->  CellData/{image/*.tif, label/*.png, splits/*.txt}
Place it at ORG_ROOT below, then run this driver.

Optional resolution backfill: RES_MAP_JSON = {image_basename: x_nm} fills
voxel_size_nm per image when given."""
import os, sys, glob, json
import numpy as np, tifffile
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop

ORG_ROOT    = os.environ.get("ORG_ROOT", os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "_work", "orgsegnet", "CellData"))
OUT         = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "orgsegnet_plant")
RES_MAP_JSON= os.environ.get("ORG_RES_MAP", "")   # optional {basename: nm}

CLASS_MAP = {1: "chloroplast", 2: "mitochondria", 3: "vacuole", 4: "nucleus"}
ORG_VALUES = {1, 2, 3, 4}   # all four are organelles (mito/nucleus are the primary targets)
PALETTE = {(0,0,0):0,(128,0,0):1,(0,128,0):2,(128,128,0):3,(0,0,128):4}

meta = {
    "name": "orgsegnet_plant",
    "source_repo": "ScienceDB (Plantorgan Hunter) + GitHub yzy0102/OrgSegNet",
    "accession": "sciencedb.01335",
    "doi": "10.57760/sciencedb.01335",
    "publication_doi": "10.1038/s41477-023-01527-5",
    "license": "CC BY 4.0",
    "paper": "Feng et al., Plantorganelle Hunter ... Nature Plants 2023",
    "gt_provenance": "manual LabelMe annotations (chloroplast/mitochondria/nucleus/vacuole), 19 plant species",
    "modality": "TEM (2D)",
    "dimensionality": "2D",
    "label_encoding": "semantic multi-class: 1=chloroplast 2=mitochondria 3=vacuole 4=nucleus",
    "organelle_classes": {str(k): v for k, v in CLASS_MAP.items()},
    "alignment": "labels 1:1 pixel-registered to EM",
    "source_url": "https://doi.org/10.11922/sciencedb.01335",
    "notes": "Official split 541/180/181 preserved in crop['split']+subdir. Primary targets mito(2)+nucleus(4); chloroplast/vacuole are plant-specific bonus organelles. Resolution per-image (optional RES_MAP backfill; otherwise estimated by patch_estimated_res.py).",
}

def load_label(p):
    im = Image.open(p)
    if im.mode == "P":
        return np.array(im).astype(np.uint8)            # already class indices
    a = np.array(im.convert("RGB"))
    out = np.zeros(a.shape[:2], np.uint8)
    for rgb, idx in PALETTE.items():
        if idx == 0: continue
        out[(a == np.array(rgb)).all(-1)] = idx
    return out

def main():
    if not os.path.isdir(ORG_ROOT):
        print(f"!! ORG_ROOT not found: {ORG_ROOT}\n   Download sciencedb.01335 first (see header)."); return
    res_map = json.load(open(RES_MAP_JSON)) if RES_MAP_JSON and os.path.exists(RES_MAP_JSON) else {}
    split_of = {}
    for sp in ("train", "val", "test"):
        f = os.path.join(ORG_ROOT, "splits", f"{sp}.txt")
        if os.path.exists(f):
            for line in open(f):
                name = os.path.splitext(line.strip())[0]
                if name: split_of[name] = sp
    ds = seg_crop.Dataset(OUT, meta, fresh=True)
    imgs = sorted(glob.glob(os.path.join(ORG_ROOT, "image", "*")))
    for x in imgs:
        base = os.path.splitext(os.path.basename(x))[0]
        lp = os.path.join(ORG_ROOT, "label", base + ".png")
        if not os.path.exists(lp): continue
        em = tifffile.imread(x) if x.lower().endswith((".tif",".tiff")) else np.array(Image.open(x))
        if em.ndim == 3: em = em[..., 0]
        lb = load_label(lp)
        if lb.shape != em.shape:
            lb = np.array(Image.fromarray(lb).resize((em.shape[1], em.shape[0]), Image.NEAREST))
        sp = split_of.get(base, "unknown")
        nm = res_map.get(base) or res_map.get(os.path.basename(x))
        vox = {"x": nm, "y": nm} if nm else None
        start = len(ds.crops)
        ds.add_plane(em, lb, source_image=f"image/{os.path.basename(x)}",
                     source_shape_xy=(em.shape[1], em.shape[0]),
                     voxel_size_nm=vox, label_kind="semantic", class_map=CLASS_MAP,
                     organelle_values=ORG_VALUES, subdir=sp, id_prefix=f"{sp[:2]}_")
        for c in ds.crops[start:]:
            c["split"] = sp
    path, n = ds.write_manifest()
    from collections import Counter
    print(f"wrote {n} crops -> {path}")
    print("by split:", Counter(c.get("split") for c in ds.crops))

if __name__ == "__main__":
    main()
