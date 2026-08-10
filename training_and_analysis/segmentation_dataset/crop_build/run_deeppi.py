"""DeepPI-EM skeletal-muscle TEM mito dataset -> GT crop collection.
Input layout: <corpus root>/_work/deeppi_em/{train,test}/{input/x_<n>.tif, target/y_<n>.png}.
EM = grayscale-as-RGB 2560x2560 uint8 (take ch0). Masks = binary mito (train 0/255 RGB;
test anti-aliased RGBA) -> binarize >=128 to semantic 1=mito. <4096 -> centered+zero-padded.
Preserves the authors' official train/test split (subdir + per-crop 'split')."""
import os, glob, sys
import numpy as np, tifffile
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop

WORK = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "_work", "deeppi_em")
OUT  = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "deeppi_em_skeletal_muscle")

meta = {
    "name": "deeppi_em_skeletal_muscle",
    "source_repo": "GitHub LAIT-CVLab/DeepPI-EM",
    "accession": "DeepPI-EM",
    "doi": "10.1038/s41598-025-03311-1",
    "license": "see repository (LAIT-CVLab/DeepPI-EM); article Sci Rep 2025 CC BY 4.0",
    "paper": "Deep learning-driven automated mitochondrial segmentation for analysis of complex TEM images, Sci Rep 2025",
    "gt_provenance": "manual mitochondria annotations (authors' DeepPI-EM dataset); mouse skeletal muscle TEM (WT & mdx Duchenne models)",
    "modality": "TEM (2D)",
    "dimensionality": "2D",
    "label_encoding": "semantic_binary (1 = mitochondria, 0 = background)",
    "organelle_classes": {"1": "mitochondria"},
    "voxel_size_nm": {"x": None, "y": None, "note": "pixel size not embedded in files; estimated value supplied by patch_estimated_res.py"},
    "alignment": "labels 1:1 pixel-registered to EM",
    "source_url": "https://drive.google.com/drive/folders/1LtnarR9R_zz0SPnEz5lSdYBQWNgYM1VT",
    "notes": "EM stored as 3ch grayscale (R==G==B) -> ch0. Train masks 0/255 RGB; test masks RGBA anti-aliased -> binarized at >=128. Authors' official train(21)/test(6) split preserved in crop['split'] and subdir. Intended as a held-out TEST source (skeletal muscle TEM, mito).",
}

ds = seg_crop.Dataset(OUT, meta, fresh=True)

def load_mask(p, shape_hw):
    im = Image.open(p)
    if im.size != (shape_hw[1], shape_hw[0]):   # (W,H); some masks are half-res
        im = im.resize((shape_hw[1], shape_hw[0]), Image.NEAREST)
    a = np.array(im)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)   # R==G==B; drop alpha
    return (a >= 128).astype(np.uint8)

for split in ("train", "test"):
    xs = sorted(glob.glob(os.path.join(WORK, split, "input", "*.tif")),
                key=lambda p: int(os.path.basename(p)[2:-4]))
    for x in xs:
        idx = os.path.basename(x)[2:-4]
        y = os.path.join(WORK, split, "target", f"y_{idx}.png")
        if not os.path.exists(y):
            print("  !! missing mask for", x); continue
        em = tifffile.imread(x)
        em = em[..., 0] if em.ndim == 3 else em
        lb = load_mask(y, em.shape)
        if lb.shape != em.shape:
            print("  !! shape mismatch", x, em.shape, lb.shape); continue
        start = len(ds.crops)
        ds.add_plane(em, lb, source_image=f"{split}/input/x_{idx}.tif",
                     source_shape_xy=(em.shape[1], em.shape[0]),
                     z_index=None, z_physical_nm=None, voxel_size_nm=None,
                     label_kind="semantic", class_map={1: "mitochondria"},
                     organelle_values={1}, subdir=split, id_prefix=f"{split[:2]}_")
        for c in ds.crops[start:]:
            c["split"] = split

path, n = ds.write_manifest()
print(f"wrote {n} crops -> {path}")
from collections import Counter
print("by split:", Counter(c["split"] for c in ds.crops))
