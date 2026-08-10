"""Zenodo 15602048 — TEM mitochondria masks, triple-negative breast cancer.
2D TEM micrographs + manual (QuPath hand-drawn) semantic mito masks.
slides: data/<DS>/<split>/slide_images/<name>.tif
masks : data/<DS>/<split>/mitochondria/masks/<name>.png  (binary -> 1=mito)
Skip the 'Mixture' dataset (remix of the other three -> dedupe).
"""
import os, sys, urllib.request, shutil, tarfile, glob
import numpy as np
import tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "zenodo_15602048_tem_breast_mito")
WORK = os.path.join(OUT, "_work")
EXT = os.path.join(WORK, "extract")
SLIDES = "https://zenodo.org/api/records/15602048/files/tem-seg-data_slide_images.tar.gz/content"
MASKS = "https://zenodo.org/api/records/15602048/files/tem-seg-data_mitochondria_masks.tar.gz/content"
SKIP_DATASETS = {"Mixture"}

META = {
    "name": "zenodo_15602048_tem_breast_mito",
    "source_repo": "Zenodo", "accession": "15602048",
    "doi": "10.5281/zenodo.15602048", "license": "CC-BY-4.0",
    "paper": "Arriojas et al., bioRxiv 2025.02.19.635300",
    "gt_provenance": "manual — 11,039 mitochondria hand-drawn in QuPath across 125 TEM micrographs",
    "modality": "TEM (2D)", "dimensionality": "2D",
    "label_encoding": "semantic_binary (1 = mitochondria, 0 = background)",
    "organelle_classes": {"1": "mitochondria"},
    "voxel_size_nm": {"note": "per-image 0.36-23 nm/px; scale bar burned into image, not clean TIFF tag"},
    "alignment": "1:1 pixel-registered (mask shares slide filename)",
    "source_url": "https://zenodo.org/records/15602048",
    "notes": "Mixture dataset skipped (remix of DRP1-KO/HCI-010/PIM001-P). Sub-4096 micrographs centered + zero-padded; larger ones grid-tiled.",
}


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, path)
    return path


def to_gray2d(a):
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., 0]
    return a


def main():
    os.makedirs(EXT, exist_ok=True)
    sl = download(SLIDES, os.path.join(WORK, "slides.tar.gz"))
    mk = os.path.join(WORK, "masks.tar.gz")
    if not os.path.exists(mk):
        download(MASKS, mk)
    print("extracting...")
    for t in (sl, mk):
        with tarfile.open(t) as tf:
            tf.extractall(EXT)

    ds = sc.Dataset(OUT, META)
    masks = sorted(glob.glob(os.path.join(EXT, "data", "*", "*", "mitochondria", "masks", "*.png")))
    n_pair, n_skip = 0, 0
    for mp in masks:
        parts = mp.replace("\\", "/").split("/")
        # .../data/<DS>/<split>/mitochondria/masks/<name>.png
        i = parts.index("data")
        dsname, split = parts[i + 1], parts[i + 2]
        name = os.path.splitext(parts[-1])[0]
        if dsname in SKIP_DATASETS:
            continue
        slide = os.path.join(EXT, "data", dsname, split, "slide_images", name + ".tif")
        if not os.path.exists(slide):
            n_skip += 1; continue
        em = to_gray2d(tifffile.imread(slide))
        mask = np.array(Image.open(mp))
        mask = to_gray2d(mask)
        lbl = (mask > 0).astype(np.uint8)  # binarize -> 1 = mito
        if em.shape != lbl.shape:
            print(f"  shape mismatch {dsname}/{name}: em{em.shape} lbl{lbl.shape}"); n_skip += 1; continue
        Y, X = em.shape
        ds.add_plane(em, lbl, source_image=f"data/{dsname}/{split}/slide_images/{name}.tif",
                     source_shape_xy=(X, Y), z_index=None, z_physical_nm=None,
                     voxel_size_nm={"note": "per-image 0.36-23 nm/px"},
                     label_kind="semantic", class_map={1: "mitochondria"},
                     organelle_values={1}, subdir=dsname, id_prefix=f"{dsname}_")
        n_pair += 1
    path, n = ds.write_manifest()
    print(f"paired {n_pair} slides ({n_skip} skipped) -> {n} crops -> {path}")
    # cleanup big artifacts
    shutil.rmtree(EXT, ignore_errors=True)
    for t in (sl, mk):
        try: os.remove(t)
        except OSError: pass


if __name__ == "__main__":
    main()
