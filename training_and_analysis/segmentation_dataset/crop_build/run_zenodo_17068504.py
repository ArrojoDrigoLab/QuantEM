"""Zenodo 17068504 — TEM cardiomyocyte masks (Med Uni Graz).
2D 16-bit TEM + per-class binary masks (U-Net output, visually inspected &
manually corrected; cell masks manual-threshold).  Organelles: mito, nucleus,
nucleolus (+ non-organelle myofibre, cell recorded too).

EM images: <base>.tif (record top level).  Masks: <Class>_Masks/<base>_<suf>.tif.
Naming mismatch: only 6 bases match cleanly; 1 EM has a stray _nucleus suffix
(stripped + logged); 4 mask sets are orphaned (no EM); 3 EM have no masks.
Composite label values: 1=mito 2=nucleus 3=nucleolus 4=myofibre 5=cell.
"""
import os, sys, json, urllib.request, shutil, zipfile, io
import numpy as np
import tifffile
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "zenodo_17068504_cardiomyocyte")
WORK = os.path.join(OUT, "_work")
REC = "https://zenodo.org/api/records/17068504"

CLASS_ZIP = {"Mito_Masks": ("_mito", 1, "mitochondria"),
             "Nucleus_Masks": ("_nucleus", 2, "nucleus"),
             "Nucleolus_Masks": ("_nucleolus", 3, "nucleolus"),
             "Myo_Masks": ("_myo", 4, "myofibre"),
             "Cell_Masks": ("_cell", 5, "cell")}
# paint order: non-organelle first so organelles overwrite on overlap
PAINT_ORDER = ["Cell_Masks", "Myo_Masks", "Mito_Masks", "Nucleus_Masks", "Nucleolus_Masks"]
ORG_VALUES = {1, 2, 3}
CLASS_MAP = {1: "mitochondria", 2: "nucleus", 3: "nucleolus", 4: "myofibre", 5: "cell"}
KNOWN_SUF = ["_nucleus", "_mito", "_nucleolus", "_myo", "_cell"]

META = {
    "name": "zenodo_17068504_cardiomyocyte",
    "source_repo": "Zenodo", "accession": "17068504",
    "doi": "10.5281/zenodo.17068504", "license": "CC-BY-4.0",
    "paper": "Cell Reports Methods 2025 (MitoMapper); Med Uni Graz",
    "gt_provenance": "U-Net masks visually inspected + manually corrected; cell masks manual-threshold; training ROIs hand-drawn",
    "modality": "TEM (2D)", "dimensionality": "2D", "em_bit_depth": 16,
    "label_encoding": "semantic multi-class: 1=mito 2=nucleus 3=nucleolus 4=myofibre 5=cell",
    "organelle_classes": {"1": "mitochondria", "2": "nucleus", "3": "nucleolus"},
    "non_organelle_classes": {"4": "myofibre", "5": "cell"},
    "voxel_size_nm": {"note": "~3500x; nm/px not in metadata (scale via TIFF tags/scale bar)"},
    "alignment": "1:1 (mask shares EM base name; images+masks resampled to 4096x4096)",
    "source_url": "https://zenodo.org/records/17068504",
    "notes": "",
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
    os.makedirs(WORK, exist_ok=True)
    d = json.load(urllib.request.urlopen(REC))
    em_keys = [f["key"] for f in d["files"] if f["key"].lower().endswith(".tif")]

    # fetch the five per-class mask zips
    zip_links = {f["key"]: f["links"]["self"] for f in d["files"]}
    for z in CLASS_ZIP:
        k = z + ".zip"
        download(zip_links.get(k, f"{REC}/files/{k}/content"), os.path.join(WORK, k))

    # index masks: base -> {zipname: bytes}
    zf = {z: zipfile.ZipFile(os.path.join(WORK, z + ".zip")) for z in CLASS_ZIP}
    mask_index = {}  # base -> {zip: member}
    for z, (suf, val, name) in CLASS_ZIP.items():
        for m in zf[z].namelist():
            if not m.lower().endswith(".tif"):
                continue
            b = os.path.splitext(os.path.basename(m))[0]
            if b.endswith(suf):
                b = b[: -len(suf)]
            mask_index.setdefault(b, {})[z] = m

    # map EM key -> base (with logged suffix-strip heuristic)
    notes = []
    em_to_base = {}
    for k in em_keys:
        b = os.path.splitext(k)[0]
        if b in mask_index:
            em_to_base[k] = b
            continue
        stripped = b
        for suf in KNOWN_SUF:
            if stripped.endswith(suf):
                cand = stripped[: -len(suf)]
                if cand in mask_index:
                    em_to_base[k] = cand
                    notes.append(f"paired EM '{k}' to mask base '{cand}' by stripping '{suf}'")
                    break
    matched = set(em_to_base.values())
    orphan_masks = sorted(set(mask_index) - matched)
    em_no_mask = sorted(os.path.splitext(k)[0] for k in em_keys if k not in em_to_base)
    notes.append(f"orphan mask sets (no EM, skipped): {orphan_masks}")
    notes.append(f"EM with no masks (skipped): {em_no_mask}")
    META["notes"] = " | ".join(notes)
    print("\n".join(notes))

    ds = sc.Dataset(OUT, META)
    for k, base in em_to_base.items():
        em = to_gray2d(tifffile.imread(download(f"{REC}/files/{k}/content", os.path.join(WORK, k))))
        Y, X = em.shape
        lbl = np.zeros((Y, X), dtype=np.uint8)
        present = {}
        for z in PAINT_ORDER:
            if z not in mask_index[base]:
                continue
            suf, val, name = CLASS_ZIP[z]
            with zf[z].open(mask_index[base][z]) as fh:
                mk = to_gray2d(tifffile.imread(io.BytesIO(fh.read())))
            if mk.shape != (Y, X):
                present[name] = f"SHAPE_MISMATCH {mk.shape} vs {(Y,X)}"
                continue
            lbl[mk > 0] = val
            present[name] = "ok"
        print(f"  {k}  em{em.shape} {em.dtype}  classes={present}")
        ds.add_plane(em, lbl, source_image=k, source_shape_xy=(X, Y),
                     z_index=None, z_physical_nm=None,
                     voxel_size_nm={"note": "~3500x, nm/px not in metadata"},
                     label_kind="semantic", class_map=CLASS_MAP,
                     organelle_values=ORG_VALUES, id_prefix="cm_")
    path, n = ds.write_manifest()
    print(f"DONE: {n} crops -> {path}")
    for z in zf.values():
        z.close()


if __name__ == "__main__":
    main()
