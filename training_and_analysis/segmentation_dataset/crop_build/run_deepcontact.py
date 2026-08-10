"""DeepContact (Liu et al., JCB 2022) — manual 2D ER/mito/LD on TEM(10nm)+SEM(5nm).
figshare 10.6084/m9.figshare.19898404 (GPL-3.0).
Labelme polygon JSON + jpg EM -> semantic multi-class crops (1=mito 2=er 3=ld).

Usage: python run_deepcontact.py <subset>
  <subset> in {tem, sem, cell}. Reads _work/deepcontact/tem_all/tem, sem_all/sem,
  or cell_all/cell_all_labeled (*.jpg + *.json).
"""
import os, sys, json, glob, base64, io
import numpy as np, tifffile, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

ROOT = os.environ.get("SEG_CORPUS_ROOT", ".")
WORK = f"{ROOT}/_work/deepcontact"
CLASS_MAP = {1: "mito", 2: "er", 3: "ld", 4: "plasma_membrane"}   # 4 = structural membrane (kept; non-organelle)
ORG_VALUES = {1, 2, 3, 4}                                          # all kept/recorded; PM flagged structural in META
def map_label(lab):
    s = lab.strip().lower()
    if "mito" in s: return 1
    if s == "er" or "reticulum" in s: return 2
    if "lipid" in s or s == "ld": return 3
    if "plasma" in s or "membrane" in s: return 4
    return 0  # anything else -> ignore

SUBSET = sys.argv[1] if len(sys.argv) > 1 else "tem"
# Original acquisition modality+pixel size per subset, from Liu 2022 JCB Materials & Methods
# (PMC9361564). The "10 nm" often cited is the model-input normalization, not acquisition.
CFG = {
  "tem":  dict(src=f"{WORK}/tem_all/tem",   out=f"{ROOT}/deepcontact_tem",  vx=4.68,
               modality="TEM", tissue="cultured_cell"),          # FEI Tecnai Spirit, COS-7 (monkey)
  "sem":  dict(src=f"{WORK}/sem_all/sem",   out=f"{ROOT}/deepcontact_sem",  vx=10.0,
               modality="SEM", tissue="testis_seminiferous"),    # Helios 600i BSE, mouse Sertoli tissue
  "cell": dict(src=f"{WORK}/cell_all/cell_all_labeled", out=f"{ROOT}/deepcontact_cell", vx=5.0,
               modality="SEM", tissue="cultured_cell"),          # Helios 600i BSE, U-2 OS (human) — has LD
}[SUBSET]

META = {
  "name": f"deepcontact_{SUBSET}",
  "source_repo": "figshare (DeepContact Training Data)", "accession": "figshare 19898404",
  "doi": "10.6084/m9.figshare.19898404", "license": "GPL-3.0",
  "paper": "Liu et al., J Cell Biol 2022 (DeepContact)",
  "gt_provenance": "manual Labelme polygons (expert); full-image dense for the labeled classes",
  "modality": CFG["modality"], "dimensionality": "2D",
  "voxel_size_nm": {"x": CFG["vx"], "y": CFG["vx"], "z": None},
  "label_encoding": "semantic multi-class: 1=mitochondria 2=endoplasmic_reticulum 3=lipid_droplet 4=plasma_membrane",
  "organelle_classes": ["mitochondria", "endoplasmic_reticulum", "lipid_droplet"],
  "structural_classes": ["plasma_membrane"],  # value 4: membrane outline, not an organelle
  "completeness": "Labelme polygons are exhaustive for annotated classes per image -> background = true negative.",
}

def rasterize(shapes, H, W):
    lab = np.zeros((H, W), dtype=np.uint8)
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    # plasma_membrane(4) is a single whole-cell filled polygon, so it is drawn first and
    # the organelles are painted on top: PM (structural background), then ER, LD, and mito
    # last — an organelle always wins over the membrane fill, and mito wins overlaps.
    order = sorted(shapes, key=lambda s: {4: 0, 2: 1, 3: 2, 1: 3}.get(map_label(s["label"]), -1))
    for s in order:
        v = map_label(s["label"])
        if v == 0 or s.get("shape_type", "polygon") != "polygon":
            continue
        pts = np.asarray(s["points"], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(lab, [pts], int(v)); counts[v] += 1
    return lab, counts

jsons = sorted(glob.glob(os.path.join(CFG["src"], "*.json")))
print(f"DeepContact[{SUBSET}]: {len(jsons)} labeled images -> {CFG['out']}")
ds = sc.Dataset(CFG["out"], META, fresh=True)
tot = {1: 0, 2: 0, 3: 0, 4: 0}   # polygons drawn
fin = {1: 0, 2: 0, 3: 0, 4: 0}   # FINAL label pixels (after overlaps) — catches class overwrites
for jf in jsons:
    d = json.load(open(jf))
    W, H = d["imageWidth"], d["imageHeight"]
    jpg = os.path.join(CFG["src"], os.path.basename(d.get("imagePath", "")) or
                       os.path.basename(jf).replace(".json", ".jpg"))
    if os.path.exists(jpg):
        em = np.asarray(Image.open(jpg).convert("L"))
    elif d.get("imageData"):                                   # Labelme base64-embedded EM
        em = np.asarray(Image.open(io.BytesIO(base64.b64decode(d["imageData"]))).convert("L"))
    else:
        print("  no image for", os.path.basename(jf)); continue
    if em.shape != (H, W):
        H, W = em.shape  # trust the image
    lab, counts = rasterize(d.get("shapes", []), H, W)
    for k in tot: tot[k] += counts[k]
    for k in fin: fin[k] += int((lab == k).sum())
    if not lab.any():
        continue
    ds.add_plane(em, lab, source_image=os.path.basename(jpg), source_shape_xy=(W, H),
                 voxel_size_nm={"x": CFG["vx"], "y": CFG["vx"]}, label_kind="semantic",
                 class_map=CLASS_MAP, organelle_values=ORG_VALUES,
                 id_prefix=f"{SUBSET}_")
path, n = ds.write_manifest()
print(f"  polygons drawn: mito={tot[1]} er={tot[2]} ld={tot[3]} plasma_membrane={tot[4]}")
print(f"  FINAL label px: mito={fin[1]:,} er={fin[2]:,} ld={fin[3]:,} plasma_membrane={fin[4]:,}")
for k, name in [(1, "mito"), (2, "er"), (3, "ld")]:
    if tot[k] > 20 and fin[k] < 5000:
        print(f"  !! WARNING: {name} has {tot[k]} polygons but only {fin[k]} final px — likely overwritten!")
print(f"WROTE {n} crops -> {CFG['out']}\n  manifest: {path}")
