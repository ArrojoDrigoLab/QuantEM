"""
Shared benchmark harness: test-crop loader + region-masked ground-truth.

Evaluation convention (split definitions and tile builds live in
../segmentation_dataset/):
  * Each crop is a TILE x TILE canvas holding the real EM (valid_region) with the
    annotation centered; outside valid_region is zero-pad => IGNORE.
  * coverage_tier 'full'  -> score the whole valid_region (image-complete, bg = TN).
  * coverage_tier 'partial'/'sparse' -> score only inside annotation_bbox
    (intersect valid_region); everything else IGNORE (not background).
  * Models receive the real-EM valid_region crop as input (at whatever size the
    model expects) and predictions are scored only inside eval_mask.

Ground-truth value resolution:
  * instance datasets (label_encoding starts with 'instance') -> gt = label > 0.
  * semantic datasets -> gt = isin(label, target_values), where target_values are
    resolved PER-CROP from manifest crops[].organelles_present (name -> value),
    unioning every class whose name matches the requested organelle. Falls back to
    a fixed per-dataset map if organelles_present is missing.
"""
import os, csv, json, glob, re, functools
import numpy as np
import tifffile

# ----------------------------------------------------------------------------
# The benchmark test datasets per organelle (splits/benchmark_<organelle>.csv,
# split=='test'). The un-annotated fourth LD dataset is excluded.
BENCHMARK_DATASETS = {
    "mito":    ["zenodo_mitoem2", "empiar_10982_mitonet_benchmark", "orgsegnet_plant",
                "deeppi_em_skeletal_muscle", "deepcontact_tem"],
    "er":      ["empiar_12885_aive", "empiar_10994_hela_sbfsem", "deepcontact_tem",
                "empiar_13156_hela_stard3_er", "lab_islet_liver_er"],
    "nucleus": ["zenodo_3675220_platynereis", "sbiad2822_nuclei",
                "segapp_islet_nucleus", "orgsegnet_plant"],
    "ld":      ["empiar_13420_macrophage_a431", "deepcontact_cell", "empiar_12885_aive"],
}

# short, figure-friendly dataset labels
DATASET_LABEL = {
    "zenodo_mitoem2": "ME2-Mossy",
    "empiar_10982_mitonet_benchmark": "MitoNet",
    "orgsegnet_plant": "PlantHunter",
    "deeppi_em_skeletal_muscle": "DeepPI",
    "deepcontact_tem": "DeepContact",
    "empiar_12885_aive": "AIVE",
    "empiar_10994_hela_sbfsem": "EMPIAR-10994",
    "empiar_13156_hela_stard3_er": "EMPIAR-13156",
    "lab_islet_liver_er": "islet/liver-ER",
    "zenodo_3675220_platynereis": "Platynereis",
    "sbiad2822_nuclei": "NucleiNet",
    "segapp_islet_nucleus": "IsletSEM",
    "empiar_13420_macrophage_a431": "MacrophageSEM",
    "deepcontact_cell": "DeepContact",
}

# Fallback fixed value maps (only used when organelles_present is missing).
FIXED_MAP = {
    "deepcontact_tem":  {"mito": [1], "er": [2], "ld": [3]},
    "deepcontact_cell": {"mito": [1], "er": [2], "ld": [3]},
    "deepcontact_sem":  {"mito": [1], "er": [2], "ld": [3]},
    "orgsegnet_plant":  {"mito": [2], "nucleus": [4]},
    "empiar_13156_hela_stard3_er": {"er": [1], "mito": [3]},
    "deeppi_em_skeletal_muscle": {"mito": [1]},
    "lab_islet_liver_er": {"er": [1]},
}


def _name_is_organelle(name, organelle):
    n = name.lower()
    if organelle == "mito":
        return n in ("mito", "mitos", "mitochondria", "mitochondrion", "mitos1", "mitos2")
    if organelle == "er":
        # ER and its subtypes; excludes endosomes / lipid / endo* keys.
        return (n == "er") or n.startswith("er_") or n.endswith("_er") or ("endoplasmic" in n)
    if organelle == "nucleus":
        return n in ("nucleus", "nuclei", "nuc")   # NOT nucleolus
    if organelle == "ld":
        return n.startswith("lipid_droplet") or n in ("ld", "lds", "lipiddroplet")
    return False


class CropSpec:
    __slots__ = ("dataset", "crop_id", "organelle", "em_path", "label_path",
                 "valid_region", "annotation_bbox", "coverage_tier",
                 "is_instance", "gt_values", "voxel_nm", "source_image",
                 "modality", "scale_band")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return f"<CropSpec {self.organelle}/{self.dataset}/{self.crop_id} cover={self.coverage_tier}>"


@functools.lru_cache(maxsize=None)
def _manifest(dataset, seg_root):
    mp = os.path.join(seg_root, dataset, "manifest.json")
    m = json.load(open(mp))
    idx = {c["crop_id"]: c for c in m.get("crops", [])}
    le = m.get("dataset", {}).get("label_encoding", "")
    return idx, le


def _resolve_values(dataset, crop_entry, organelle):
    op = crop_entry.get("organelles_present")
    if isinstance(op, dict):
        vals = []
        for name, v in op.items():
            if isinstance(v, dict) and "value" in v and _name_is_organelle(name, organelle):
                vals.append(int(v["value"]))
        if vals:
            return sorted(set(vals))
    fm = FIXED_MAP.get(dataset, {})
    if organelle in fm:
        return list(fm[organelle])
    return None


def iter_test_crops(organelle, seg_root, splits_dir=None):
    """Yield a CropSpec for every split=='test' crop of `organelle`.
    `seg_root` is the root directory holding one sub-directory per dataset (each
    with its manifest.json + crop TIFFs); `splits_dir` defaults to <seg_root>/splits."""
    if splits_dir is None:
        splits_dir = os.path.join(seg_root, "splits")
    rows = [r for r in csv.DictReader(open(os.path.join(splits_dir, f"benchmark_{organelle}.csv")))
            if r["split"] == "test"]
    for r in rows:
        ds = r["dataset"]
        if ds not in BENCHMARK_DATASETS[organelle]:
            continue
        idx, le = _manifest(ds, seg_root)
        c = idx.get(r["crop_id"])
        if c is None:
            continue
        is_instance = le.strip().lower().startswith("instance")
        gt_values = None if is_instance else _resolve_values(ds, c, organelle)
        em_rel = c["em_file"]
        lab_rel = c.get("label_file", em_rel.replace("_em.tif", "_label.tif"))
        yield CropSpec(
            dataset=ds, crop_id=r["crop_id"], organelle=organelle,
            em_path=os.path.join(seg_root, ds, em_rel),
            label_path=os.path.join(seg_root, ds, lab_rel),
            valid_region=c.get("valid_region_in_canvas_xyxy"),
            annotation_bbox=c.get("annotation_bbox_in_canvas_xyxy"),
            coverage_tier=c.get("coverage_tier", "sparse"),
            is_instance=is_instance, gt_values=gt_values,
            voxel_nm=(c.get("voxel_size_nm") or {}).get("x"),
            source_image=c.get("source_image"),
            modality=r.get("modality"), scale_band=r.get("scale_band"),
        )


# --- DeepContact GT rebuild (optional) -------------------------------------
# Rebuilds a binary organelle mask for a deepcontact crop directly from the
# source Labelme polygons (label-audit path). The shipped crop labels are
# correct as stored; the rebuild only runs when a polygon directory is given.
_DC_LABELS = {"er": {"er"}, "mito": {"mito", "mitochondria"},
              "ld": {"ld", "lipid droplet", "lipid_droplet", "lipiddroplet"}}
_dc_jmap = {}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", os.path.splitext(os.path.basename(s))[0].lower())


def _deepcontact_gt(spec, shape, seg_root, dc_json_dir=None):
    """Rebuild a binary organelle mask for a deepcontact crop from its source JSON
    polygons, placed into the canvas at the crop's valid_region.

    dc_json_dir: directory holding the source Labelme *.json polygon files for
    spec.dataset (absolute, or relative to seg_root). Optional: when omitted, or
    when no matching source JSON exists, returns None."""
    if dc_json_dir is None or not spec.source_image:
        return None
    jdir = os.path.join(seg_root, dc_json_dir)
    if spec.dataset not in _dc_jmap:
        _dc_jmap[spec.dataset] = {_norm(f): f for f in glob.glob(os.path.join(jdir, "*.json"))}
    jmap = _dc_jmap[spec.dataset]
    key = _norm(spec.source_image)
    jf = jmap.get(key) or next((v for k, v in jmap.items() if key in k or k in key), None)
    if jf is None:
        return None
    try:
        from skimage.draw import polygon as sk_polygon
    except Exception:
        return None
    j = json.load(open(jf))
    vr = spec.valid_region or [0, 0, shape[1], shape[0]]
    ox, oy = int(round(vr[0])), int(round(vr[1]))
    want = _DC_LABELS.get(spec.organelle, set())
    mask = np.zeros(shape, bool)
    for s in j.get("shapes", []):
        if s.get("label", "").lower() in want:
            pts = np.asarray(s["points"], float)
            if len(pts) < 3:
                continue
            rr, cc = sk_polygon(pts[:, 1] + oy, pts[:, 0] + ox, shape=shape)
            mask[rr, cc] = True
    return mask


def _bbox_mask(shape, bbox):
    m = np.zeros(shape, bool)
    if bbox is None:
        m[:] = True
        return m
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    H, W = shape
    x0 = max(0, min(x0, W)); x1 = max(0, min(x1, W))
    y0 = max(0, min(y0, H)); y1 = max(0, min(y1, H))
    if x1 <= x0 or y1 <= y0:
        return m
    m[y0:y1, x0:x1] = True
    return m


def eval_mask_for(spec, shape):
    """Boolean mask of pixels that are scored (region-masked, ignore elsewhere)."""
    valid = _bbox_mask(shape, spec.valid_region)
    if spec.coverage_tier in ("partial", "sparse") and spec.annotation_bbox is not None:
        valid &= _bbox_mask(shape, spec.annotation_bbox)
    return valid


def load_crop(spec, want_instance=False):
    """
    Returns dict:
      em         : (H,W) uint8 grayscale EM canvas
      gt_binary  : (H,W) bool  ground-truth foreground for this organelle
      gt_instance: (H,W) int32 instance labels (only if instance dataset) or None
      eval_mask  : (H,W) bool  scored region (ignore elsewhere)
      valid_slice: (y0,y1,x0,x1) real-EM extent (for cropping model input)
    """
    em = tifffile.imread(spec.em_path)
    if em.ndim == 3:
        em = em[em.shape[0] // 2]
    em = np.asarray(em)
    lab = tifffile.imread(spec.label_path)
    if lab.ndim == 3:
        lab = lab[lab.shape[0] // 2]
    lab = np.asarray(lab)
    H, W = lab.shape
    if em.shape != lab.shape:
        # center/resize safety: crop or pad em to label shape
        eh, ew = em.shape
        out = np.zeros((H, W), em.dtype)
        h = min(H, eh); w = min(W, ew)
        out[:h, :w] = em[:h, :w]
        em = out

    emask = eval_mask_for(spec, lab.shape)

    gt_instance = None
    if spec.is_instance:
        gt_binary = lab > 0
        if want_instance:
            gt_instance = lab.astype(np.int32)
    else:
        vals = spec.gt_values or []
        gt_binary = np.isin(lab, vals) if vals else np.zeros_like(lab, bool)
        # (semantic value maps are correct as stored; no label rebuild is applied)

    # restrict GT to eval region (outside is ignore)
    gt_binary = gt_binary & emask
    if gt_instance is not None:
        gt_instance = np.where(emask, gt_instance, 0)

    vr = spec.valid_region or [0, 0, W, H]
    x0, y0, x1, y1 = [int(round(v)) for v in vr]
    x0 = max(0, min(x0, W)); x1 = max(0, min(x1, W))
    y0 = max(0, min(y0, H)); y1 = max(0, min(y1, H))

    return dict(em=em, gt_binary=gt_binary, gt_instance=gt_instance,
                eval_mask=emask, valid_slice=(y0, y1, x0, x1), shape=lab.shape)


# model x organelle applicability matrix (which models are run on which organelle)
MODEL_ORGANELLES = {
    "mitonet":     ["mito"],
    "nucleonet":   ["nucleus"],
    "lipidnet":    ["ld"],
    "deepcontact": ["mito", "er"],
    "microsam":    ["mito", "er", "nucleus", "ld"],
    "incasem":     ["mito", "er"],
}
