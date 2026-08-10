"""EMPIAR-12885 — AIVE organelle segmentation (Padman/Lazarou).
Whole-cell multi-class labels (Set1/2/3): per-set EM stack + 'ORGANELLE CLASS LABELS.tif'
with 'CLASS IDS.txt' giving the ordered value->class names.  AIVE boundaries are
AI-derived membrane x raw with HUMAN class assignment (flagged human-directed).
Plus DATASET1_TEST_ROI/Human_Vs_Unet: pure-human-classified mito (Mito1/2 HUMAN; Mito3/4 = Unet, excluded).
"""
import os, sys, re, urllib.request, urllib.parse, shutil
import numpy as np
import tifffile
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc
from readers import to_uint8_fullrange

B = "https://ftp.ebi.ac.uk/empiar/world_availability/12885/data"
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "empiar_12885_aive")
WORK = os.path.join(OUT, "_work")
NON_ORGANELLE = {"cytoskel", "plasmem", "chromatin"}
SETS = [(1, (6.5104, 6.5104, 10.0)), (2, (6.5104, 6.5104, 10.0)), (3, (6.5104, 6.5104, 10.0))]

META = {
    "name": "empiar_12885_aive",
    "source_repo": "EMPIAR", "accession": "EMPIAR-12885",
    "doi": "10.6019/EMPIAR-12885", "license": "CC0",
    "paper": "Padman et al., JCB 2024 (10.1083/jcb.202411138) — AIVE",
    "gt_provenance": "Hand-drawn MIB training labels + human class assignment; AIVE boundaries are AI-derived membrane x raw (human-directed, NOT pure hand-traced). Human_Vs_Unet Mito1/2 = single-human classification.",
    "modality": "FIB-SEM (3D)", "dimensionality": "3D",
    "label_encoding": "semantic, per-set value->class (from CLASS IDS.txt)",
    "organelle_classes": "mito/LD/endosomes/Golgi/nucleus/ER/vesicles (non-organelle: cytoskeleton, plasma-membrane, chromatin)",
    "z_rule": "z=10 nm -> every 40th plane",
    "alignment": "1:1 (labels share EM grid per set)",
    "source_url": B,
    "caveat": "AIVE labels = AI membrane geometry + human class ID. Flagged human-directed; whole-cell labels included.",
    "notes": "",
}


def dl(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=900) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, path)
    return path


def listing(url):
    try:
        html = urllib.request.urlopen(url, timeout=60).read().decode()
    except Exception as e:
        print("  listing fail", url, e); return []
    return re.findall(r'href="([^"?/][^"]*)"', html)


def g2(a):
    return a[..., 0] if (a.ndim == 3 and a.shape[-1] in (3, 4)) else a


def proc_set(ds, n, vox):
    d = f"{B}/Set{n}_-_AIVE_SOURCE_DATA/"
    entries = listing(d)
    dec = {urllib.parse.unquote(e): e for e in entries}
    em_name = next((dec[k] for k in dec if k.lower().endswith("overview.tif")
                    and "class" not in k.lower() and "clahe" not in k.lower()
                    and "membrane" not in k.lower()), None)
    lab_name = next((dec[k] for k in dec if "organelle class labels" in k.lower()), None)
    ids_name = next((dec[k] for k in dec if "class ids" in k.lower()), None)
    if not (em_name and lab_name and ids_name):
        print(f"  Set{n}: missing {em_name=} {lab_name=} {ids_name=}"); return
    ids_txt = urllib.request.urlopen(d + ids_name, timeout=60).read().decode(errors="ignore")
    names = [ln.strip() for ln in ids_txt.splitlines() if ln.strip()]
    class_map = {i + 1: nm.lower() for i, nm in enumerate(names)}
    org_vals = {v for v, nm in class_map.items() if nm not in NON_ORGANELLE}
    print(f"  Set{n}: map={class_map} org={sorted(org_vals)}")
    em = g2(tifffile.imread(dl(d + em_name, os.path.join(WORK, f"set{n}_em.tif"))))
    lb = g2(tifffile.imread(dl(d + lab_name, os.path.join(WORK, f"set{n}_lb.tif"))))
    if em.ndim == 2:
        em, lb = em[None], lb[None]
    if em.shape != lb.shape:
        print(f"  Set{n} shape {em.shape} vs {lb.shape}");
    Z = em.shape[0]
    step = sc.zstep_for_spacing(vox[2])
    vx = {"x": vox[0], "y": vox[1], "z": vox[2]}
    for zi in range(0, Z, step):
        ds.add_plane(to_uint8_fullrange(em[zi]), lb[zi].astype(np.int32),
                     source_image=f"Set{n}_-_AIVE_SOURCE_DATA/{urllib.parse.unquote(em_name)}",
                     source_shape_xy=(em.shape[2], em.shape[1]), z_index=int(zi),
                     z_physical_nm=float(zi * vox[2]), voxel_size_nm=vx,
                     label_kind="semantic", class_map=class_map, organelle_values=org_vals,
                     subdir=f"set{n}", id_prefix=f"set{n}_")
    for f in (f"set{n}_em.tif", f"set{n}_lb.tif"):
        try: os.remove(os.path.join(WORK, f))
        except OSError: pass


def proc_human_mito(ds):
    d = f"{B}/DATASET1_TEST_ROI/Human_Vs_Unet_-_Mito_Classification/"
    entries = listing(d)
    dec = {urllib.parse.unquote(e): e for e in entries}
    vox = (3.2552, 3.2552, 10.0); vx = {"x": vox[0], "y": vox[1], "z": vox[2]}
    step = sc.zstep_for_spacing(vox[2])
    for k in dec:
        kl = k.lower()
        if "- raw -" in kl and "human" in kl:  # only HUMAN raw/label pairs
            base = k[:k.lower().index("- raw -")]
            lab_k = next((x for x in dec if x.lower().startswith(base.lower())
                          and "- aive -" in x.lower() and "human" in x.lower()), None)
            if not lab_k:
                continue
            em = g2(tifffile.imread(dl(d + dec[k], os.path.join(WORK, "hv_em.tif"))))
            lb = g2(tifffile.imread(dl(d + dec[lab_k], os.path.join(WORK, "hv_lb.tif"))))
            if em.ndim == 2:
                em, lb = em[None], lb[None]
            tag = re.sub(r'[^a-z0-9]+', '', base.lower())
            print(f"  Human mito {base.strip()}: {em.shape}")
            for zi in range(0, em.shape[0], step):
                ds.add_plane(to_uint8_fullrange(em[zi]), lb[zi],
                             source_image=f"DATASET1_TEST_ROI/Human_Vs_Unet/{k.strip()}",
                             source_shape_xy=(em.shape[2], em.shape[1]), z_index=int(zi),
                             z_physical_nm=float(zi * vox[2]), voxel_size_nm=vx,
                             label_kind="instance_single", organelle_name="mitochondria",
                             subdir="human_vs_unet", id_prefix=f"hv_{tag}_")
            for f in ("hv_em.tif", "hv_lb.tif"):
                try: os.remove(os.path.join(WORK, f))
                except OSError: pass


def main():
    os.makedirs(WORK, exist_ok=True)
    ds = sc.Dataset(OUT, META)
    for n, vox in SETS:
        proc_set(ds, n, vox)
    proc_human_mito(ds)
    path, nc = ds.write_manifest()
    print(f"DONE: {nc} crops -> {path}")
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
