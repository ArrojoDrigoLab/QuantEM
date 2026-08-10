"""EMPIAR-10982 (MitoNet held-out benchmark) -> GT crop collection.

6 3D mitochondria instance volumes + 100 2D TEM pairs. All labels are
manual / proofread / 2-reviewer-reviewed held-out ground truth (Conrad &
Narayan, Cell Systems 2023).  Mito only, instance-encoded.
"""
import os, sys, argparse, urllib.request, shutil
import numpy as np
import tifffile
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/10982/data"
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "empiar_10982_mitonet_benchmark")
WORK = os.path.join(OUT, "_work")

# benchmark -> isotropic voxel size (nm)  [from Conrad & Narayan 2023]
VOX = {"c_elegans": 24.0, "fly_brain": 12.0, "glycolytic_muscle": 18.0,
       "hela_cell": 15.0, "lucchi_pp": 5.0, "salivary_gland": 15.0}

META = {
    "name": "empiar_10982_mitonet_benchmark",
    "source_repo": "EMPIAR", "accession": "EMPIAR-10982",
    "doi": "10.6019/EMPIAR-10982", "license": "CC0",
    "paper": "Conrad & Narayan, Cell Systems 2023 (PMC9883049) — MitoNet/empanada benchmark",
    "gt_provenance": "manual ground truth (ariadne + in-house; >=2 expert reviewers); held-out eval set, never trained on",
    "modality": "mixed volume-EM (FIB/SBF/ssSEM) + 2D TEM",
    "label_encoding": "instance (mitochondria; nonzero = unique object id, 0=bg)",
    "organelle_classes": {"nonzero": "mitochondria"},
    "z_rule": "planes sampled >= 400 nm apart (per-benchmark isotropic voxel)",
    "alignment": "labels 1:1 voxel-registered to EM",
    "source_url": BASE,
    "notes": "TEM set = 100 independent 2D micrographs (>=6 nm/px). Sub-4096 sources centered + zero-padded.",
}


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, path)
    return path


def to_gray2d(a):
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., 0]
    return a


def proc_volume(ds, name):
    vox = VOX[name]
    em_u = f"{BASE}/mito_benchmarks/{name}/{name}_em.tif"
    mt_u = f"{BASE}/mito_benchmarks/{name}/{name}_mito.tif"
    em_p = download(em_u, os.path.join(WORK, f"{name}_em.tif"))
    mt_p = download(mt_u, os.path.join(WORK, f"{name}_mito.tif"))
    em = tifffile.imread(em_p)
    mt = tifffile.imread(mt_p)
    if em.ndim == 2:  # safety
        em, mt = em[None], mt[None]
    assert em.shape == mt.shape, f"{name}: {em.shape} vs {mt.shape}"
    Z, Y, X = em.shape
    step = sc.zstep_for_spacing(vox)
    kept = list(range(0, Z, step))
    print(f"  {name}: {em.shape} vox={vox}nm zstep={step} -> {len(kept)} planes")
    for zi in kept:
        ds.add_plane(em[zi], mt[zi], source_image=f"mito_benchmarks/{name}/{name}_em.tif",
                     source_shape_xy=(X, Y), z_index=int(zi),
                     z_physical_nm=float(zi * vox),
                     voxel_size_nm={"x": vox, "y": vox, "z": vox},
                     label_kind="instance_single", organelle_name="mitochondria",
                     subdir=name, id_prefix=f"{name[:4]}_")
    os.remove(em_p); os.remove(mt_p)


def proc_tem(ds):
    import re
    html = urllib.request.urlopen(f"{BASE}/tem_benchmark/images/", timeout=60).read().decode()
    names = sorted(set(re.findall(r'href="([^"?/][^"]*\.tiff?)"', html)))
    print(f"  TEM: {len(names)} micrographs")
    for nm in names:
        em_p = download(f"{BASE}/tem_benchmark/images/{nm}", os.path.join(WORK, "tem_img_" + nm))
        mk_p = download(f"{BASE}/tem_benchmark/masks/{nm}", os.path.join(WORK, "tem_msk_" + nm))
        em = to_gray2d(tifffile.imread(em_p))
        mk = to_gray2d(tifffile.imread(mk_p))
        if em.shape != mk.shape:
            print(f"    skip {nm}: shape {em.shape} vs {mk.shape}");
            os.remove(em_p); os.remove(mk_p); continue
        Y, X = em.shape
        ds.add_plane(em, mk, source_image=f"tem_benchmark/images/{nm}",
                     source_shape_xy=(X, Y), z_index=None, z_physical_nm=None,
                     voxel_size_nm={"note": ">=6 nm/px, per-image"},
                     label_kind="instance_single", organelle_name="mitochondria",
                     subdir="tem", id_prefix="tem_")
        os.remove(em_p); os.remove(mk_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--tem", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    ds = sc.Dataset(OUT, META)
    bms = list(VOX) if a.all else ([x for x in a.only.split(",") if x] if a.only else [])
    for b in bms:
        proc_volume(ds, b)
    if a.tem or a.all:
        proc_tem(ds)
    path, n = ds.write_manifest()
    print(f"DONE: {n} crops -> {path}")
    # tidy work dir if empty
    try:
        if not os.listdir(WORK):
            os.rmdir(WORK)
    except OSError:
        pass


if __name__ == "__main__":
    main()
