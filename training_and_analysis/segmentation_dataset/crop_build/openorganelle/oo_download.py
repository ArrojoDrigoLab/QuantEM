"""Download OpenOrganelle target crops (ER/mito/LD) into <corpus root>/openOrganelle.

Per target crop:
  - raw EM: 4096x4096 (clamped) window centered on the crop, EM s0 native res,
    z-planes >=200nm apart, native dtype, lossless TIFF (z,y,x).
  - segmentations (Option A): all, mito, ld, er at native s0, full crop, as-is.
  - a per-crop manifest fragment; merged into a top-level manifest.json by the runner.

Resumable: skips a crop whose <crop>/crop_manifest.json already exists.
Usage: python oo_download.py <folder> [<folder> ...]
"""
import os, sys, json, math
import numpy as np, requests, tensorstore as ts
import tifffile

OUTROOT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "openOrganelle")
B = "https://janelia-cosem-datasets.s3.amazonaws.com"
H = {"User-Agent": "quantem-dataset-build/1.0"}
# Reslice z-spacing (nm). Default 200.0.
# Override, additively, via env var OO_Z_SPACING_NM or the --z-spacing-nm CLI flag (see resolve_z_spacing).
# The original (baseline) grid is always 200nm; an active spacing < 200 densifies the grid and marks the
# extra planes as "dense-only" so a later step can recover the exact 200nm subset.
Z_SPACING_BASELINE_NM = 200.0
Z_SPACING_NM = float(os.environ.get("OO_Z_SPACING_NM", Z_SPACING_BASELINE_NM))
CAP = 4096
SEG_CLASSES = ["all", "mito", "ld", "er"]

_HERE = os.path.dirname(os.path.abspath(__file__))
DSETS = {d["folder"]: d for d in json.load(open(os.path.join(_HERE, "oo_datasets.json")))}
CC = json.load(open(os.path.join(_HERE, "oo_crop_classes.json")))
MITO = ["mito","mito_mem","mito_lum","mito_ribo"]; ER = ["er","er_mem","er_lum","er_mem_all"]; LD = ["ld","ld_mem","ld_lum"]


def to_https(url):
    return (url.replace("s3://janelia-cosem-datasets/", B + "/")
               .replace("open.quiltdata.com/b/janelia-cosem-datasets/tree/", B + "/")).rstrip("/")


def keyof(https): return https.split(B + "/")[1]


def jget(url):
    r = requests.get(url, timeout=40, headers=H)
    return (r.json() if r.status_code == 200 else None), r.status_code


def open_zarr(base_url):  # base_url ends at the array dir (no trailing /s0)
    return ts.open({"driver": "zarr", "kvstore": {"driver": "http", "base_url": base_url.rstrip("/") + "/"}},
                   read=True).result()


def save_tiff(path, arr):
    a = arr
    if a.dtype == np.uint64 or a.dtype == np.int64:
        mx = int(a.max())
        a = a.astype(np.uint32 if mx < 2**32 else np.float64)
    tifffile.imwrite(path, a, compression="zlib")
    return str(a.dtype)


def annotation_type(recon_root, crop, cls):
    za, st = jget(f"{B}/{keyof(recon_root)}/labels/groundtruth/{crop}/{cls}/.zattrs")
    if not za: return None
    return za.get("cellmap", {}).get("annotation", {}).get("annotation_type", {}).get("type")


def has(classes, names): s = set(classes); return any(n in s for n in names)


def resolve_z_spacing(argv):
    """Return (z_spacing_nm, remaining_argv). Resolution order (later wins):
    module default (200) -> env OO_Z_SPACING_NM -> --z-spacing-nm CLI flag.
    Strips the flag (and its value) from argv so the positional folder list is unaffected."""
    z = Z_SPACING_NM  # already folds in the env var at import time
    out = []
    it = iter(argv)
    for a in it:
        if a == "--z-spacing-nm":
            z = float(next(it))
        elif a.startswith("--z-spacing-nm="):
            z = float(a.split("=", 1)[1])
        else:
            out.append(a)
    return z, out


def dense_flags(z_plane_indices, emz, z_spacing_nm, baseline_nm=Z_SPACING_BASELINE_NM):
    """Per-plane bool: True iff this plane is 'dense-only' -- it exists only because spacing<baseline
    and its physical z is NOT on the baseline (200nm) grid measured from the crop's first plane
    (z-origin = z_plane_indices[0]). Dropping every True marker recovers the exact baseline subset.
    When spacing>=baseline, all planes lie on the baseline grid -> all False."""
    if not z_plane_indices:
        return []
    z0 = z_plane_indices[0]
    flags = []
    for z in z_plane_indices:
        offset_nm = (z - z0) * emz
        # on the baseline grid iff offset is (within rounding) an integer multiple of baseline_nm
        r = offset_nm % baseline_nm
        on_grid = min(r, baseline_nm - r) < 1e-6
        flags.append(bool(z_spacing_nm < baseline_nm and not on_grid))
    return flags


def download_crop(folder, crop, classes, z_spacing_nm=None):
    if z_spacing_nm is None:
        z_spacing_nm = Z_SPACING_NM
    d = json.load(open(os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "_work", "oo_gt_out", f"{folder}.json")))
    em = d["em_scale_zyx_nm"]; emz, emy, emx = em
    Z, Y, X = d["em_shape_zyx"]
    recon = d["recon_root"]
    em_url = to_https(DSETS[folder]["url"])
    cinfo = next(c for c in d["crops"] if c["crop"] == crop)
    sz, sy, sx = cinfo["shape_zyx"]; csz, csy, csx = cinfo["scale_zyx_nm"]; tz, ty, tx = cinfo["trans_zyx_nm"]
    x_nm, y_nm, z_nm = sx*csx, sy*csy, sz*csz

    outdir = os.path.join(OUTROOT, folder, crop)
    if os.path.exists(os.path.join(outdir, "crop_manifest.json")):
        return None  # resume: already done
    os.makedirs(outdir, exist_ok=True)

    # --- annotation footprint in EM s0 voxels ---
    ax0 = (tx - 0) / emx; ax1 = (tx + x_nm) / emx
    ay0 = (ty - 0) / emy; ay1 = (ty + y_nm) / emy
    az0 = (tz - 0) / emz; az1 = (tz + z_nm) / emz
    annx = [int(round(ax0)), int(round(ax1))]; anny = [int(round(ay0)), int(round(ay1))]
    annz = [max(0, int(math.floor(az0))), min(Z, int(math.ceil(az1)))]

    # --- 4096 window centered on annotation center, clamped/shifted into volume ---
    def window(c0, c1, dim):
        center = (c0 + c1) / 2.0
        w = min(CAP, dim)
        w0 = int(round(center - w / 2.0))
        w0 = max(0, min(w0, dim - w))
        return w0, w0 + w
    wx0, wx1 = window(annx[0], annx[1], X)
    wy0, wy1 = window(anny[0], anny[1], Y)
    ww, wh = wx1 - wx0, wy1 - wy0

    # --- z planes >=z_spacing_nm apart over the crop z-range ---
    step = max(1, int(math.ceil(z_spacing_nm / emz)))
    zs = list(range(annz[0], max(annz[0] + 1, annz[1]), step))
    dense = dense_flags(zs, emz, z_spacing_nm)

    # --- read raw EM in z-chunk-aligned batches (touch each chunk once) ---
    em_arr = open_zarr(f"{em_url}/s0")
    zarray, _ = jget(f"{em_url}/s0/.zarray")
    chunk_z = zarray["chunks"][0]
    raw_dtype = str(em_arr.dtype.numpy_dtype)
    planes = {}
    z0a = (zs[0] // chunk_z) * chunk_z
    for zc0 in range(z0a, zs[-1] + 1, chunk_z):
        zc1 = min(zc0 + chunk_z, Z)
        zsel = [z for z in zs if zc0 <= z < zc1]
        if not zsel: continue
        block = np.asarray(em_arr[zc0:zc1, wy0:wy1, wx0:wx1].read().result())
        for z in zsel: planes[z] = block[z - zc0]
    raw = np.stack([planes[z] for z in zs], axis=0)  # (n, wh, ww)
    raw_saved_dtype = save_tiff(os.path.join(outdir, "raw.tif"), raw)

    # --- segmentations (Option A: all, mito, ld, er where present) ---
    seg_entries = []
    for cls in SEG_CLASSES:
        present = (cls == "all") or (cls in classes)
        if not present: continue
        cbase = f"{recon}/labels/groundtruth/{crop}/{cls}"
        czarray, st = jget(f"{cbase}/s0/.zarray")
        if not czarray:
            continue
        seg_arr = open_zarr(f"{cbase}/s0")
        seg = np.asarray(seg_arr.read().result())
        atype = annotation_type(recon, crop, cls)
        fname = f"seg_{cls}.tif"
        saved_dtype = save_tiff(os.path.join(outdir, fname), seg)
        # seg multiscale scale (nm) from its .zattrs
        sza, _ = jget(f"{cbase}/.zattrs")
        sscale = None
        try:
            ms = sza["multiscales"][0]
            for ct in ms["datasets"][0]["coordinateTransformations"]:
                if ct.get("type") == "scale": sscale = ct["scale"]
        except Exception:
            pass
        # seg origin in EM voxels = annotation footprint origin; offset within raw crop (EM px)
        seg_entries.append({
            "class_name": cls,
            "annotation_type": atype,
            "file": fname,
            "source_url": f"{cbase}/s0",
            "resolution_nm_zyx": sscale,
            "shape_zyx": list(seg.shape),
            "source_dtype": str(seg.dtype),
            "saved_dtype": saved_dtype,
            "location_in_original_image_voxels_emS0": {"z": annz, "y": anny, "x": annx},
            "location_in_raw_crop_px": {"y": [anny[0]-wy0, anny[1]-wy0], "x": [annx[0]-wx0, annx[1]-wx0]},
            "physical_origin_nm_zyx": [tz, ty, tx],
            "note": "seg at its own native res; align to raw via physical_origin_nm + resolution ratio",
        })

    target_present = [o for o, names in (("mito", MITO), ("er", ER), ("ld", LD)) if has(classes, names)]
    entry = {
        "crop_id": f"{folder}/{crop}",
        "dataset": folder,
        "experiment": DSETS[folder]["exp"],
        "original_image": {
            "name_id": f"{folder} ({em_url.split('/em/')[-1]})",
            "source_url": f"{em_url}/s0",
            "resolution_nm_zyx": em,
            "full_shape_zyx": [Z, Y, X],
            "dtype": raw_dtype,
        },
        "raw_crop": {
            "file": "raw.tif",
            "shape_zyx": list(raw.shape),
            "resolution_nm_zyx": em,
            "saved_dtype": raw_saved_dtype,
            "window_bbox_in_original_voxels_emS0": {"y": [wy0, wy1], "x": [wx0, wx1]},
            "z_plane_indices_emS0": zs,
            "z_plane_physical_nm": [round(z * emz, 3) for z in zs],
            "z_spacing_nm_min": z_spacing_nm,
            "z_spacing_nm_baseline": Z_SPACING_BASELINE_NM,
            "z_plane_dense_only": dense,  # True = exists only at <baseline spacing (drop these -> exact 200nm subset)
            "padded_to": [CAP, CAP],
            "actual_window_xy": [ww, wh],
            "note": "real EM context; window centered on annotation, clamped to volume bounds",
        },
        "segmented_area": {
            "annotation_bbox_in_original_voxels_emS0": {"z": annz, "y": anny, "x": annx},
            "annotation_bbox_in_raw_crop_px": {"y": [anny[0]-wy0, anny[1]-wy0], "x": [annx[0]-wx0, annx[1]-wx0]},
            "annotation_size_px_xy": [annx[1]-annx[0], anny[1]-anny[0]],
            "annotation_physical_size_nm_xyz": [x_nm, y_nm, z_nm],
            "annotation_crop_native_resolution_nm_zyx": [csz, csy, csx],
        },
        "organelles_target_present": target_present,
        "all_classes_present_in_crop": sorted(classes),
        "segmentations": seg_entries,
    }
    json.dump(entry, open(os.path.join(outdir, "crop_manifest.json"), "w"), indent=2)
    return entry


def target_crops(folder):
    info = CC.get(folder)
    if not info: return []
    out = []
    for c in info["crops"]:
        if has(c["classes"], MITO) or has(c["classes"], ER) or has(c["classes"], LD):
            out.append((c["crop"], c["classes"]))
    return out


if __name__ == "__main__":
    os.makedirs(OUTROOT, exist_ok=True)
    z_spacing, rest = resolve_z_spacing(sys.argv[1:])
    print(f"[oo_download] z_spacing_nm={z_spacing} (baseline {Z_SPACING_BASELINE_NM})", flush=True)
    folders = rest or sorted(CC.keys())
    for folder in folders:
        crops = target_crops(folder)
        for crop, classes in crops:
            try:
                e = download_crop(folder, crop, classes, z_spacing_nm=z_spacing)
                if e is None:
                    print(f"SKIP {folder}/{crop} (done)", flush=True)
                else:
                    rc = e["raw_crop"]; print(f"OK   {folder}/{crop} raw{rc['shape_zyx']} segs={[s['class_name'] for s in e['segmentations']]}", flush=True)
            except Exception as ex:
                print(f"FAIL {folder}/{crop}: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
