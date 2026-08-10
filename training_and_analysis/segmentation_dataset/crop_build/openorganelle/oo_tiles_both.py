"""Produce BOTH orientations per crop: raw_xy.tif (X-Y planes along Z) and
raw_xz.tif (X-Z planes along Y), each z-sampled >=200nm. Segmentations (3D crops)
are shared. A single-orientation raw.tif from a prior oo_download run is reused as the
orientation its manifest declares (no re-download). Resumable: skips a crop once both
raw_xy.tif and raw_xz.tif exist.

Usage: python oo_tiles_both.py <folder> [<folder> ...]
"""
import os, sys, json, math
import numpy as np, tifffile
from oo_download import (to_https, open_zarr, save_tiff, jget, OUTROOT, CAP, Z_SPACING_NM,
                         Z_SPACING_BASELINE_NM, DSETS, resolve_z_spacing, dense_flags)

import json as _j
CC = _j.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oo_crop_classes.json")))
def _has(c, n): s = set(c); return any(x in s for x in n)
MITO=["mito","mito_mem","mito_lum","mito_ribo"]; ER=["er","er_mem","er_lum","er_mem_all"]; LD=["ld","ld_mem","ld_lum"]


def _win(c0, c1, dim):
    center = (c0 + c1) / 2.0; w = min(CAP, dim)
    w0 = max(0, min(int(round(center - w / 2.0)), dim - w)); return w0, w0 + w


def geom(d, cinfo, z_spacing_nm=None):
    if z_spacing_nm is None:
        z_spacing_nm = Z_SPACING_NM
    emz, emy, emx = d["em_scale_zyx_nm"]; Z, Y, X = d["em_shape_zyx"]
    sz, sy, sx = cinfo["shape_zyx"]; csz, csy, csx = cinfo["scale_zyx_nm"]; tz, ty, tx = cinfo["trans_zyx_nm"]
    x_nm, y_nm, z_nm = sx * csx, sy * csy, sz * csz
    annz = [max(0, int(math.floor(tz / emz))), min(Z, int(math.ceil((tz + z_nm) / emz)))]
    anny = [max(0, int(math.floor(ty / emy))), min(Y, int(math.ceil((ty + y_nm) / emy)))]
    annx = [max(0, int(math.floor(tx / emx))), min(X, int(math.ceil((tx + x_nm) / emx)))]
    # X-Y: window in Y,X; sample along Z (dense-only markers vs the 200nm baseline grid)
    wy0, wy1 = _win(anny[0], anny[1], Y); wx0, wx1 = _win(annx[0], annx[1], X)
    zs = list(range(annz[0], max(annz[0] + 1, annz[1]), max(1, int(math.ceil(z_spacing_nm / emz)))))
    zs_dense = dense_flags(zs, emz, z_spacing_nm)
    # X-Z: window in Z,X; sample along Y (spacing applied along the Y sampling axis)
    z0, z1 = _win(annz[0], annz[1], Z); x0, x1 = _win(annx[0], annx[1], X)
    ys = list(range(anny[0], max(anny[0] + 1, anny[1]), max(1, int(math.ceil(z_spacing_nm / emy)))))
    ys_dense = dense_flags(ys, emy, z_spacing_nm)
    return dict(em=(emz, emy, emx), shape=(Z, Y, X), annz=annz, anny=anny, annx=annx,
                z_spacing_nm=z_spacing_nm,
                xy=dict(wy=(wy0, wy1), wx=(wx0, wx1), zs=zs, dense=zs_dense,
                        shape=(len(zs), wy1 - wy0, wx1 - wx0)),
                xz=dict(wz=(z0, z1), wx=(x0, x1), ys=ys, dense=ys_dense,
                        shape=(len(ys), z1 - z0, x1 - x0)))


def read_xy(arr, chunk_z, g):
    (wy0, wy1), (wx0, wx1), zs = g["xy"]["wy"], g["xy"]["wx"], g["xy"]["zs"]; Z = g["shape"][0]
    planes = {}; z0a = (zs[0] // chunk_z) * chunk_z
    for zc0 in range(z0a, zs[-1] + 1, chunk_z):
        zc1 = min(zc0 + chunk_z, Z); sel = [z for z in zs if zc0 <= z < zc1]
        if not sel: continue
        blk = np.asarray(arr[zc0:zc1, wy0:wy1, wx0:wx1].read().result())
        for z in sel: planes[z] = blk[z - zc0]
    return np.stack([planes[z] for z in zs], axis=0)


def read_xz(arr, chunk_y, g):
    (z0, z1), (x0, x1), ys = g["xz"]["wz"], g["xz"]["wx"], g["xz"]["ys"]; Y = g["shape"][1]
    planes = {}; y0a = (ys[0] // chunk_y) * chunk_y
    for yc0 in range(y0a, ys[-1] + 1, chunk_y):
        yc1 = min(yc0 + chunk_y, Y); sel = [y for y in ys if yc0 <= y < yc1]
        if not sel: continue
        blk = np.asarray(arr[z0:z1, yc0:yc1, x0:x1].read().result())
        for y in sel: planes[y] = blk[:, y - yc0, :]
    return np.stack([planes[y] for y in ys], axis=0)


def tiff_shape(path):
    try:
        with tifffile.TiffFile(path) as t:
            n = len(t.pages); p0 = t.pages[0].shape; t.pages[n - 1].asarray()
            return (n, p0[0], p0[1])
    except Exception:
        return None


def xy_entry(g, emy, emx):
    a = g["xy"]; (wy0, wy1), (wx0, wx1) = a["wy"], a["wx"]
    return {"file": "raw_xy.tif", "orientation": "X-Y planes sampled along Z",
            "tile_axes_rows_cols": ["y", "x"], "tile_resolution_nm_rows_cols": [emy, emx],
            "sample_axis": "z", "shape_planes_rows_cols": list(a["shape"]),
            "window_bbox_in_original_voxels_emS0": {"y": [wy0, wy1], "x": [wx0, wx1]},
            "plane_indices_emS0": a["zs"], "plane_physical_nm": [round(z * g["em"][0], 3) for z in a["zs"]],
            "spacing_nm_min": g["z_spacing_nm"], "spacing_nm_baseline": Z_SPACING_BASELINE_NM,
            "plane_dense_only": a["dense"], "padded_to": [CAP, CAP],
            "annotation_bbox_in_tile_px": {"y": [g["anny"][0]-wy0, g["anny"][1]-wy0],
                                           "x": [g["annx"][0]-wx0, g["annx"][1]-wx0]}}


def xz_entry(g, emz, emx):
    a = g["xz"]; (z0, z1), (x0, x1) = a["wz"], a["wx"]
    return {"file": "raw_xz.tif", "orientation": "X-Z planes sampled along Y",
            "tile_axes_rows_cols": ["z", "x"], "tile_resolution_nm_rows_cols": [emz, emx],
            "sample_axis": "y", "shape_planes_rows_cols": list(a["shape"]),
            "window_bbox_in_original_voxels_emS0": {"z": [z0, z1], "x": [x0, x1]},
            "plane_indices_emS0": a["ys"], "plane_physical_nm": [round(y * g["em"][1], 3) for y in a["ys"]],
            "spacing_nm_min": g["z_spacing_nm"], "spacing_nm_baseline": Z_SPACING_BASELINE_NM,
            "plane_dense_only": a["dense"], "padded_to": [CAP, CAP],
            "annotation_bbox_in_tile_px": {"z": [g["annz"][0]-z0, g["annz"][1]-z0],
                                           "x": [g["annx"][0]-x0, g["annx"][1]-x0]}}


def process_crop(folder, crop, z_spacing_nm=None):
    if z_spacing_nm is None:
        z_spacing_nm = Z_SPACING_NM
    d = json.load(open(os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "_work", "oo_gt_out", f"{folder}.json")))
    cinfo = next(c for c in d["crops"] if c["crop"] == crop)
    cdir = os.path.join(OUTROOT, folder, crop)
    mfp = os.path.join(cdir, "crop_manifest.json")
    if not os.path.exists(mfp):
        return "no-manifest"
    m = json.load(open(mfp))
    pxy, pxz = os.path.join(cdir, "raw_xy.tif"), os.path.join(cdir, "raw_xz.tif")
    # Resume-skip only if both orientations already exist AT THE REQUESTED spacing; a spacing change
    # (e.g. 200 -> 100) forces a rebuild since the plane count/markers differ.
    cur_xy = (m.get("raw_xy") or {}).get("spacing_nm_min")
    cur_xz = (m.get("raw_xz") or {}).get("spacing_nm_min")
    same_spacing = (cur_xy == z_spacing_nm and cur_xz == z_spacing_nm)
    if (m.get("raw_xy") and m.get("raw_xz") and os.path.getsize(pxy) > 0
            and os.path.getsize(pxz) > 0 and same_spacing):
        return "skip"
    g = geom(d, cinfo, z_spacing_nm=z_spacing_nm)
    emz, emy, emx = g["em"]

    # reuse a prior single-orientation raw.tif as the orientation its manifest declares
    oldraw = os.path.join(cdir, "raw.tif")
    if os.path.exists(oldraw):
        is_xz = "y_plane_indices_emS0" in m.get("raw_crop", {})
        dst = pxz if is_xz else pxy
        if not os.path.exists(dst):
            os.rename(oldraw, dst)
        else:
            os.remove(oldraw)
    m.pop("raw_crop", None)

    em_url = to_https(DSETS[folder]["url"])
    arr = None
    chunks = jget(f"{em_url}/s0/.zarray")[0]["chunks"]

    if tiff_shape(pxy) != g["xy"]["shape"]:
        arr = arr or open_zarr(f"{em_url}/s0")
        save_tiff(pxy, read_xy(arr, chunks[0], g))
    if tiff_shape(pxz) != g["xz"]["shape"]:
        arr = arr or open_zarr(f"{em_url}/s0")
        save_tiff(pxz, read_xz(arr, chunks[1], g))

    m["raw_xy"] = xy_entry(g, emy, emx)
    m["raw_xz"] = xz_entry(g, emz, emx)
    # keep segmentations; remove orientation-specific fields replaced by raw_xy/raw_xz
    for s in m.get("segmentations", []):
        s.pop("location_in_raw_crop_px", None)
        s["annotation_bbox_in_original_voxels_emS0"] = {"z": g["annz"], "y": g["anny"], "x": g["annx"]}
        s["note"] = "3D label crop shared by both raw orientations; align via physical_origin_nm + resolution"
    json.dump(m, open(mfp, "w"), indent=2)
    return f"ok xy={g['xy']['shape']} xz={g['xz']['shape']}"


if __name__ == "__main__":
    z_spacing, rest = resolve_z_spacing(sys.argv[1:])
    print(f"[oo_tiles_both] z_spacing_nm={z_spacing} (baseline {Z_SPACING_BASELINE_NM})", flush=True)
    for folder in (rest or sorted(CC.keys())):
        for c in CC[folder]["crops"]:
            if not (_has(c["classes"], MITO) or _has(c["classes"], ER) or _has(c["classes"], LD)):
                continue
            try:
                print(f"{process_crop(folder, c['crop'], z_spacing_nm=z_spacing):<34} {folder}/{c['crop']}", flush=True)
            except Exception as e:
                print(f"FAIL {folder}/{c['crop']}: {type(e).__name__}: {str(e)[:110]}", flush=True)
