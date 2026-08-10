"""Analyze one OpenOrganelle/COSEM dataset: full EM dims + ground-truth crop
coverage + storage for full XY planes of annotated z-slices.

Shared helpers; the entry point is oo_gt_batch.py
  em_array_url is the DB download_url, e.g.
  s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8

Outputs a single JSON object on stdout.
"""
import sys, json, re
import requests, xml.etree.ElementTree as ET

H = {"User-Agent": "quantem-dataset-build/1.0"}
HOST = "https://janelia-cosem-datasets.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
DTYPE_BYTES = {"|u1": 1, "|i1": 1, "<u1": 1, "<u2": 2, "<i2": 2, "|u2": 2,
               "<f4": 4, "<f8": 8, ">u2": 2, ">i2": 2}


def to_https(url):
    url = url.strip()
    if url.startswith("s3://janelia-cosem-datasets/"):
        return HOST + "/" + url[len("s3://janelia-cosem-datasets/"):]
    if "open.quiltdata.com/b/janelia-cosem-datasets/tree/" in url:
        tail = url.split("tree/", 1)[1]
        return HOST + "/" + tail
    return url


def key_of(https_url):
    return https_url[len(HOST) + 1:]


def get(url):
    try:
        r = requests.get(url, timeout=40, headers=H)
        return r.status_code, r.text
    except Exception as e:
        return None, f"ERR {type(e).__name__}: {e}"


def s3_list(prefix, delimiter="/"):
    cps, keys, token = [], [], None
    for _ in range(20):
        url = f"{HOST}/?list-type=2&prefix={prefix}&delimiter={delimiter}"
        if token:
            url += f"&continuation-token={requests.utils.quote(token, safe='')}"
        r = requests.get(url, timeout=40, headers=H)
        if r.status_code != 200:
            return None, None, r.status_code
        root = ET.fromstring(r.text)
        cps += [e.find(NS + "Prefix").text for e in root.findall(NS + "CommonPrefixes")]
        keys += [(e.find(NS + "Key").text, int(e.find(NS + "Size").text))
                 for e in root.findall(NS + "Contents")]
        trunc = root.findtext(NS + "IsTruncated") == "true"
        token = root.findtext(NS + "NextContinuationToken")
        if not (trunc and token):
            break
    return cps, keys, 200


def read_multiscale(base):
    """Return (axes, datasets) where each dataset = {path, scale[zyx], translation[zyx]}.
    Handles zarr v2 (.zattrs/multiscales). Returns (None, None) if not found."""
    st, txt = get(f"{base}/.zattrs")
    if st != 200:
        return None, None
    try:
        za = json.loads(txt)
    except Exception:
        return None, None
    ms = (za.get("multiscales") or [None])[0]
    if not ms:
        return None, None
    axes = [a.get("name") if isinstance(a, dict) else a for a in ms.get("axes", [])]
    out = []
    for d in ms.get("datasets", []):
        sc = tr = None
        for ct in d.get("coordinateTransformations", []):
            if ct.get("type") == "scale":
                sc = ct.get("scale")
            elif ct.get("type") == "translation":
                tr = ct.get("translation")
        out.append({"path": d.get("path"), "scale": sc, "translation": tr or [0, 0, 0]})
    return axes, out


def read_zarray(base, path):
    st, txt = get(f"{base}/{path}/.zarray")
    if st != 200:
        return None
    try:
        m = json.loads(txt)
        return {"shape": m["shape"], "dtype": m["dtype"]}
    except Exception:
        return None


def analyze(em_url):
    https = to_https(em_url).rstrip("/")
    res = {"em_url": em_url, "https": https}
    store_type = "zarr" if ".zarr" in https else ("n5" if ".n5" in https else
                 ("precomputed" if "precomputed" in https or "neuroglancer" in https else "?"))
    res["store_type"] = store_type

    # recon root = parent of /em/
    m = re.search(r"^(.*?)/em/", https)
    recon_root = m.group(1) if m else https.rsplit("/", 2)[0]
    res["recon_root"] = recon_root

    # --- ground-truth crops (works for any store: pure S3 key listing) ---
    gt_prefix = key_of(recon_root) + "/labels/groundtruth/"
    crop_dirs, _, st = s3_list(gt_prefix)
    res["gt_prefix"] = gt_prefix
    res["gt_list_status"] = st
    crop_dirs = crop_dirs or []
    res["n_crops"] = len(crop_dirs)

    if store_type != "zarr":
        # Non-zarr stores in this catalog all have 0 GT crops; still report full size.
        res["note"] = f"store_type={store_type}; size only (0 GT crops expected)"
        res["crops"] = [c.split('/')[-2] for c in crop_dirs]
        dims = dtype = scale = None
        if store_type == "n5":
            st, txt = get(f"{https}/s0/attributes.json")
            if st == 200:
                a = json.loads(txt)
                dims = a.get("dimensions")
                dt = a.get("dataType", "uint8")
                dtype = {"uint8": 1, "uint16": 2, "int16": 2, "int8": 1}.get(dt, 1)
            st, txt = get(f"{https}/attributes.json")
            if st == 200:
                ms = (json.loads(txt).get("multiscales") or [{}])[0]
                d0 = (ms.get("datasets") or [{}])[0]
                scale = (d0.get("transform") or {}).get("scale")
        elif store_type == "precomputed":
            st, txt = get(f"{https}/info")
            if st == 200:
                info = json.loads(txt)
                s0i = info["scales"][0]
                dims = s0i.get("size")  # x,y,z
                scale = list(reversed(s0i.get("resolution", [])))
                dt = info.get("data_type", "uint8")
                dtype = {"uint8": 1, "uint16": 2, "int16": 2}.get(dt, 1)
        if dims:
            vox = 1
            for d in dims:
                vox *= int(d)
            res["em_dims_raw"] = dims
            res["em_scale_nm"] = scale
            res["em_itemsize"] = dtype or 1
            res["em_full_voxels"] = vox
            res["em_full_bytes_native"] = vox * (dtype or 1)
            res["em_full_bytes_8bit"] = vox * 1
        res["annotated_z_slices"] = 0
        res["storage_bytes_8bit"] = 0
        res["storage_bytes_native"] = 0
        return res

    # --- EM full dimensions ---
    axes, dsets = read_multiscale(https)
    if not dsets:
        res["error"] = "no EM multiscale .zattrs"
        return res
    s0 = dsets[0]
    za = read_zarray(https, s0["path"])
    if not za:
        res["error"] = "no EM s0 .zarray"
        return res
    em_shape = za["shape"]            # (z, y, x)
    em_dtype = za["dtype"]
    em_scale = s0["scale"]            # (z, y, x) nm
    em_trans = s0["translation"]      # (z, y, x) nm
    Z, Y, X = em_shape
    itemsize = DTYPE_BYTES.get(em_dtype, 1)
    res.update({
        "em_shape_zyx": em_shape,
        "em_scale_zyx_nm": em_scale,
        "em_dtype": em_dtype,
        "em_itemsize": itemsize,
        "em_full_voxels": Z * Y * X,
        "em_full_bytes_native": Z * Y * X * itemsize,
        "em_full_bytes_8bit": Z * Y * X * 1,
    })

    if res["n_crops"] == 0:
        res["annotated_z_slices"] = 0
        res["gt_z_fraction"] = 0.0
        res["storage_bytes_8bit"] = 0
        res["storage_bytes_native"] = 0
        res["gt_bbox_voxels_em"] = 0
        return res

    # --- per-crop bounding boxes -> EM z-slice union ---
    annotated_z = set()
    crops_info = []
    gt_bbox_voxels_em = 0  # sum of crop bbox volume expressed in EM s0 voxels
    em_sz = em_scale[0]
    for cd in crop_dirs:
        crop_name = cd.split('/')[-2]
        ckey = key_of(recon_root) + f"/labels/groundtruth/{crop_name}/"
        # choose a class subgroup: prefer 'all', else first available
        cls_dirs, _, _ = s3_list(ckey)
        cls_dirs = cls_dirs or []
        cls_names = [c.split('/')[-2] for c in cls_dirs]
        pick = "all" if "all" in cls_names else (cls_names[0] if cls_names else None)
        if not pick:
            crops_info.append({"crop": crop_name, "error": "no class subgroup"})
            continue
        cbase = f"{recon_root}/labels/groundtruth/{crop_name}/{pick}"
        caxes, cdsets = read_multiscale(cbase)
        if not cdsets:
            crops_info.append({"crop": crop_name, "class": pick, "error": "no crop multiscale"})
            continue
        cs0 = cdsets[0]
        cza = read_zarray(cbase, cs0["path"])
        if not cza:
            crops_info.append({"crop": crop_name, "class": pick, "error": "no crop s0 .zarray"})
            continue
        csh = cza["shape"]            # (z, y, x) voxels
        csc = cs0["scale"]            # (z, y, x) nm
        ctr = cs0["translation"]      # (z, y, x) nm
        # physical z extent (nm)
        z0_nm = ctr[0]
        z1_nm = ctr[0] + csh[0] * csc[0]
        # map to EM s0 z indices
        ez0 = (z0_nm - em_trans[0]) / em_sz
        ez1 = (z1_nm - em_trans[0]) / em_sz
        import math
        a = max(0, math.floor(ez0))
        b = min(Z, math.ceil(ez1))
        for z in range(a, b):
            annotated_z.add(z)
        # crop bbox volume in EM voxels (physical nm^3 / EM voxel nm^3)
        phys_vol = (csh[0] * csc[0]) * (csh[1] * csc[1]) * (csh[2] * csc[2])
        em_voxel_nm3 = em_scale[0] * em_scale[1] * em_scale[2]
        gt_bbox_voxels_em += phys_vol / em_voxel_nm3
        crops_info.append({
            "crop": crop_name, "class": pick, "shape_zyx": csh, "scale_zyx_nm": csc,
            "trans_zyx_nm": ctr, "em_z_range": [a, b], "em_z_span": b - a,
        })

    n_annot_z = len(annotated_z)
    plane_bytes_8bit = Y * X * 1
    res.update({
        "n_crops": res["n_crops"],
        "crops": crops_info,
        "annotated_z_slices": n_annot_z,
        "gt_z_fraction": round(n_annot_z / Z, 5) if Z else 0,
        "gt_bbox_voxels_em": int(gt_bbox_voxels_em),
        "gt_bbox_volume_fraction": round(gt_bbox_voxels_em / (Z * Y * X), 6) if Z * Y * X else 0,
        "plane_bytes_8bit": plane_bytes_8bit,
        "storage_bytes_8bit": n_annot_z * plane_bytes_8bit,
        "storage_bytes_native": n_annot_z * Y * X * itemsize,
    })
    return res


if __name__ == "__main__":
    out = analyze(sys.argv[1])
    print(json.dumps(out))
