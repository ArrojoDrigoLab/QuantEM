"""webKnossos FAST-EM (TU Delft) — human-traced + 2nd-annotator-proofread mito GT.
Anonymous OME-Zarr stream. GT label layer 'Mito_GT_EM_realigned_SOFIMA' (uint32 instance),
paired with its co-registered color layer (same bbox). MitoNet layers excluded.
Efficiency: read full-z slab per 4096 tile (chunks are 32 deep) and skip tiles with empty GT.
"""
import os, sys, json, urllib.request
import numpy as np
import tensorstore as ts
sys.path.insert(0, os.path.dirname(__file__))
import seg_crop as sc

HOST = "https://webknossos.tnw.tudelft.nl"
OUT = os.path.join(os.environ.get("SEG_CORPUS_ROOT", "."), "webknossos_fastem_mito")
DSETS = [("20230626_Rat_Pancreas_OTO", "6672df47010000c400a1be49"),
         ("20230711_MCF7_UAc", "6672e7da010000f100a1be53")]

META = {
    "name": "webknossos_fastem_mito",
    "source_repo": "webKnossos (TU Delft Hoogenboom group)",
    "accession": "Rat_Pancreas_OTO + MCF7_UAc",
    "doi": "10.1515/mim-2024-0005", "license": "see Kievits et al. 2024",
    "paper": "Kievits et al., FAST-EM array tomography, Methods in Microscopy 2024 (PMC11308914)",
    "gt_provenance": "manual WebKnossos annotation, proofread by a 2nd annotator; GT layer distinct from MitoNet predictions (excluded)",
    "modality": "FAST-EM array tomography (3D)", "dimensionality": "3D",
    "voxel_size_nm": {"x": 4, "y": 4, "z": 100},
    "label_encoding": "instance (mitochondria, uint32)",
    "organelle_classes": {"nonzero": "mitochondria"},
    "z_rule": "z=100 nm -> every 4th plane (>=400 nm)",
    "alignment": "1:1 (GT label + realigned color layer share bbox)",
    "source_url": HOST,
    "notes": "Streamed anonymously via OME-Zarr; *MitoNet* prediction layers excluded.",
}


def open_arr(dsid, layer, mag="1"):
    url = f"{HOST}/data/zarr/{dsid}/{layer}/{mag}/"
    return ts.open({"driver": "zarr", "kvstore": {"driver": "http", "base_url": url},
                    "open": True}).result()


def pick_layers(props):
    color = [L for L in props["dataLayers"] if L.get("category") == "color"]
    segs = [L for L in props["dataLayers"] if L.get("category") == "segmentation"]
    gt = [L for L in segs if "gt" in L["name"].lower() and "mitonet" not in L["name"].lower()]
    gt.sort(key=lambda L: ("realigned" not in L["name"].lower(), len(L["name"])))
    gtL = gt[0]
    tl = tuple(gtL["boundingBox"]["topLeft"])
    col = [L for L in color if tuple(L["boundingBox"]["topLeft"]) == tl]
    colL = col[0] if col else sorted(color, key=lambda L: "realigned" not in L["name"].lower())[0]
    return colL, gtL


SCAN_FILE = os.path.join(OUT, "_scanned.json")


def load_scanned():
    if os.path.exists(SCAN_FILE):
        return set(json.load(open(SCAN_FILE)))
    return set()


def save_scanned(s):
    json.dump(sorted(s), open(SCAN_FILE, "w"))


def purge_orphans():
    import glob
    m = json.load(open(os.path.join(OUT, "manifest.json")))
    keep = set()
    for c in m["crops"]:
        keep.add(os.path.normpath(os.path.join(OUT, c["em_file"])))
        keep.add(os.path.normpath(os.path.join(OUT, c["label_file"])))
    rm = 0
    for f in glob.glob(os.path.join(OUT, "crops", "**", "*.tif"), recursive=True):
        if os.path.normpath(f) not in keep:
            os.remove(f); rm += 1
    return rm


def gt_footprint_tiles(dsid, gtname, x0, y0, z0, w, h, depth, coarse=32):
    """Read GT at a coarse XY mag (cheap) -> set of native (ax0,ay0) tiles containing GT."""
    gc = open_arr(dsid, gtname, mag=f"{coarse}-{coarse}-1")
    cx0, cy0 = x0 // coarse, y0 // coarse
    cw = -(-w // coarse); ch = -(-h // coarse)
    cz1 = min(z0 + depth, gc.shape[3])
    blk = np.asarray(gc[0, cx0:cx0 + cw, cy0:cy0 + ch, z0:cz1].read().result())  # (cw,ch,z)
    rsz = 4096 // coarse
    tiles = set()
    for ty in sc.tile_starts(h):
        for tx in sc.tile_starts(w):
            rx0, ry0 = tx // coarse, ty // coarse
            if blk[rx0:rx0 + rsz, ry0:ry0 + rsz, :].any():
                tiles.add((x0 + tx, y0 + ty))
    return tiles


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = sc.Dataset(OUT, META, resume=True)   # kill-safe: resume from prior manifest
    scanned = load_scanned()
    zstep = sc.zstep_for_spacing(100)
    for dsname, dsid in DSETS:
        props = json.load(urllib.request.urlopen(f"{HOST}/data/zarr/{dsid}/datasource-properties.json", timeout=60))
        colL, gtL = pick_layers(props)
        bb = gtL["boundingBox"]; x0, y0, z0 = bb["topLeft"]; w, h, depth = bb["width"], bb["height"], bb["depth"]
        em = open_arr(dsid, colL["name"]); gt = open_arr(dsid, gtL["name"])
        full_x, full_y = em.shape[1], em.shape[2]
        try:
            gtset = gt_footprint_tiles(dsid, gtL["name"], x0, y0, z0, w, h, depth)
            ftxt = f"{len(gtset)} GT-containing tiles"
        except Exception as ex:
            gtset = None
            ftxt = f"footprint failed ({type(ex).__name__}); processing all tiles + native empty-check"
        print(f"{dsname}: color={colL['name']} gt={gtL['name']} bb=({x0},{y0}) {w}x{h}x{depth} | {ftxt}", flush=True)
        for ty in sc.tile_starts(h):
            for tx in sc.tile_starts(w):
                ax0, ay0 = x0 + tx, y0 + ty
                key = f"{dsname}|{ax0}|{ay0}"
                if key in scanned:
                    continue
                if gtset is not None and (ax0, ay0) not in gtset:  # footprint says empty -> skip native read
                    scanned.add(key); save_scanned(scanned); continue
                gblk = np.asarray(gt[0, ax0:ax0 + 4096, ay0:ay0 + 4096, z0:z0 + depth].read().result())
                if gblk.any():
                    eblk = np.asarray(em[0, ax0:ax0 + 4096, ay0:ay0 + 4096, z0:z0 + depth].read().result())
                    for zi in range(0, depth, zstep):
                        gp = gblk[:, :, zi].T
                        if not gp.any():
                            continue
                        ds.add_plane(eblk[:, :, zi].T, gp, source_image=f"{dsname}/{gtL['name']}",
                                     source_shape_xy=(full_x, full_y), z_index=int(z0 + zi),
                                     z_physical_nm=float((z0 + zi) * 100), voxel_size_nm={"x": 4, "y": 4, "z": 100},
                                     label_kind="instance_single", organelle_name="mitochondria",
                                     subdir=dsname, id_prefix=f"{dsname[9:13]}_", origin_xy=(ax0, ay0))
                scanned.add(key)
                save_scanned(scanned)
                ds.write_manifest()    # persist after every tile (kill-safe)
            print(f"  {dsname} row ty={ty} done; crops so far {len(ds.crops)}", flush=True)
        del em, gt
    rm = purge_orphans()
    path, n = ds.write_manifest()
    print(f"DONE: {n} crops ({rm} orphans purged) -> {path}", flush=True)


if __name__ == "__main__":
    main()
