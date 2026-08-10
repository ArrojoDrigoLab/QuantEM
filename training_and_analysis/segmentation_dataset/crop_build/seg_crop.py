"""
seg_crop.py — core utilities for the ground-truth EM segmentation crop collection.

Output spec:
- Every crop is a TILE x TILE (default 4096) canvas.
- 3D volumes: sample z-planes >= 400 nm apart.
- Grid-tile the full plane; keep a tile iff it contains >= 1 in-scope-organelle voxel.
- Sources smaller than TILE in an axis are centered in the canvas; the surrounding
  pad is zeros (there is no real EM beyond the source).  Real EM "context" is the
  genuine pixels that fall inside the tile window around the annotation.
- Labels keep their native encoding (instance vs semantic); never interpolated.
- Per-dataset manifest.json records, per crop: which source file/plane it came from,
  the source bbox it was cut from, where the real EM sits inside the canvas
  (valid_region), the padding, the annotation bbox, and the organelles present.

This module is dataset-agnostic.  Per-dataset driver scripts build (em_plane,
label_plane) arrays and call add_plane().
"""
import os, json, math, hashlib
import numpy as np
import tifffile

TILE = 4096


# --------------------------------------------------------------------------- #
# tiling geometry
# --------------------------------------------------------------------------- #
def tile_starts(size, tile=TILE):
    """Start coords along one axis. None => source smaller than tile (center-pad)."""
    if size <= tile:
        return [None]
    starts = list(range(0, size - tile + 1, tile))
    if starts[-1] + tile < size:          # flush a final tile to the far edge
        starts.append(size - tile)
    return starts


def _place_axis(size, start, tile=TILE):
    """Return (src0, src1, dst0, dst1, pad_before, pad_after)."""
    if start is None:                      # center the (sub-tile) source
        off = (tile - size) // 2
        return 0, size, off, off + size, off, tile - (off + size)
    return start, start + tile, 0, tile, 0, 0


def zstep_for_spacing(z_nm, min_nm=400):
    """Plane stride so that kept planes are >= min_nm apart."""
    if not z_nm or z_nm <= 0:
        return 1
    return max(1, math.ceil(min_nm / float(z_nm)))


# --------------------------------------------------------------------------- #
# label introspection
# --------------------------------------------------------------------------- #
def organelles_in_window(lbl, label_kind, organelle_name=None,
                         class_map=None, organelle_values=None):
    """
    Returns (has_organelle, summary_dict).
    label_kind:
      'instance_single' : nonzero = instances of one organelle (organelle_name)
      'semantic'        : pixel value -> class via class_map {int:name};
                          organelle_values = set of int values that are in-scope organelles
    """
    if label_kind == 'instance_single':
        ids = np.unique(lbl)
        ids = ids[ids != 0]
        if ids.size == 0:
            return False, {}
        return True, {organelle_name: {"instances": int(ids.size),
                                       "area_px": int(np.count_nonzero(lbl))}}
    elif label_kind == 'semantic':
        vals, counts = np.unique(lbl, return_counts=True)
        summary = {}
        has = False
        for v, c in zip(vals.tolist(), counts.tolist()):
            if v == 0:
                continue
            name = class_map.get(v, f"value_{v}")
            is_org = (organelle_values is None) or (v in organelle_values)
            summary[name] = {"area_px": int(c), "value": int(v), "is_organelle": bool(is_org)}
            if is_org:
                has = True
        return has, summary
    else:
        raise ValueError(label_kind)


def annotation_bbox(lbl, organelle_values=None):
    """xyxy bbox of organelle voxels inside a canvas-sized label (or None)."""
    if organelle_values is None:
        mask = lbl != 0
    else:
        mask = np.isin(lbl, list(organelle_values))
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# --------------------------------------------------------------------------- #
# collector
# --------------------------------------------------------------------------- #
class Dataset:
    def __init__(self, out_dir, dataset_meta, tile=TILE, fresh=True, resume=False):
        self.out_dir = out_dir
        self.crops_dir = os.path.join(out_dir, "crops")
        self.tile = tile
        self.meta = dataset_meta
        self.crops = []
        self._n = 0
        mpath = os.path.join(out_dir, "manifest.json")
        if resume and os.path.exists(mpath):
            m = json.load(open(mpath))
            self.crops = m.get("crops", [])
            ns = [int(c["crop_id"][-5:]) for c in self.crops if c["crop_id"][-5:].isdigit()]
            self._n = (max(ns) + 1) if ns else 0
            os.makedirs(self.crops_dir, exist_ok=True)
            return
        if fresh and os.path.isdir(self.crops_dir):
            import shutil as _sh
            _sh.rmtree(self.crops_dir, ignore_errors=True)
        os.makedirs(self.crops_dir, exist_ok=True)

    def add_plane(self, em_plane, label_plane, *, source_image, source_shape_xy,
                  z_index=None, z_physical_nm=None, voxel_size_nm=None,
                  label_kind, organelle_name=None, class_map=None,
                  organelle_values=None, subdir=None, id_prefix="", origin_xy=(0, 0)):
        """Grid-tile one 2D plane; write kept crops; append manifest records.

        origin_xy: global (x,y) offset added to recorded source_bbox, for callers that
        pre-tile a large image into sub-blocks (e.g. streamed zarr). Tiling still
        operates on the passed array; only the recorded coordinates are offset."""
        ox, oy = origin_xy
        assert em_plane.shape == label_plane.shape, \
            f"EM {em_plane.shape} != label {label_plane.shape}"
        H, W = em_plane.shape
        tile = self.tile
        crops_subdir = self.crops_dir if not subdir else os.path.join(self.crops_dir, subdir)
        os.makedirs(crops_subdir, exist_ok=True)

        for ys in tile_starts(H, tile):
            for xs in tile_starts(W, tile):
                sy0, sy1, dy0, dy1, pt, pb = _place_axis(H, ys, tile)
                sx0, sx1, dx0, dx1, pl, pr = _place_axis(W, xs, tile)
                lwin = label_plane[sy0:sy1, sx0:sx1]
                has, summary = organelles_in_window(
                    lwin, label_kind, organelle_name, class_map, organelle_values)
                if not has:
                    continue

                em = np.zeros((tile, tile), dtype=em_plane.dtype)
                lb = np.zeros((tile, tile), dtype=label_plane.dtype)
                em[dy0:dy1, dx0:dx1] = em_plane[sy0:sy1, sx0:sx1]
                lb[dy0:dy1, dx0:dx1] = lwin

                # valid_region = the real EM (non-zero) extent inside the canvas. Normally this equals
                # the placed-source window [dx0:dx1, dy0:dy1], but some sources store planes zero-padded
                # around the tissue (e.g. MitoEM nii.gz bounding boxes: float32 0 -> uint8 0), so trim to
                # the actual non-zero pixels. valid_region must never claim EM where pixels are padding:
                # outside valid_region == zero-pad == not real EM.
                _nz = em > 0
                if _nz.any():
                    _ys, _xs = np.where(_nz)
                    vx0, vy0, vx1, vy1 = int(_xs.min()), int(_ys.min()), int(_xs.max()) + 1, int(_ys.max()) + 1
                else:
                    vx0, vy0, vx1, vy1 = dx0, dy0, dx1, dy1

                ann = annotation_bbox(lb, organelle_values)
                cid = f"{id_prefix}{self._n:05d}"
                self._n += 1
                rel = subdir + "/" if subdir else ""
                em_rel = f"crops/{rel}{cid}_em.tif"
                lb_rel = f"crops/{rel}{cid}_label.tif"
                tifffile.imwrite(os.path.join(self.out_dir, em_rel), em, compression="zlib")
                tifffile.imwrite(os.path.join(self.out_dir, lb_rel), lb, compression="zlib")

                padded = bool(pl or pr or pt or pb)
                cov = ((vx1 - vx0) * (vy1 - vy0)) / float(tile * tile)  # real-EM (non-zero) coverage
                tier = "full" if cov >= 0.999 else ("partial" if cov >= 0.25 else "sparse")
                self.crops.append({
                    "crop_id": cid,
                    "em_file": em_rel,
                    "label_file": lb_rel,
                    "source_image": source_image,
                    "source_image_size_xy": list(source_shape_xy),
                    "source_bbox_xyxy": [ox + sx0, oy + sy0, ox + sx1, oy + sy1],
                    "z_index": z_index,
                    "z_physical_nm": z_physical_nm,
                    "valid_region_in_canvas_xyxy": [vx0, vy0, vx1, vy1],
                    "coverage_fraction": round(cov, 4),
                    "coverage_tier": tier,
                    "padding": {"left": pl, "top": pt, "right": pr, "bottom": pb,
                                "fill": "zeros" if padded else "context"},
                    "annotation_bbox_in_canvas_xyxy": ann,
                    "organelles_present": summary,
                    "voxel_size_nm": voxel_size_nm,
                    "em_dtype": str(em.dtype),
                    "label_dtype": str(lb.dtype),
                })
        return

    def write_manifest(self):
        manifest = {"dataset": self.meta,
                    "n_crops": len(self.crops),
                    "tile_size": self.tile,
                    "crops": self.crops}
        path = os.path.join(self.out_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        return path, len(self.crops)
