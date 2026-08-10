"""OpenOrganelle 3D-seg -> 2D-plane alignment.

Each OOSample carries the in-plane source nm/px (``src_nm_row`` / ``src_nm_col``) so the
canonical-resample step downstream knows the native pixel size of the derived EM plane (raw tiles are
at the EM S0 resolution).

Each crop has dual-orientation raw tiles (raw_xy / raw_xz) sharing one 3D seg volume at its own native
resolution. Alignment of raw to seg is in physical nanometres (never integer voxels), with a
nearest-neighbour resize of seg -> EM grid (labels must not interpolate), and each (orientation, plane)
is treated as an independent 2D sample. seg_er==255 ("unknown") -> ignore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import read_json, read_planes, read_tif

_AXIS = {"z": 0, "y": 1, "x": 2}
_OO_UNKNOWN = 255  # seg_er value meaning "unknown" -> ignore


@dataclass
class OOSample:
    orientation: str  # "raw_xy" | "raw_xz"
    plane_k: int
    raw_plane: np.ndarray  # (rows, cols) uint8 — the full clamped EM plane
    win_rows: tuple[int, int]  # annotation window (row0, row1) in plane px
    win_cols: tuple[int, int]  # annotation window (col0, col1) in plane px
    fg: np.ndarray  # bool, shape (win_rows, win_cols) on the EM grid — organelle foreground
    unknown: np.ndarray  # bool, same shape — seg "unknown" (255) -> ignore
    seg_slice: int
    inst: np.ndarray | None = None  # instance ids on the window (mito only; seg_mito is instance)
    src_nm_row: float | None = None  # native nm/px along the plane's row axis (the segmentation resample input)
    src_nm_col: float | None = None  # native nm/px along the plane's col axis (always 'x')
    oo_z_spacing_nm: float | None = None  # crop reslice spacing (nm); 200 baseline, 100 when densified
    plane_z_nm: float | None = None  # physical z (nm) of this plane along its sampling axis
    oo_dense_z: bool = False  # True = plane exists only at <baseline spacing (drop -> exact 200nm subset)


def _resize_nn(seg2d: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a 2D label slice to ``out_hw`` (no interpolation of label ids)."""
    h, w = out_hw
    sh, sw = seg2d.shape
    if (sh, sw) == (h, w):
        return seg2d
    ys = np.clip((np.arange(h) * sh) // max(h, 1), 0, sh - 1)
    xs = np.clip((np.arange(w) * sw) // max(w, 1), 0, sw - 1)
    return seg2d[ys][:, xs]


def iter_oo_samples(crop_dir: str | Path, organelle: str) -> list[OOSample]:
    """Yield aligned (EM-plane, foreground, unknown) samples for one OO crop, both orientations.

    ``organelle`` in {"mito","er"}. Mito (instance) -> ``seg>0``; ER (semantic) -> ``seg==1`` with
    ``seg==255`` flagged unknown.
    """
    crop_dir = Path(crop_dir)
    man = read_json(crop_dir / "crop_manifest.json")
    em_res = man["original_image"]["resolution_nm_zyx"]  # [z,y,x] nm/px of the raw EM
    segs = {s["class_name"]: s for s in man.get("segmentations", [])}
    seg_name = "mito" if organelle == "mito" else "er"
    if seg_name not in segs:
        return []
    seg_meta = segs[seg_name]
    seg_path = crop_dir / seg_meta["file"]
    if not seg_path.exists():
        return []
    seg_vol = np.asarray(read_tif(seg_path))  # (z, y, x)
    seg_origin = seg_meta["physical_origin_nm_zyx"]
    seg_res = seg_meta["resolution_nm_zyx"]

    out: list[OOSample] = []
    for orient in ("raw_xy", "raw_xz"):
        rw = man.get(orient)
        if not rw:
            continue
        raw_path = crop_dir / rw["file"]
        if not raw_path.exists():
            continue
        raw = read_planes(raw_path)  # (planes, rows, cols)
        sax = rw["sample_axis"]
        ai = _AXIS[sax]
        rows_ax, _cols_ax = rw["tile_axes_rows_cols"]
        # In-plane native nm/px of the derived EM plane: rows along tile row-axis, cols along 'x'.
        src_nm_row = float(em_res[_AXIS[rows_ax]])
        src_nm_col = float(em_res[_AXIS["x"]])
        bbox = rw["annotation_bbox_in_tile_px"]
        a0, a1 = bbox[rows_ax]
        c0, c1 = bbox["x"]
        a0, a1 = int(a0), int(a1)
        c0, c1 = int(c0), int(c1)
        # Densification metadata (optional: a manifest without these fields falls back to the
        # baseline 200nm spacing and no dense-only planes).
        oo_z_spacing_nm = float(rw.get("spacing_nm_min", 200.0))
        dense_flags = rw.get("plane_dense_only") or []
        for k, plane_nm in enumerate(rw["plane_physical_nm"]):
            if k >= raw.shape[0]:
                break
            is_dense = bool(dense_flags[k]) if k < len(dense_flags) else False
            raw_plane = raw[k]
            # clamp window to the actual plane extent (tiles are clamped, not padded)
            ww0, ww1 = max(a0, 0), min(a1, raw_plane.shape[0])
            wc0, wc1 = max(c0, 0), min(c1, raw_plane.shape[1])
            if ww1 <= ww0 or wc1 <= wc0:
                continue
            center = float(plane_nm) + 0.5 * float(em_res[ai])
            si = int(math.floor((center - float(seg_origin[ai])) / float(seg_res[ai])))
            si = min(max(si, 0), seg_vol.shape[ai] - 1)
            if ai == 0:
                seg2d = seg_vol[si, :, :]
            elif ai == 1:
                seg2d = seg_vol[:, si, :]
            else:
                seg2d = seg_vol[:, :, si]
            # seg2d's footprint is the full (unclamped) annotation window; resize to that full grid
            # first, then crop to the clamped sub-window (else labels shift/scale at the tile edge).
            seg_full = _resize_nn(seg2d, (a1 - a0, c1 - c0))
            seg_em = seg_full[ww0 - a0:ww1 - a0, wc0 - c0:wc1 - c0]
            if organelle == "er":
                fg = seg_em == 1
                unknown = seg_em == _OO_UNKNOWN
                inst = None
            else:
                fg = seg_em > 0
                unknown = np.zeros_like(fg, dtype=bool)
                inst = seg_em.astype(np.int32) * fg  # seg_mito ids are already instances
            out.append(
                OOSample(
                    orientation=orient,
                    plane_k=k,
                    raw_plane=raw_plane,
                    win_rows=(ww0, ww1),
                    win_cols=(wc0, wc1),
                    fg=fg,
                    unknown=unknown,
                    seg_slice=si,
                    inst=inst,
                    src_nm_row=src_nm_row,
                    src_nm_col=src_nm_col,
                    oo_z_spacing_nm=oo_z_spacing_nm,
                    plane_z_nm=float(plane_nm),
                    oo_dense_z=is_dense,
                )
            )
    return out
