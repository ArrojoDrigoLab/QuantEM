"""Unified extraction: one source crop -> derived (EM, canonical-mask) 2D samples.

Handles both collections behind one return type (``DerivedSample``):
  * gt canvas crops: crop to the real-EM field of view around the annotation -- ``valid_region``
    under dense/full coverage, the whole non-zero EM extent under partial/sparse coverage (where
    context-filled crops carry real source EM across the canvas, so a small annotation in a large
    EM image keeps its real context) -- foreground via ``canonicalize``, ignore outside
    ``annotation_bbox`` for partial/sparse coverage (true background for ``full``).
  * openOrganelle: per (orientation, plane), via the physical-nm alignment in ``openorganelle``.

Mask encoding: 0=background, 1=foreground, 255=ignore (see constants).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..constants import BACKGROUND, FOREGROUND, IGNORE_INDEX
from .canonicalize import canonical_mask
from .io import read_planes, read_tif
from .openorganelle import iter_oo_samples
from .splits import CropRow

@dataclass
class DerivedSample:
    sample_id: str
    em: np.ndarray  # (H, W) uint8
    mask: np.ndarray  # (H, W) uint8 in {0,1,255}
    extra: dict = field(default_factory=dict)
    inst: np.ndarray | None = None  # (H, W) int32 instance ids (mito only; 0=bg/ignore)
    gt_is_instance: bool = False  # True = real instance ids; False = connected-components pseudo

    @property
    def valid_px(self) -> int:
        return int((self.mask != IGNORE_INDEX).sum())

    @property
    def fg_px(self) -> int:
        return int((self.mask == FOREGROUND).sum())

    @property
    def ignore_px(self) -> int:
        return int((self.mask == IGNORE_INDEX).sum())

def _slide_to_fit(lo: int, hi: int, clo: int, chi: int, min_px: int) -> tuple[int, int]:
    """Slide the 1-D window ``[lo,hi)`` inside the clamp ``[clo,chi)`` so it spans ``min_px`` of real
    data when the clamp region can afford it, without leaving usable clamp region unused.

    After the frac+min_px expand-then-clamp, a window can be shorter than ``min_px`` even though the
    clamp region is wide enough — because the expansion pushed past one clamp edge and got cut. The
    window is grown/slid within the clamp to cover ``min_px`` of real EM instead of padding that
    shortfall. Only the genuinely-small case (clamp region < ``min_px``) stays short.
    """
    span = hi - lo
    if min_px <= 0 or span >= min_px:
        return lo, hi
    avail = chi - clo
    want = min(min_px, avail)  # cannot exceed the real clamp region
    if hi - lo >= want:
        return lo, hi
    # centre the wanted span on the current window centre, then shove inside the clamp bounds.
    c = (lo + hi) / 2.0
    nlo = int(round(c - want / 2.0))
    nhi = nlo + want
    if nlo < clo:
        nlo, nhi = clo, clo + want
    if nhi > chi:
        nhi, nlo = chi, chi - want
    return nlo, nhi

def _expand_clamp(bbox_xyxy, frac: float, clamp_xyxy, min_px: int = 0) -> tuple[int, int, int, int]:
    """Expand a bbox by ``frac`` of its width/height each side and (if ``min_px`` > 0) to at least
    ``min_px`` in each dimension centred on the bbox centre, then clamp to ``clamp_xyxy``.

    ``min_px`` lets the derived crop carry a fixed real-EM field of view (e.g. 1024) around the
    annotation so a large-context encoder is probed on real surrounding tissue rather than
    reflect-padding. It is capped by the available real-EM ``clamp_xyxy`` (valid) region: crops whose
    valid region is smaller stay smaller (even-0-padded to tile_size at build / reflect-padded at load
    — unavoidable where the tissue simply does not extend further). After clamping, the window is
    slid within the clamp so it still spans ``min_px`` of real EM whenever the clamp region can
    afford it, rather than leaving real EM unused to be padded. ``min_px=0`` disables both the
    minimum size and the slide.
    """
    x0, y0, x1, y1 = (int(v) for v in bbox_xyxy)
    cx0, cy0, cx1, cy1 = (int(v) for v in clamp_xyxy)
    dw = int(round((x1 - x0) * frac))
    dh = int(round((y1 - y0) * frac))
    ex0, ey0, ex1, ey1 = x0 - dw, y0 - dh, x1 + dw, y1 + dh
    if min_px > 0:
        cxc, cyc = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = min_px / 2.0
        ex0 = min(ex0, int(round(cxc - half)))
        ey0 = min(ey0, int(round(cyc - half)))
        ex1 = max(ex1, int(round(cxc + half)))
        ey1 = max(ey1, int(round(cyc + half)))
    nx0 = max(cx0, ex0)
    ny0 = max(cy0, ey0)
    nx1 = min(cx1, ex1)
    ny1 = min(cy1, ey1)
    if min_px > 0:
        nx0, nx1 = _slide_to_fit(nx0, nx1, cx0, cx1, min_px)
        ny0, ny1 = _slide_to_fit(ny0, ny1, cy0, cy1, min_px)
    return nx0, ny0, nx1, ny1

def _even_pad_tile(em, mask, inst, ann_xyxy, target: int):
    """Even-0-pad (em, mask, inst) so each dim is ``max(dim, target)``; return padded arrays plus the
    annotation- and valid-EM- bboxes expressed in the padded tile frame.

    Padding is centred (equal both sides, remainder to bottom/right): EM padded with 0, mask with
    IGNORE_INDEX (255), inst with 0. Large crops (both dims >= ``target``) are shifted by 0, so the
    metadata is just the input bboxes. ``ann_xyxy`` is the annotation bbox in the pre-pad crop frame
    (x0,y0,x1,y1); it is offset by the top/left pad. ``valid_em`` is the whole pre-pad crop (all real
    EM), likewise offset. ``target<=0`` disables padding (metadata still computed at zero offset).
    """
    H, W = em.shape[:2]
    th = max(H, target) if target > 0 else H
    tw = max(W, target) if target > 0 else W
    ptop = (th - H) // 2
    pbot = th - H - ptop
    plft = (tw - W) // 2
    prgt = tw - W - plft
    if ptop or pbot or plft or prgt:
        em = np.pad(em, ((ptop, pbot), (plft, prgt)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((ptop, pbot), (plft, prgt)), mode="constant", constant_values=IGNORE_INDEX)
        if inst is not None:
            inst = np.pad(inst, ((ptop, pbot), (plft, prgt)), mode="constant", constant_values=0)
    ax0, ay0, ax1, ay1 = ann_xyxy
    ann_in_tile = [int(ax0 + plft), int(ay0 + ptop), int(ax1 + plft), int(ay1 + ptop)]
    valid_in_tile = [int(plft), int(ptop), int(plft + W), int(ptop + H)]
    return em, mask, inst, ann_in_tile, valid_in_tile

def _real_em_bbox(em, fallback):
    """Bounding box [x0,y0,x1,y1] of the real EM = the non-zero pixels -- the authoritative real-EM
    field of view around the annotation. It equals ``valid_region`` for cleanly-cropped data, reaches
    the whole canvas for context-filled crops (real source EM tiling it), and excludes zero padding
    that some source datasets include in ``valid_region``, which for those sources can extend well
    beyond the real EM with all-zero borders. The ``min_em_px`` gate + context window use this,
    not ``valid_region``, so padding is never mistaken for EM. ``valid_region`` is a subset of this
    wherever it carries no padding. Falls back to ``fallback`` only for an all-zero crop.
    """
    nz = em > 0
    if not nz.any():
        return list(fallback)
    ys = np.any(nz, axis=1)
    xs = np.any(nz, axis=0)
    y0 = int(np.argmax(ys)); y1 = int(len(ys) - np.argmax(ys[::-1]))
    x0 = int(np.argmax(xs)); x1 = int(len(xs) - np.argmax(xs[::-1]))
    return [x0, y0, x1, y1]

def extract_gt(em_path, label_path, crop_entry: dict, row: CropRow, organelle: str,
               context_frac: float = 0.5, min_context_px: int = 0, min_em_px: int = 0,
               pad_even_to: int = 0) -> list[DerivedSample]:
    """Extract a single derived sample from a gt canvas crop (or [] if unusable).

    Optional knobs (both default to 0 = disabled):
      * ``min_em_px`` > 0: drop the crop when the real-EM window min-dim < ``min_em_px``.
      * ``pad_even_to`` > 0: even-0-pad the crop up to at least this many px per dim (EM 0, mask 255,
        inst 0, centred). When set, the record carries ``annotation_bbox_in_tile_xyxy`` +
        ``valid_em_in_tile_xyxy``, which the harness loader uses to crop so the annotation is
        contained.
    """
    em = np.asarray(read_tif(em_path))
    if em.ndim != 2:
        em = np.asarray(read_planes(em_path))[0]
    label = np.asarray(read_tif(label_path))
    if label.ndim != 2:
        label = label.reshape(label.shape[-2], label.shape[-1]) if label.ndim >= 2 else label
    H, W = em.shape[:2]

    valid = crop_entry.get("valid_region_in_canvas_xyxy") or [0, 0, W, H]
    ann = crop_entry.get("annotation_bbox_in_canvas_xyxy") or valid
    coverage = str(crop_entry.get("coverage_tier", "sparse")).lower()
    op = crop_entry.get("organelles_present", {})

    # The context clamp is the real-EM field of view around the annotation. min_em_px gates on how
    # much real EM is available, not on annotation size, so a small annotation inside a large image
    # qualifies. Under partial or sparse coverage, EM beyond the tight valid_region is real tissue that
    # supplies context and becomes ignore rather than background, so clamp to the real-EM extent.
    # Dense/full coverage keeps valid_region, since extending further would be false background.
    clamp = _real_em_bbox(em, valid) if coverage in ("partial", "sparse") else valid
    rx0, ry0, rx1, ry1 = _expand_clamp(ann, context_frac, clamp, min_context_px)
    if rx1 <= rx0 or ry1 <= ry0:
        return []
    # min_em_px: drop crops whose real-EM window is smaller than min_em_px on its short side.
    if min_em_px > 0 and min(rx1 - rx0, ry1 - ry0) < min_em_px:
        return []
    em_crop = em[ry0:ry1, rx0:rx1]
    fg_full = canonical_mask(label, op, organelle)
    fg = fg_full[ry0:ry1, rx0:rx1]

    mask = np.where(fg, FOREGROUND, BACKGROUND).astype(np.uint8)
    # annotation bbox within the pre-pad crop frame (used for both the ignore mask and the in-tile metadata)
    ax0 = max(int(ann[0]) - rx0, 0)
    ay0 = max(int(ann[1]) - ry0, 0)
    ax1 = min(int(ann[2]) - rx0, mask.shape[1])
    ay1 = min(int(ann[3]) - ry0, mask.shape[0])
    if coverage in ("partial", "sparse"):
        # ignore everything outside the densely-annotated bbox (unlabelled, not background)
        ig = np.ones(mask.shape, dtype=bool)
        if ax1 > ax0 and ay1 > ay0:
            ig[ay0:ay1, ax0:ax1] = False
        mask[ig] = IGNORE_INDEX
    # a clamped/empty annotation bbox falls back to the whole crop for the metadata centre.
    ann_crop = [ax0, ay0, ax1, ay1] if (ax1 > ax0 and ay1 > ay0) else [0, 0, mask.shape[1], mask.shape[0]]

    inst, is_inst = None, False
    if organelle == "mito":
        from .canonicalize import instance_map
        inst_full, is_inst = instance_map(label, op, organelle)
        inst = np.ascontiguousarray(inst_full[ry0:ry1, rx0:rx1], dtype=np.int32)
        inst[mask == IGNORE_INDEX] = 0  # don't carry instances into ignore regions

    em_crop = np.ascontiguousarray(em_crop, dtype=np.uint8)
    extra = {
        "coverage_tier": coverage,
        "voxel_nm": crop_entry.get("voxel_size_nm"),
        "source_region_xyxy": [rx0, ry0, rx1, ry1],
    }
    if pad_even_to > 0:
        em_crop, mask, inst, ann_in_tile, valid_in_tile = _even_pad_tile(
            em_crop, mask, inst, ann_crop, pad_even_to)
        extra["annotation_bbox_in_tile_xyxy"] = ann_in_tile
        extra["valid_em_in_tile_xyxy"] = valid_in_tile

    sample = DerivedSample(
        sample_id="",  # filled by the builder
        em=em_crop,
        mask=mask,
        extra=extra,
        inst=inst,
        gt_is_instance=is_inst,
    )
    if sample.valid_px == 0:
        return []
    return [sample]

def extract_oo(crop_dir, row: CropRow, organelle: str, context_frac: float = 0.5,
               min_context_px: int = 0, min_em_px: int = 0, pad_even_to: int = 0) -> list[DerivedSample]:
    """Extract per-(orientation, plane) derived samples from an openOrganelle crop.

    The optional knobs mirror ``extract_gt``: ``min_em_px`` drops small real-EM windows, ``pad_even_to``
    even-0-pads to the target and records the in-tile annotation/valid metadata."""
    samples: list[DerivedSample] = []
    for s in iter_oo_samples(crop_dir, organelle):
        plane = s.raw_plane
        ph, pw = plane.shape[:2]
        wr0, wr1 = s.win_rows
        wc0, wc1 = s.win_cols
        rx0, ry0, rx1, ry1 = _expand_clamp((wc0, wr0, wc1, wr1), context_frac, (0, 0, pw, ph), min_context_px)
        if min_em_px > 0 and min(rx1 - rx0, ry1 - ry0) < min_em_px:
            continue
        em_crop = np.ascontiguousarray(plane[ry0:ry1, rx0:rx1], dtype=np.uint8)
        mask = np.full(em_crop.shape, IGNORE_INDEX, dtype=np.uint8)
        # place the densely-annotated window (background, then foreground, then unknown->ignore)
        lr0, lr1 = wr0 - ry0, wr1 - ry0
        lc0, lc1 = wc0 - rx0, wc1 - rx0
        win = mask[lr0:lr1, lc0:lc1]
        win[...] = BACKGROUND
        win[s.fg] = FOREGROUND
        win[s.unknown] = IGNORE_INDEX
        mask[lr0:lr1, lc0:lc1] = win
        inst = None
        if organelle == "mito" and s.inst is not None:
            inst = np.zeros(em_crop.shape, dtype=np.int32)
            inst[lr0:lr1, lc0:lc1] = s.inst
            inst[mask == IGNORE_INDEX] = 0
        # annotation window in the pre-pad crop frame (clamped to crop bounds).
        ann_crop = [max(lc0, 0), max(lr0, 0), min(lc1, mask.shape[1]), min(lr1, mask.shape[0])]
        if not (ann_crop[2] > ann_crop[0] and ann_crop[3] > ann_crop[1]):
            ann_crop = [0, 0, mask.shape[1], mask.shape[0]]
        extra = {
            "coverage_tier": "oo_window",
            "orientation": s.orientation,
            "plane_k": s.plane_k,
            "seg_slice": s.seg_slice,
            "source_region_xyxy": [rx0, ry0, rx1, ry1],
            # densification metadata (additive; baseline 200nm crops report 200 / False)
            "oo_z_spacing_nm": s.oo_z_spacing_nm,
            "plane_z_nm": s.plane_z_nm,
            "oo_dense_z": bool(s.oo_dense_z),
        }
        if pad_even_to > 0:
            em_crop, mask, inst, ann_in_tile, valid_in_tile = _even_pad_tile(
                em_crop, mask, inst, ann_crop, pad_even_to)
            extra["annotation_bbox_in_tile_xyxy"] = ann_in_tile
            extra["valid_em_in_tile_xyxy"] = valid_in_tile
        sample = DerivedSample(
            sample_id="",
            em=em_crop,
            mask=mask,
            extra=extra,
            inst=inst,
            gt_is_instance=(inst is not None),  # OO seg_mito is real instances
        )
        if sample.valid_px > 0:
            samples.append(sample)
    return samples

def extract_row(corpus_root, row: CropRow, organelle: str, cache, context_frac: float = 0.5,
                min_context_px: int = 0, min_em_px: int = 0, pad_even_to: int = 0) -> list[DerivedSample]:
    """Dispatch one split row to the right extractor; skip-with-warning on missing data."""
    from pathlib import Path

    from .splits import resolve_gt_paths, resolve_oo_paths

    try:
        if row.is_oo:
            em_xy, _em_xz, _seg = resolve_oo_paths(corpus_root, row, organelle)
            crop_dir = Path(em_xy).parent
            if not crop_dir.exists():
                warnings.warn(f"[skip] OO crop dir missing: {crop_dir}")
                return []
            return extract_oo(crop_dir, row, organelle, context_frac, min_context_px,
                              min_em_px, pad_even_to)
        em_path, label_path = resolve_gt_paths(corpus_root, row)
        if not em_path.exists() or not label_path.exists():
            warnings.warn(f"[skip] missing em/label for {row.dataset}/{row.crop_id}: {em_path.name}")
            return []
        entry = cache.crop_entry(row.dataset, row.crop_id)
        if entry is None:
            warnings.warn(f"[skip] no manifest entry for {row.dataset}/{row.crop_id}")
            return []
        return extract_gt(em_path, label_path, entry, row, organelle, context_frac, min_context_px,
                          min_em_px, pad_even_to)
    except Exception as exc:  # never let one bad crop kill the whole build
        warnings.warn(f"[skip] extract failed for {row.dataset}/{row.crop_id}: {exc!r}")
        return []
