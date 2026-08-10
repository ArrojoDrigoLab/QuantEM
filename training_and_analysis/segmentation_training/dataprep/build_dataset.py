"""Entry point — build the per-organelle canonical-scale derived segmentation dataset.

Reads the group2 split CSVs (the held-out-source design), extracts each crop with the ignore/coverage
contract, per-crop label decode and OpenOrganelle physical-nm registration, then resamples every
sample to the organelle's canonical nm/px (ER -> 2 nm, mito -> 8 nm) and writes a plain, tar-friendly
folder + a ``manifest.jsonl`` the harness consumes + a ``build_report``.

Re-runnable & dynamic: re-reads the split CSVs + manifests every run, hard-codes no crop counts or
dataset lists, and skips with a warning anything absent from disk. The on-disk split value
``train_pool`` is written under the derived dir name ``train`` (the adaptation pool). Crops with no
voxel size (``scale_band=unknown``, 160 in the real corpus) follow ``--null-scale-policy`` (default
``drop`` — see constants; a native-bucket and an assumed-scale branch are also selectable).

Usage:
    python -m segmentation_training.dataprep.build_dataset \
        --corpus-root <supplied at launch> --out <supplied at launch> \
        --organelles er mito [--splits train val test] [--context-frac 0.5] [--limit N] \
        [--null-scale-policy drop|native_bucket|estimate] [--target-nm 2.0]
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..constants import (
    ASSUMED_SCALE_BAND_NM,
    CANONICAL_NM,
    DEFAULT_CORPUS_ROOT,
    DEFAULT_DERIVED_ROOT,
    FOREGROUND,
    IGNORE_INDEX,
    NULL_SCALE_POLICY,
    SPLIT_VALUE_TO_DIR,
    VALID_ORGANELLES,
)
from .extract import extract_row
from .io import write_json, write_png_L, write_tif_u16
from .resample import resample_arrays, resolve_src_nm
from .splits import load_split_rows, make_cache, sanitize_id


def _sample_id(row, sample) -> str:
    base = f"{row.dataset}__{sanitize_id(row.crop_id)}"
    ex = sample.extra
    if "orientation" in ex:
        base = f"{base}__{ex['orientation']}__p{ex.get('plane_k', 0)}"
    return base


def _bucket_dir(group: str, bucket: str, split: str, dataset: str) -> str:
    """Derived sub-dir for a bucket. Canonical is the flat default; native / native_unscaled get their
    own top-level bucket dir so the harness can select a resolution view (the input-scale experiment) via
    ``load_manifest(..., bucket=)``."""
    if bucket == "canonical":
        return f"{group}/{split}/{dataset}"
    return f"{group}/{bucket}/{split}/{dataset}"


def _scale_ann_bbox(ann_crop, factor) -> list[int]:
    """Scale a native-frame annotation bbox (x0,y0,x1,y1) into a resampled frame by ``factor`` = (fr, fc)
    — fr scales rows/y, fc scales cols/x. ``factor`` None (no resample) returns the bbox unchanged. The
    canonical metadata requirement: the annotation position is recomputed in the post-resample
    pixel frame, and the resample uses scipy.ndimage.zoom (out_dim = round(in_dim * factor)) so scaling
    the bbox coordinates by the same factor tracks the label grid."""
    if factor is None:
        return [int(v) for v in ann_crop]
    fr, fc = factor
    x0, y0, x1, y1 = ann_crop
    return [int(round(x0 * fc)), int(round(y0 * fr)), int(round(x1 * fc)), int(round(y1 * fr))]


def _even_pad(em, mask, inst, ann_frame, target: int):
    """Even-0-pad (em, mask, inst) so each dim is ``max(dim, target)`` (EM 0, mask IGNORE_INDEX, inst 0,
    centred, remainder bottom/right); return padded arrays + the annotation- and valid-EM- bboxes in the
    final padded tile frame. ``ann_frame`` is the annotation bbox already in this variant's (pre-pad)
    pixel frame. ``target<=0`` disables padding (metadata still computed at zero offset)."""
    H, W = em.shape[:2]
    th = max(H, target) if target > 0 else H
    tw = max(W, target) if target > 0 else W
    ptop, plft = (th - H) // 2, (tw - W) // 2
    pbot, prgt = th - H - ptop, tw - W - plft
    if ptop or pbot or plft or prgt:
        em = np.pad(em, ((ptop, pbot), (plft, prgt)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((ptop, pbot), (plft, prgt)), mode="constant", constant_values=IGNORE_INDEX)
        if inst is not None:
            inst = np.pad(inst, ((ptop, pbot), (plft, prgt)), mode="constant", constant_values=0)
    ax0, ay0, ax1, ay1 = ann_frame
    # clamp the (possibly rounded) annotation bbox to the pre-pad crop, then offset by the top/left pad.
    ax0 = max(0, min(int(ax0), W)); ax1 = max(ax0, min(int(ax1), W))
    ay0 = max(0, min(int(ay0), H)); ay1 = max(ay0, min(int(ay1), H))
    ann_in_tile = [ax0 + plft, ay0 + ptop, ax1 + plft, ay1 + ptop]
    valid_in_tile = [plft, ptop, plft + W, ptop + H]
    return em, mask, inst, ann_in_tile, valid_in_tile


def _scale_variants(sample, row, target_nm: float, policy: str, scale_mode: str, pad_even_to: int = 0):
    """Emit-variants for one DerivedSample given the scale mode (does not mutate ``sample``).

    Returns a list of ``(bucket, canonical_nm, factor, em, mask, inst, ann_in_tile, valid_in_tile)``
    tuples:
      * canonical (resampled to ``target_nm``, bucket ``canonical``) for scale_mode in {canonical, both};
      * native   (source resolution, unresampled, bucket ``native``) for scale_mode in {native, both}.
    Crop inclusion follows the canonical/null-scale logic so the native and canonical sets are the same
    crops (the input-scale experiment = the same crops at two resolutions). Returns ``[]`` to drop the crop.

    ``pad_even_to`` > 0 even-0-pads each emitted variant after resampling and computes the in-tile
    annotation/valid metadata in that variant's (post-resample) canonical pixel frame, the frame the
    loader reads those fields in. When ``pad_even_to`` == 0 the arrays are returned unpadded and the
    *_in_tile fields are None, so records carry no in-tile metadata and the loader takes its fallback path.
    """
    ann_native = sample.extra.get("annotation_bbox_in_crop_xyxy") or [0, 0, sample.em.shape[1], sample.em.shape[0]]
    src_row, src_col = resolve_src_nm(sample.extra)
    canonical = None  # (bucket, canonical_nm, factor, em, mask, inst)
    if src_row is None or src_col is None:
        # unknown-scale crop -> the inclusion decision follows the null-scale policy.
        if policy == "native_bucket":
            canonical = ("native_unscaled", None, None, sample.em, sample.mask, sample.inst)
        elif policy == "estimate":
            est = ASSUMED_SCALE_BAND_NM.get(str(row.scale_band))
            if est is None:
                warnings.warn(f"[skip null-scale] no band prior for scale_band={row.scale_band!r} "
                              f"({row.dataset}/{row.crop_id})")
                return []
            src_row = src_col = float(est)
        else:  # "drop" (default) -> excluded from both views (keeps the sets matched)
            return []
    if canonical is None:
        em2, mask2, inst2, factor = resample_arrays(
            sample.em, sample.mask, sample.inst, src_row, src_col, target_nm)
        canonical = ("canonical", float(target_nm),
                     [round(factor[0], 6), round(factor[1], 6)], em2, mask2, inst2)

    raw_variants = []  # (bucket, canonical_nm, factor, em, mask, inst)
    if scale_mode in ("canonical", "both"):
        raw_variants.append(canonical)
    # native view = the source-resolution crop (no resample); only for genuinely resampled crops so a
    # null-scale crop already living in native_unscaled is not double-emitted. Its nm/px is recorded
    # via src_nm_row/col on the record; canonical_nm is None because the view is already native.
    if scale_mode in ("native", "both") and canonical[0] == "canonical":
        raw_variants.append(("native", None, None, sample.em, sample.mask, sample.inst))

    out = []
    for bucket, canonical_nm, factor, em, mask, inst in raw_variants:
        # annotation bbox in this variant's pixel frame: canonical -> scale native bbox by factor;
        # native / native_unscaled -> factor None (unresampled) -> native bbox unchanged.
        ann_frame = _scale_ann_bbox(ann_native, factor if bucket == "canonical" else None)
        if pad_even_to > 0:
            em, mask, inst, ann_in_tile, valid_in_tile = _even_pad(em, mask, inst, ann_frame, pad_even_to)
        else:
            ann_in_tile = valid_in_tile = None
        out.append((bucket, canonical_nm, factor, em, mask, inst, ann_in_tile, valid_in_tile))
    return out


def run(args) -> dict:
    corpus_root = Path(args.corpus_root)
    out_root = Path(args.out)
    organelles = [o for o in args.organelles if o in VALID_ORGANELLES]
    splits = set(args.splits) if args.splits else None
    policy = args.null_scale_policy
    scale_mode = getattr(args, "scale_mode", "canonical")  # programmatic callers default to canonical
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.jsonl"
    report: dict = {
        "corpus_root": str(corpus_root),
        "out_root": str(out_root),
        "context_frac": args.context_frac,
        "min_context_px": getattr(args, "min_context_px", 0),
        "min_em_px": getattr(args, "min_em_px", 0),
        "pad_even_to": getattr(args, "pad_even_to", 0),
        "organelles": organelles,
        "null_scale_policy": policy,
        "scale_mode": scale_mode,
        "target_nm": {},
        "per_group": {},
        "skipped_rows": 0,
        "skipped_null_scale": 0,
        "warnings": [],
    }
    n_written = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for organelle in organelles:
            group = f"group2_{organelle}"
            target_nm = float(args.target_nm) if args.target_nm else float(CANONICAL_NM[organelle])
            report["target_nm"][organelle] = target_nm
            cache = make_cache(corpus_root)
            try:
                rows = load_split_rows(corpus_root, organelle)
            except FileNotFoundError as exc:
                warnings.warn(f"[skip group] {exc}")
                report["warnings"].append(str(exc))
                continue
            # Map the on-disk split value -> derived dir name, and optionally filter.
            for r in rows:
                r.split = SPLIT_VALUE_TO_DIR.get(r.split, r.split)
            if splits:
                rows = [r for r in rows if r.split in splits]
            if args.limit:
                rows = rows[: args.limit]

            counts: dict = defaultdict(lambda: {"crops": 0, "samples": 0, "fg_px": 0, "ignore_px": 0,
                                                 "valid_px": 0, "empty_fg_samples": 0, "skipped": 0,
                                                 "null_scale": 0})
            print(f"[{group}] {len(rows)} crop rows -> canonical {target_nm} nm/px")
            for i, row in enumerate(rows):
                samples = extract_row(corpus_root, row, organelle, cache, args.context_frac,
                                      getattr(args, "min_context_px", 0),
                                      getattr(args, "min_em_px", 0))
                key = f"{row.split}/{row.dataset}"
                if not samples:
                    counts[key]["skipped"] += 1
                    report["skipped_rows"] += 1
                    continue
                wrote_any = False
                for s in samples:
                    variants = _scale_variants(s, row, target_nm, policy, scale_mode,
                                               getattr(args, "pad_even_to", 0))
                    if not variants:
                        counts[key]["null_scale"] += 1
                        report["skipped_null_scale"] += 1
                        continue
                    sid = _sample_id(row, s)
                    for bucket, canonical_nm, factor, em, mask, inst, ann_in_tile, valid_in_tile in variants:
                        valid_px = int((mask != IGNORE_INDEX).sum())
                        if valid_px == 0:  # resample can, in principle, wipe a 1px sample
                            continue
                        fg_px = int((mask == FOREGROUND).sum())
                        ignore_px = int((mask == IGNORE_INDEX).sum())
                        rel_dir = _bucket_dir(group, bucket, row.split, row.dataset)
                        (out_root / rel_dir).mkdir(parents=True, exist_ok=True)
                        em_rel = f"{rel_dir}/{sid}_em.png"
                        mask_rel = f"{rel_dir}/{sid}_mask.png"
                        write_png_L(out_root / em_rel, em)
                        write_png_L(out_root / mask_rel, mask)
                        inst_rel = None
                        if inst is not None and int(inst.max()) > 0:
                            inst_rel = f"{rel_dir}/{sid}_inst.tif"
                            write_tif_u16(out_root / inst_rel, inst)
                        rec = {
                            "sample_id": sid,
                            "organelle": organelle,
                            "group": group,
                            "bucket": bucket,
                            "scale_mode": scale_mode,
                            "split": row.split,
                            "subgroup": row.subgroup,
                            "collection": row.collection,
                            "dataset": row.dataset,
                            "crop_id": row.crop_id,
                            "modality": row.modality,
                            "scale_band": row.scale_band,
                            "tissue_context": row.tissue_context,
                            "species_group": row.species_group,
                            "coverage_tier": s.extra.get("coverage_tier"),
                            "orientation": s.extra.get("orientation"),
                            "plane_k": s.extra.get("plane_k"),
                            # OO densification (additive; non-OO records -> None/None/False, and a
                            # filter 'drop oo_dense_z==True' recovers the pre-densification set exactly)
                            "oo_z_spacing_nm": s.extra.get("oo_z_spacing_nm"),
                            "plane_z_nm": s.extra.get("plane_z_nm"),
                            "oo_dense_z": bool(s.extra.get("oo_dense_z", False)),
                            "canonical_nm": canonical_nm,
                            "src_nm_row": s.extra.get("src_nm_row"),
                            "src_nm_col": s.extra.get("src_nm_col"),
                            "resample_factor": factor,
                            "em_path": em_rel,
                            "mask_path": mask_rel,
                            "inst_path": inst_rel,
                            "gt_is_instance": bool(s.gt_is_instance),
                            "height": int(em.shape[0]),
                            "width": int(em.shape[1]),
                            "fg_px": fg_px,
                            "ignore_px": ignore_px,
                            "valid_px": valid_px,
                        }
                        # Present only when --pad-even-to > 0: annotation + real-EM position in
                        # the (even-0-padded) tile, in the canonical post-resample pixel frame for the
                        # canonical bucket (native bbox scaled by resample_factor), or the native frame
                        # for native/native_unscaled buckets. The harness loader detects these fields by
                        # the presence of annotation_bbox_in_tile_xyxy.
                        if ann_in_tile is not None:
                            rec["annotation_bbox_in_tile_xyxy"] = ann_in_tile
                            rec["valid_em_in_tile_xyxy"] = valid_in_tile
                        write_json(out_root / f"{em_rel[:-7]}_meta.json", rec)
                        mf.write(json.dumps(rec) + "\n")
                        n_written += 1
                        wrote_any = True
                        c = counts[key]
                        c["samples"] += 1
                        c["fg_px"] += fg_px
                        c["ignore_px"] += ignore_px
                        c["valid_px"] += valid_px
                        if fg_px == 0:
                            c["empty_fg_samples"] += 1
                if wrote_any:
                    counts[key]["crops"] += 1
                if (i + 1) % 200 == 0:
                    print(f"  ... {i + 1}/{len(rows)} rows, {n_written} samples")
            report["per_group"][group] = {
                "n_rows": len(rows),
                "n_samples": sum(c["samples"] for c in counts.values()),
                "n_null_scale": sum(c["null_scale"] for c in counts.values()),
                "by_split_dataset": {k: dict(v) for k, v in sorted(counts.items())},
            }
            print(f"[{group}] wrote {report['per_group'][group]['n_samples']} samples "
                  f"({report['per_group'][group]['n_null_scale']} null-scale via '{policy}')")

    report["n_samples_total"] = n_written
    write_json(out_root / "build_report.json", report)
    print(f"Done: {n_written} samples -> {manifest_path}")
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the canonical-scale derived segmentation dataset.")
    p.add_argument("--corpus-root", default=DEFAULT_CORPUS_ROOT)
    p.add_argument("--out", default=DEFAULT_DERIVED_ROOT)
    p.add_argument("--organelles", nargs="+", default=list(VALID_ORGANELLES))
    p.add_argument("--splits", nargs="*", default=None,
                   help="Subset of derived dir names {train,val,test} (default all).")
    p.add_argument("--context-frac", type=float, default=0.5,
                   help="Real-EM context margin around the annotation bbox, as a fraction of its size.")
    p.add_argument("--min-context-px", type=int, default=0,
                   help="Expand each native real-EM crop to at least this many px per dim (centred on "
                        "the annotation, slide-to-fit, capped by the valid region) so a large-context "
                        "encoder sees real tissue not padding. 0 = off (default); reported builds use 1024.")
    p.add_argument("--min-em-px", type=int, default=0,
                   help="Drop a crop when the available native real-EM image context (min dim) is below "
                        "this many px, applied before resample. Gates on EM context, not annotation size: "
                        "a tiny annotation in a large real-EM image is kept; only genuinely small EM "
                        "images (e.g. empiar_10791 @150px) drop. Context-filled crops use the real source "
                        "EM tiling the whole canvas, not just the annotated valid_region (see "
                        "extract._real_em_bbox). 0 = off (default); the reported builds use 512.")
    p.add_argument("--pad-even-to", type=int, default=0,
                   help="Even-0-pad each emitted variant after resampling up to at least this many px "
                        "per dim (EM 0, mask 255, inst 0, centred); records the in-tile annotation/valid "
                        "metadata in the canonical post-resample frame. 0 = off (default); 1024 as reported.")
    p.add_argument("--limit", type=int, default=0, help="Cap rows per organelle.")
    p.add_argument("--null-scale-policy", choices=("drop", "native_bucket", "estimate"),
                   default=NULL_SCALE_POLICY,
                   help="Disposition of unknown-scale crops (default from constants: drop).")
    p.add_argument("--target-nm", type=float, default=0.0,
                   help="Override the canonical nm/px for all built organelles (0 = per-organelle default).")
    p.add_argument("--scale-mode", choices=("canonical", "native", "both"), default="canonical",
                   help="Resolution view(s) to emit: canonical (the default), native (source "
                        "resolution, for the native-vs-standardised comparison), or both.")
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
