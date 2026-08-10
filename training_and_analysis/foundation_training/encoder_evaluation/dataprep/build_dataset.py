"""Build the derived split-aware ``(EM, mask)`` dataset the decoder probe reads.

Re-runnable & dynamic: re-reads the split CSVs + dataset manifests every run, hard-codes no crop
counts or dataset lists, and skips with a warning anything absent from disk, so it builds
whatever subset of the corpus is present. Output is a plain, tar-friendly folder + a ``manifest.jsonl``
the harness consumes, plus a ``build_report.json`` summary.

Usage:
    python -m encoder_evaluation.dataprep.build_dataset \
        --corpus-root <annotated sources> --out <derived dataset> \
        --organelles mito er [--splits train val test] [--context-frac 0.5] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

from ..constants import DEFAULT_CORPUS_ROOT, DEFAULT_DERIVED_ROOT, VALID_ORGANELLES
from .extract import extract_row
from .io import write_json, write_png_L, write_tif_u16
from .splits import load_split_rows, make_cache, sanitize_id

def _sample_id(row, sample) -> str:
    base = f"{row.dataset}__{sanitize_id(row.crop_id)}"
    ex = sample.extra
    if "orientation" in ex:
        base = f"{base}__{ex['orientation']}__p{ex.get('plane_k', 0)}"
    return base

def run(args) -> dict:
    corpus_root = Path(args.corpus_root)
    out_root = Path(args.out)
    organelles = [o for o in args.organelles if o in VALID_ORGANELLES]
    splits = set(args.splits) if args.splits else None
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
        "per_group": {},
        "skipped_rows": 0,
        "warnings": [],
    }
    n_written = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for organelle in organelles:
            group = f"group1_{organelle}"
            cache = make_cache(corpus_root)
            try:
                rows = load_split_rows(corpus_root, organelle)
            except FileNotFoundError as exc:
                warnings.warn(f"[skip group] {exc}")
                report["warnings"].append(str(exc))
                continue
            if splits:
                rows = [r for r in rows if r.split in splits]
            if args.limit:
                rows = rows[: args.limit]

            counts: dict = defaultdict(lambda: {"crops": 0, "samples": 0, "fg_px": 0, "ignore_px": 0,
                                                 "valid_px": 0, "empty_fg_samples": 0, "skipped": 0})
            print(f"[{group}] {len(rows)} crop rows")
            for i, row in enumerate(rows):
                samples = extract_row(corpus_root, row, organelle, cache, args.context_frac,
                                      getattr(args, "min_context_px", 0),
                                      getattr(args, "min_em_px", 0),
                                      getattr(args, "pad_even_to", 0))
                key = f"{row.split}/{row.dataset}"
                if not samples:
                    counts[key]["skipped"] += 1
                    report["skipped_rows"] += 1
                    continue
                counts[key]["crops"] += 1
                sample_dir = out_root / group / row.split / row.dataset
                sample_dir.mkdir(parents=True, exist_ok=True)
                for s in samples:
                    s.sample_id = _sample_id(row, s)
                    em_rel = f"{group}/{row.split}/{row.dataset}/{s.sample_id}_em.png"
                    mask_rel = f"{group}/{row.split}/{row.dataset}/{s.sample_id}_mask.png"
                    write_png_L(out_root / em_rel, s.em)
                    write_png_L(out_root / mask_rel, s.mask)
                    inst_rel = None
                    if s.inst is not None and int(s.inst.max()) > 0:
                        inst_rel = f"{group}/{row.split}/{row.dataset}/{s.sample_id}_inst.tif"
                        write_tif_u16(out_root / inst_rel, s.inst)
                    rec = {
                        "sample_id": s.sample_id,
                        "organelle": organelle,
                        "group": group,
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
                        # OO densification (additive; non-OO records -> None/None/False, and a filter
                        # 'drop oo_dense_z==True' recovers the baseline-spacing set exactly)
                        "oo_z_spacing_nm": s.extra.get("oo_z_spacing_nm"),
                        "plane_z_nm": s.extra.get("plane_z_nm"),
                        "oo_dense_z": bool(s.extra.get("oo_dense_z", False)),
                        "em_path": em_rel,
                        "mask_path": mask_rel,
                        "inst_path": inst_rel,
                        "gt_is_instance": bool(s.gt_is_instance),
                        "height": int(s.em.shape[0]),
                        "width": int(s.em.shape[1]),
                        "fg_px": s.fg_px,
                        "ignore_px": s.ignore_px,
                        "valid_px": s.valid_px,
                    }
                    # v2 (present only when --pad-even-to > 0): where the annotation + real EM sit
                    # within the (even-0-padded) tile, in native px. The harness loader auto-detects
                    # v2 by the presence of annotation_bbox_in_tile_xyxy.
                    if "annotation_bbox_in_tile_xyxy" in s.extra:
                        rec["annotation_bbox_in_tile_xyxy"] = s.extra["annotation_bbox_in_tile_xyxy"]
                        rec["valid_em_in_tile_xyxy"] = s.extra.get("valid_em_in_tile_xyxy")
                    write_json(out_root / f"{em_rel[:-7]}_meta.json", rec)
                    mf.write(json.dumps(rec) + "\n")
                    n_written += 1
                    c = counts[key]
                    c["samples"] += 1
                    c["fg_px"] += s.fg_px
                    c["ignore_px"] += s.ignore_px
                    c["valid_px"] += s.valid_px
                    if s.fg_px == 0:
                        c["empty_fg_samples"] += 1
                if (i + 1) % 200 == 0:
                    print(f"  ... {i + 1}/{len(rows)} rows, {n_written} samples")
            report["per_group"][group] = {
                "n_rows": len(rows),
                "n_samples": sum(c["samples"] for c in counts.values()),
                "by_split_dataset": {k: dict(v) for k, v in sorted(counts.items())},
            }
            print(f"[{group}] wrote {report['per_group'][group]['n_samples']} samples")

    report["n_samples_total"] = n_written
    write_json(out_root / "build_report.json", report)
    print(f"Done: {n_written} samples -> {manifest_path}")
    return report

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the derived segmentation dataset for the encoder probe.")
    p.add_argument("--corpus-root", default=DEFAULT_CORPUS_ROOT)
    p.add_argument("--out", default=DEFAULT_DERIVED_ROOT)
    p.add_argument("--organelles", nargs="+", default=list(VALID_ORGANELLES))
    p.add_argument("--splits", nargs="*", default=None, help="Subset of {train,val,test} (default all)")
    p.add_argument("--context-frac", type=float, default=0.5,
                   help="Real-EM context margin around the annotation bbox, as a fraction of its size.")
    p.add_argument("--min-context-px", type=int, default=0,
                   help="Also expand each derived crop to at least this many px per dimension of real EM "
                        "(centred on the annotation, capped by the valid region), so a large-context "
                        "encoder is probed on real tissue rather than reflect-padding. 0 = off "
                        "(default). Use e.g. 1024 to build a native-resolution dataset.")
    p.add_argument("--min-em-px", type=int, default=0,
                   help="Drop a crop when the available real-EM image context (min dim) is below this "
                        "many px (0 = off, default). Gates on EM context rather than annotation size: a tiny "
                        "annotation (e.g. 150px of ER) inside a large real-EM image is kept; only "
                        "genuinely small EM images (e.g. empiar_10791 @150px) are dropped. Context-filled "
                        "crops use the real source EM tiling the whole canvas, not just the tight "
                        "annotated valid_region (see extract._real_em_bbox). v2 uses 512.")
    p.add_argument("--pad-even-to", type=int, default=0,
                   help="Even-0-pad each derived crop up to at least this many px per dim (EM 0, mask "
                        "255, inst 0, centred). Large crops are not padded. When >0 the record carries "
                        "annotation_bbox_in_tile_xyxy so the v2 loader crops to contain the annotation. "
                        "0 = off (default). v2 uses 1024.")
    p.add_argument("--limit", type=int, default=0, help="Cap rows per organelle (debug).")
    args = p.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()
