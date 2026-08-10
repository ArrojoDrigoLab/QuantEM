"""Shared CLI argument helpers so all data tools filter/resolve paths consistently."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data.filters import DEFAULT_BLOCKING_WARNING_TOKENS, SSLFilterConfig
from ..data.manifest import DEFAULT_MANIFEST_PATH, infer_exports_root

def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, required=DEFAULT_MANIFEST_PATH is None,
                        help="Tile manifest JSONL (from dataset_assembly/tiling/build_manifest.py)")
    parser.add_argument(
        "--exports-root",
        default=None,
        help="Root that output_tile_path is relative to (default: inferred as manifest grandparent).",
    )
    parser.add_argument("--tile-root", default=None, help="Optional alternate tile root override.")
    parser.add_argument("--output-root", default=None, help="Where to write artifacts/reports.")
    parser.add_argument(
        "--no-source-index",
        action="store_true",
        help="Skip the source_id->run_dir filesystem scan (only safe if every record has run_dir).",
    )

def add_filter_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("SSL tile filter (library defaults are permissive)")
    g.add_argument("--allowed-status", nargs="+", default=["accepted"], help="Accepted manifest statuses.")
    g.add_argument("--min-side", type=int, default=0, help="Require min(width,height) >= this (0 = off).")
    g.add_argument("--max-artifact-fraction", type=float, default=None, help="Drop tiles above this artifact fraction.")
    g.add_argument("--min-tissue-score", type=float, default=None, help="Drop tiles below this tissue score.")
    g.add_argument("--max-background-fraction", type=float, default=None, help="Drop tiles above this background fraction.")
    g.add_argument(
        "--blocking-warnings",
        nargs="*",
        default=sorted(DEFAULT_BLOCKING_WARNING_TOKENS),
        help="normalization_warning tokens that exclude a tile (default blocks low_dynamic_range / "
        "insufficient_valid_support; 'auto_reported_contrast_inverted' is benign and kept).",
    )
    g.add_argument("--include-low-dynamic-range", action="store_true", help="Keep low_dynamic_range tiles.")
    g.add_argument("--no-dtype-filter", action="store_true", help="Do not require tile_storage_dtype==uint8.")
    g.add_argument("--allowed-source-kinds", nargs="*", default=None, help="Restrict to these source_kinds.")
    g.add_argument(
        "--allowed-licenses",
        nargs="*",
        default=None,
        help="License whitelist: keep only tiles whose dataset license is one of these "
        "(e.g. --allowed-licenses 'CC0 1.0' 'CC BY 4.0' 'ODC-BY 1.0'). Empty/missing license is "
        "kept unless --exclude-unlicensed. Omit to disable license filtering.",
    )
    g.add_argument(
        "--exclude-unlicensed",
        action="store_true",
        help="With --allowed-licenses, also drop tiles that have no license (default keeps them).",
    )

def build_filter_config(args, min_side_default: int | None = None) -> SSLFilterConfig:
    min_side = args.min_side if getattr(args, "min_side", 0) else (min_side_default or 0)
    return SSLFilterConfig(
        allowed_status=frozenset(args.allowed_status),
        required_dtype=None if getattr(args, "no_dtype_filter", False) else "uint8",
        exclude_low_dynamic_range=not getattr(args, "include_low_dynamic_range", False),
        blocking_warning_tokens=frozenset(args.blocking_warnings or []),
        min_side=int(min_side or 0),
        max_artifact_fraction=args.max_artifact_fraction,
        min_tissue_score=args.min_tissue_score,
        max_background_fraction=args.max_background_fraction,
        allowed_source_kinds=frozenset(args.allowed_source_kinds) if args.allowed_source_kinds else None,
        allowed_licenses=frozenset(args.allowed_licenses) if getattr(args, "allowed_licenses", None) else None,
        allow_unlicensed=not getattr(args, "exclude_unlicensed", False),
    )

def resolve_exports_root(args) -> Path:
    if getattr(args, "exports_root", None):
        return Path(args.exports_root)
    return infer_exports_root(args.manifest)
