"""Build balanced, reproducible WebDataset shards from filtered EM tiles.

Writes the full transfer-bundle artifact set:
    <shard_dir>/<prefix>-NNNNNN.tar                 # the shards
    <manifests_dir>/ssl_manifest.filtered.jsonl     # exact kept records (faithful)
    <manifests_dir>/ssl_manifest.filtered.parquet   # flattened analytic table
    <manifests_dir>/shard_index.json                # per-shard sha256 + source counts
    <manifests_dir>/dataset_fingerprint.json        # content hash over kept tile_ids
    <manifests_dir>/source_distribution.csv         # per-shard + global source summary

    python -m em_ssl.tools.build_shards \
        --manifest <MASTER_MANIFEST_PATH> [--tile-root <root>] \
        --output-root <OUTPUT_ROOT>/shards/em_tiles_v0 \
        --samples-per-shard 1000 --seed 1337 --balance-shards-by-source [--min-side 512] \
        [--compress-level 6] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..data.filters import SSLTileFilter
from ..data.manifest import (
    ResolvedTile,
    build_source_run_index,
    iter_manifest,
    resolve_tile_path,
    tile_metadata,
)
from ..data.shard_writer import build_shards
from ..utils.fingerprint import dataset_fingerprint, shard_fingerprint, write_json
from .common import add_common_data_args, add_filter_args, build_filter_config, resolve_exports_root

_PARQUET_FIELDS = [
    "tile_id", "source_id", "asset_id", "source_kind", "run_id", "status",
    "width", "height", "tile_size", "effective_nm_per_px", "tissue_score",
    "artifact_fraction", "background_fraction", "tile_mean_uint8", "tile_std_uint8",
    "normalization_warning", "low_dynamic_range",
]

def _reservoir_sample_ids(manifest_path, filter_config, n: int, seed: int) -> set[str]:
    """Uniform reservoir sample of n kept tile_ids across the whole filtered corpus (1 pass)."""
    import random

    rng = random.Random(seed)
    filt = SSLTileFilter(filter_config)
    reservoir: list[str] = []
    kept = 0
    for rec in iter_manifest(manifest_path):
        if not filt(rec):
            continue
        kept += 1
        tid = str(rec.get("tile_id"))
        if len(reservoir) < n:
            reservoir.append(tid)
        else:
            j = rng.randint(0, kept - 1)  # include the k-th kept item with prob n/k
            if j < n:
                reservoir[j] = tid
    return set(reservoir)

def _derive_manifests_dir(output_root: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    parts = output_root.parts
    if "shards" in parts:
        idx = parts.index("shards")
        return Path(*parts[:idx]) / "manifests"
    return output_root

def run(args) -> dict:
    exports_root = resolve_exports_root(args)
    tile_root = Path(args.tile_root) if args.tile_root else None
    shard_dir = Path(args.output_root)
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = _derive_manifests_dir(shard_dir, args.manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    filter_config = build_filter_config(args, min_side_default=args.min_side)

    # Resolve records that carry only tile_path and no run_dir, via a source->run scan.
    source_run_index = None if args.no_source_index else build_source_run_index(exports_root)
    if source_run_index is not None:
        print(f"[build_shards] source->run index: {len(source_run_index):,} sources scanned.")

    # For a representative subset, uniform-reservoir-sample tile_ids across the whole filtered
    # corpus first (uniform random over tiles ⇒ proportional to the source distribution),
    # rather than taking the first N in manifest order (which over-represents early tiling runs).
    sample_ids: set[str] | None = None
    if args.sample_tiles:
        sample_ids = _reservoir_sample_ids(args.manifest, filter_config, args.sample_tiles, args.seed)
        print(f"[build_shards] reservoir-sampled {len(sample_ids):,} tile_ids for a representative subset.")

    filt = SSLTileFilter(filter_config)
    resolved: list[ResolvedTile] = []
    parquet_rows: list[dict] = []
    missing = 0
    filtered_jsonl = manifests_dir / "ssl_manifest.filtered.jsonl"
    with open(filtered_jsonl, "w", encoding="utf-8") as fj:
        for rec in iter_manifest(args.manifest):
            if not filt(rec):
                continue
            if sample_ids is not None and str(rec.get("tile_id")) not in sample_ids:
                continue
            # Records without run_dir need verify=True (several candidate run dirs are tried).
            verify = (not args.no_verify) or (rec.get("run_dir") is None)
            p = resolve_tile_path(rec, exports_root, tile_root, verify=verify, source_run_index=source_run_index)
            if p is None:
                missing += 1
                continue
            fj.write(json.dumps(rec, default=str) + "\n")
            meta = tile_metadata(rec)
            tid = str(rec.get("tile_id") or p.stem)
            resolved.append(ResolvedTile(tile_id=tid, path=p, source_id=str(rec.get("source_id", "unknown")), metadata=meta))
            parquet_rows.append({k: rec.get(k) for k in _PARQUET_FIELDS} | {"tile_path": str(p)})
            if args.collect_limit and sample_ids is None and len(resolved) >= args.collect_limit:
                # Stop streaming early (small/sample builds) — keeps filtered manifest == packed set.
                break

    print(f"[build_shards] filter kept {filt.kept:,}/{filt.total:,}; resolved {len(resolved):,}; missing files {missing:,}")

    if args.max_tiles and len(resolved) > args.max_tiles:
        resolved = resolved[: args.max_tiles]
        parquet_rows = parquet_rows[: args.max_tiles]
        print(f"[build_shards] capped to --max-tiles {args.max_tiles:,}")

    result = build_shards(
        resolved,
        shard_dir,
        shard_prefix=args.shard_prefix,
        samples_per_shard=args.samples_per_shard,
        seed=args.seed,
        balance_by_source=args.balance_shards_by_source,
        verify_png=args.verify_png,
        progress=not args.no_progress,
        compress_level=args.compress_level,
        num_workers=args.workers,
        sequential_read=args.sequential_read,
        read_workers=args.read_workers,
    )

    # --- artifacts ---
    shard_index = result.to_dict()
    shard_index["exports_root"] = str(exports_root)
    shard_index["filter_config"] = filt.config.to_dict()
    write_json(shard_index, manifests_dir / "shard_index.json")

    ds_fp = dataset_fingerprint(
        args.manifest,
        kept_tile_ids=[t.tile_id for t in resolved],
        filter_config=filt.config.to_dict(),
        extra={"shards": shard_fingerprint(shard_index)},
        hash_manifest_content=not args.no_hash_manifest,
    )
    write_json(ds_fp, manifests_dir / "dataset_fingerprint.json")

    _write_source_distribution(manifests_dir / "source_distribution.csv", result)
    _write_parquet(manifests_dir / "ssl_manifest.filtered.parquet", parquet_rows)

    print(
        f"[build_shards] wrote {result.num_shards} shards ({result.num_tiles:,} tiles), "
        f"skipped_missing={result.skipped_missing}, dup_ids={result.skipped_duplicate_ids}"
    )
    print(f"[build_shards] artifacts -> {manifests_dir}")
    return shard_index

def _write_source_distribution(path: Path, result) -> None:
    # One summary row per shard (sample count, distinct sources, top source) plus a GLOBAL row.
    all_sources = sorted(result.global_source_counts.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shard", "num_samples", "num_distinct_sources", "top_source", "top_source_count"])
        for s in result.shards:
            counts = s.source_counts
            top = max(counts.items(), key=lambda kv: kv[1]) if counts else ("", 0)
            w.writerow([s.name, s.num_samples, len(counts), top[0], top[1]])
        gtop = max(result.global_source_counts.items(), key=lambda kv: kv[1]) if result.global_source_counts else ("", 0)
        w.writerow(["GLOBAL", result.num_tiles, len(all_sources), gtop[0], gtop[1]])

def _write_parquet(path: Path, rows: list[dict]) -> None:
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
    except Exception as exc:
        # Fall back to CSV so the artifact still exists if pyarrow/pandas is unavailable.
        print(f"[build_shards] parquet write failed ({exc!r}); writing CSV fallback.")
        if rows:
            with open(path.with_suffix(".csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build balanced WebDataset shards from filtered EM tiles.")
    add_common_data_args(p)
    add_filter_args(p)
    p.add_argument("--shard-prefix", default="em_tiles_v0")
    p.add_argument("--samples-per-shard", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--balance-shards-by-source", action="store_true", default=False)
    p.add_argument("--manifests-dir", default=None, help="Where to write manifest artifacts (default: bundle manifests/).")
    p.add_argument("--max-tiles", type=int, default=None, help="Cap tiles after full collection (post-balance).")
    p.add_argument("--collect-limit", type=int, default=None, help="Stop streaming after N kept tiles, in manifest order (fast tiny test builds).")
    p.add_argument("--sample-tiles", type=int, default=None, help="Build a representative subset of N tiles via uniform reservoir sampling across the whole filtered corpus (proportional to sources). Preferred over --collect-limit for real subsets.")
    p.add_argument("--no-verify", action="store_true", help="Skip per-tile path existence check during collection.")
    p.add_argument("--verify-png", action="store_true", help="Decode-verify each PNG while packing (slow).")
    p.add_argument("--no-hash-manifest", action="store_true", help="Skip hashing the full manifest file in the fingerprint.")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--compress-level",
        type=int,
        default=None,
        help="Re-encode each PNG losslessly at this zlib level (0-9); default copies bytes verbatim. "
        "6 ~halves EM shard size (~1.8x) at ~160ms/tile encode, parallelized across --workers.",
    )
    p.add_argument("--workers", type=int, default=None, help="Parallel shard-packing processes (default: CPU count). With --sequential-read this is the encode-thread count instead.")
    p.add_argument(
        "--sequential-read",
        action="store_true",
        help="Read tiles once in manifest (on-disk) order via a pipelined reader->encode->writer, "
        "routing each tile into its balanced shard. Eliminates the random-read thrash of per-shard "
        "packing on spinning disks; keeps balance + png6 (byte-identical, deterministic output).",
    )
    p.add_argument(
        "--read-workers",
        type=int,
        default=8,
        help="With --sequential-read: number of concurrent disk readers (overlaps per-file open "
        "latency; 4-8 is the HDD sweet spot, default 8). --workers stays the encode-thread count.",
    )
    args = p.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()
