"""Compute EM corpus intensity mean/std over real tile pixels.

Samples filtered tiles, accumulates pixel sum / sum-of-squares in [0,1], and writes the
canonical tile_intensity_stats.json schema, which ``em_ssl.config.schema`` reads back from
a data-root bundle to set a run's ``mean``/``std``:

    {
      "computed_from_tiles": N, "mean_01": .., "std_01": ..,
      "recommended_dino_mean_single_channel": [..], "recommended_dino_std_single_channel": [..],
      "recommended_dino_mean_rgb_replicated": [..,..,..], ...   # convenience only; RGB is never used
      "provenance": {...}
    }

These single-channel values are frozen into each run's resolved config; ImageNet statistics
are never used for the EM corpus.

    python -m em_ssl.tools.compute_tile_stats \
        --manifest <MASTER_MANIFEST_PATH> --output-root <OUTPUT_ROOT>/data_prep \
        [--sample 4000] [--all] [--pixel-stride 2] [--seed 1337]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from ..data.filters import SSLTileFilter
from ..data.manifest import build_source_run_index, iter_manifest, resolve_tile_path
from .common import add_common_data_args, add_filter_args, build_filter_config, resolve_exports_root

def run(args) -> dict:
    import numpy as np
    from PIL import Image

    exports_root = resolve_exports_root(args)
    tile_root = Path(args.tile_root) if args.tile_root else None
    filt = SSLTileFilter(build_filter_config(args, min_side_default=args.min_side))

    source_run_index = None if args.no_source_index else build_source_run_index(exports_root)
    if source_run_index is not None:
        print(f"[compute_tile_stats] source->run index: {len(source_run_index):,} sources scanned.")

    kept = []
    for rec in iter_manifest(args.manifest):
        if filt(rec):
            # verify=False is fast; the index gives the right run dir for single-run sources
            # (the vast majority). Rare multi-run mis-picks fail the PIL read and are skipped.
            p = resolve_tile_path(rec, exports_root, tile_root, verify=False, source_run_index=source_run_index)
            if p is not None:
                kept.append((rec.get("tile_id"), p))

    rng = random.Random(args.seed)
    if not args.all and args.sample < len(kept):
        sampled = rng.sample(kept, args.sample)
    else:
        sampled = kept

    stride = max(1, int(args.pixel_stride))
    count = 0
    s = 0.0
    s2 = 0.0
    used = 0
    failed = 0
    for tid, p in sampled:
        try:
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.float64)
        except Exception:
            failed += 1
            continue
        if stride > 1:
            arr = arr[::stride, ::stride]
        x = arr.ravel() / 255.0
        count += x.size
        s += float(x.sum())
        s2 += float((x * x).sum())
        used += 1

    mean = s / max(count, 1)
    var = max(s2 / max(count, 1) - mean * mean, 0.0)
    std = math.sqrt(var)

    stats = {
        "computed_from_tiles": used,
        "failed_tiles": failed,
        "total_pixels": count,
        "mean_01": round(mean, 6),
        "std_01": round(std, 6),
        "recommended_dino_mean_single_channel": [round(mean, 6)],
        "recommended_dino_std_single_channel": [round(std, 6)],
        "recommended_dino_mean_rgb_replicated": [round(mean, 6)] * 3,
        "recommended_dino_std_rgb_replicated": [round(std, 6)] * 3,
        "provenance": {
            "manifest": str(args.manifest),
            "num_kept_tiles": len(kept),
            "sampled": len(sampled),
            "pixel_stride": stride,
            "seed": args.seed,
            "filter_config": filt.config.to_dict(),
            "note": "Single-channel EM corpus stats over real pixels. Do not use ImageNet mean/std.",
        },
    }

    if args.output_root:
        out = Path(args.output_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tile_intensity_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[compute_tile_stats] mean_01={stats['mean_01']} std_01={stats['std_01']} "
              f"(from {used} tiles) -> {out/'tile_intensity_stats.json'}")
    return stats

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Compute EM corpus intensity mean/std.")
    add_common_data_args(p)
    add_filter_args(p)
    p.add_argument("--sample", type=int, default=4000, help="Number of tiles to sample for stats.")
    p.add_argument("--all", action="store_true", help="Use all filtered tiles (slow).")
    p.add_argument("--pixel-stride", type=int, default=2, help="Subsample pixels by this stride for speed.")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main()
