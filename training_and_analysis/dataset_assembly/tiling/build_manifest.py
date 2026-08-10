"""Collect tile sidecars into the pretraining manifest.

    python build_manifest.py --tiles tiles/ --out manifest.jsonl
    python build_manifest.py --tiles tiles/ --out manifest.jsonl --summary composition.json

Reads every `.json` sidecar written by `tile_asset.py` under `<tiles>/tiles/` and
emits one JSON line per accepted tile. That manifest is what the shard builder
(`em_ssl.tools.build_shards`) packs the pretraining shards from.

With `--summary`, also writes the corpus composition: tile and asset counts broken
down by dimensionality and by any per-asset metadata supplied via `--asset-meta`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

MANIFEST_FIELDS = [
    # identity and placement
    "tile_id", "source_id", "tile_path", "z", "z_source_index",
    "x", "y", "width", "height", "tile_size", "stride",
    # acceptance and content scores
    "status", "tissue_score", "non_background_fraction",
    "texture_fraction", "gradient_fraction", "artifact_fraction",
    # storage and normalization — downstream training filters on these,
    # so they belong in the default manifest, not only under --full-records
    "tile_storage_dtype", "raw_dtype", "low_dynamic_range",
    "normalization_warning", "normalization_hash", "scoring",
]


def iter_sidecars(tiles_root: Path):
    root = Path(tiles_root)
    if (root / "tiles").is_dir():
        root = root / "tiles"
    for sidecar in sorted(root.rglob("*.json")):
        try:
            record = json.loads(sidecar.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping {sidecar}: {exc}", file=sys.stderr)
            continue
        if record.get("status") == "accepted":
            yield record


def load_asset_meta(path: Path | None) -> dict[str, dict]:
    """Optional CSV keyed by source_id; any other column becomes a summary facet."""
    if not path:
        return {}
    meta = {}
    with Path(path).open(encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            key = row.get("source_id")
            if key:
                meta[key] = {k: v for k, v in row.items() if k != "source_id"}
    return meta


def summarize(records: list[dict], meta: dict[str, dict]) -> dict:
    tiles_by_source = Counter(r["source_id"] for r in records)
    dims = Counter("3D" if r.get("z") is not None else "2D" for r in records)
    assets_by_dim = Counter()
    seen = set()
    for r in records:
        key = r["source_id"]
        if key in seen:
            continue
        seen.add(key)
        assets_by_dim["3D" if r.get("z") is not None else "2D"] += 1

    out = {
        "tiles": len(records),
        "assets": len(tiles_by_source),
        "tiles_by_dimensionality": dict(dims),
        "assets_by_dimensionality": dict(assets_by_dim),
        "tiles_per_asset": {
            "min": min(tiles_by_source.values()) if tiles_by_source else 0,
            "max": max(tiles_by_source.values()) if tiles_by_source else 0,
            "at_cap": sum(1 for v in tiles_by_source.values()
                          if v == max(tiles_by_source.values() or [0])),
        },
    }

    facets = sorted({k for m in meta.values() for k in m})
    for facet in facets:
        tile_counts, asset_counts = Counter(), Counter()
        for source_id, n in tiles_by_source.items():
            value = (meta.get(source_id) or {}).get(facet) or "Unknown"
            tile_counts[value] += n
            asset_counts[value] += 1
        out[f"tiles_by_{facet}"] = dict(tile_counts.most_common())
        out[f"assets_by_{facet}"] = dict(asset_counts.most_common())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tiles", required=True, help="tile output root from tile_asset.py")
    ap.add_argument("--out", required=True, help="manifest .jsonl to write")
    ap.add_argument("--summary", help="also write corpus composition to this .json")
    ap.add_argument("--asset-meta",
                    help="optional CSV keyed by source_id; extra columns become summary facets")
    ap.add_argument("--full-records", action="store_true",
                    help="write the complete sidecar record rather than the manifest fields")
    args = ap.parse_args(argv)

    records = list(iter_sidecars(Path(args.tiles).expanduser()))
    if not records:
        print(f"no accepted tiles under {args.tiles}", file=sys.stderr)
        return 1

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as fh:
        for record in records:
            row = record if args.full_records else {
                k: record.get(k) for k in MANIFEST_FIELDS if k in record
            }
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    n_assets = len({r["source_id"] for r in records})
    print(f"{len(records)} tiles across {n_assets} assets -> {out_path}")

    if args.summary:
        meta = load_asset_meta(Path(args.asset_meta).expanduser() if args.asset_meta else None)
        summary = summarize(records, meta)
        summary_path = Path(args.summary).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
        print(f"composition -> {summary_path}")
        for key in ("tiles_by_dimensionality", "assets_by_dimensionality"):
            print(f"  {key}: {summary[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
