"""Join per-asset metadata onto a tile manifest.

    python join_asset_metadata.py --manifest manifest.jsonl --asset-meta assets.csv \
        --out manifest_enriched.jsonl

Tiles carry only what can be measured from the pixels. Facts about the asset they
came from — licence, imaging modality, organ, pixel size — live in the dataset
catalogue, and downstream training needs them per tile: to filter the corpus by
licence, and to condition on or stratify by acquisition metadata.

The metadata CSV is keyed by `source_id`; every other column is copied onto each
tile of that asset. Tiles whose `source_id` is absent from the CSV are kept
unchanged unless `--require-match` is given.

    source_id,license,modality,organ,species,effective_nm_per_px,source_kind
    liver_fibsem,CC BY 4.0,FIB-SEM,Liver,Mus musculus,8,public
    islet_tem,CC0 1.0,TEM,Pancreas,Homo sapiens,2.2,internal
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# Columns that are numeric where present; everything else is copied as text.
NUMERIC = {"effective_nm_per_px", "nm_per_px", "z_nm"}


def load_metadata(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    # utf-8-sig: spreadsheet exports routinely carry a BOM, which would otherwise
    # turn the first column name into "﻿source_id".
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "source_id" not in reader.fieldnames:
            raise SystemExit(f"{path}: needs a 'source_id' column; got {reader.fieldnames}")
        for row in reader:
            key = (row.get("source_id") or "").strip()
            if not key:
                continue
            fields = {}
            for col, value in row.items():
                if col == "source_id":
                    continue
                value = (value or "").strip()
                if value == "":
                    continue
                if col in NUMERIC:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                fields[col] = value
            rows[key] = fields
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--manifest", required=True, help="tile manifest from build_manifest.py")
    ap.add_argument("--asset-meta", required=True, help="CSV keyed by source_id")
    ap.add_argument("--out", required=True, help="enriched manifest to write")
    ap.add_argument("--require-match", action="store_true",
                    help="fail if any tile's source_id is missing from the CSV")
    ap.add_argument("--overwrite", action="store_true",
                    help="let CSV columns replace fields already on the tile record")
    args = ap.parse_args(argv)

    meta = load_metadata(Path(args.asset_meta).expanduser())
    manifest = Path(args.manifest).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    unmatched: set[str] = set()
    added: Counter = Counter()

    with manifest.open(encoding="utf8") as src, out_path.open("w", encoding="utf8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            counts["tiles"] += 1
            fields = meta.get(str(record.get("source_id", "")))
            if fields is None:
                counts["unmatched"] += 1
                unmatched.add(str(record.get("source_id", "")))
            else:
                counts["matched"] += 1
                for key, value in fields.items():
                    if key in record and not args.overwrite:
                        continue
                    record[key] = value
                    added[key] += 1
            dst.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    print(f"{counts['tiles']} tiles: {counts['matched']} matched, {counts['unmatched']} unmatched")
    if added:
        print("fields added: " + ", ".join(f"{k} ({n})" for k, n in added.most_common()))
    if unmatched:
        shown = sorted(unmatched)[:8]
        print(f"{len(unmatched)} source_id(s) absent from the CSV: {', '.join(shown)}"
              + (" …" if len(unmatched) > 8 else ""), file=sys.stderr)
        if args.require_match:
            return 1
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
