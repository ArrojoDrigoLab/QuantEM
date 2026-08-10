"""Command-line entry point for the dataset-directory export.

    python -m quantem_directory build   --extract <dir> --urls <csv> --out ../data
    python -m quantem_directory thumbs  --tile-index <csv> --tile-root <dir> --out ../thumbs
    python -m quantem_directory verify  --data ../data

Every path is an argument; nothing about a particular machine is baked in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import build as build_module
from . import extract as extract_module
from . import verify as verify_module

HERE = Path(__file__).resolve().parents[2]
DEFAULT_DATA = HERE.parent / "data"
DEFAULT_EXCLUDED = HERE / "excluded.json"
DEFAULT_OVERRIDES = HERE / "vocabulary_overrides.json"
DEFAULT_ACCESSIONS = HERE / "dataset_links.json"


def _load_json(path: Path, fallback: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return fallback


def _load_excluded(path: Path) -> dict:
    return _load_json(path, {"dataset_names": [], "url_substrings": []})


def _cmd_build(args: argparse.Namespace) -> int:
    excluded = _load_excluded(Path(args.excluded))
    corpus = extract_module.load(
        Path(args.extract),
        urls_csv=Path(args.urls) if args.urls else None,
        exclude_datasets=set(excluded.get("dataset_names", [])),
        vocabulary_overrides=_load_json(Path(args.overrides), {}),
        link_overrides=_load_json(Path(args.accessions), {}).get("datasets", {}),
    )

    thumb_ids = None
    thumb_meta = None
    index_path = Path(args.thumbs) / "index.json" if args.thumbs else None
    if index_path and index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        thumb_ids = list(index["assets"])
        thumb_meta = index["meta"]
        print(f"thumbnail index: {len(thumb_ids)} assets")
    elif args.thumbs:
        print(f"note: no thumbnail index at {index_path} — building with no thumbnails", file=sys.stderr)

    report = build_module.build(
        corpus,
        Path(args.out),
        thumb_ids=thumb_ids,
        thumb_meta=thumb_meta,
        source_snapshot=args.snapshot,
    )

    print("\ncorpus")
    for key, value in report["counts"].items():
        print(f"  {key:18s} {value:>8,}")
    print("\nartifacts")
    for name, size in sorted(report["sizes"].items()):
        print(f"  {name:18s} {size / 1024:>8.1f} KiB")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        notes = verify_module.verify(
            Path(args.data),
            expected_counts=Path(args.expected) if args.expected else None,
            thumbs_dir=Path(args.thumbs) if args.thumbs else None,
            excluded=_load_excluded(Path(args.excluded)),
        )
    except verify_module.VerificationFailed as failure:
        print("VERIFY FAILED\n" + str(failure), file=sys.stderr)
        return 1
    for note in notes:
        print(f"  ok  {note}")
    print("\nverify passed")
    return 0


def _cmd_thumbs(args: argparse.Namespace) -> int:
    from . import thumbs as thumbs_module

    return thumbs_module.run(args)


def _cmd_pack(args: argparse.Namespace) -> int:
    from . import thumbs as thumbs_module

    return thumbs_module.pack(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantem_directory", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="write the published JSON artifacts")
    p_build.add_argument("--extract", required=True, help="directory holding the corpus extract CSVs")
    p_build.add_argument("--urls", help="dataset URL/DOI CSV (three separate DOI columns)")
    p_build.add_argument("--out", default=str(DEFAULT_DATA), help="output directory")
    p_build.add_argument("--thumbs", help="thumbnail directory, to pick up index.json")
    p_build.add_argument("--excluded", default=str(DEFAULT_EXCLUDED))
    p_build.add_argument("--overrides", default=str(DEFAULT_OVERRIDES),
                         help="facet vocabulary corrections")
    p_build.add_argument("--accessions", default=str(DEFAULT_ACCESSIONS),
                         help="dataset links to apply as depositions complete")
    p_build.add_argument("--snapshot", default="", help="source snapshot date, e.g. 2026-08-05")
    p_build.set_defaults(func=_cmd_build)

    p_verify = sub.add_parser("verify", help="gate the published artifacts")
    p_verify.add_argument("--data", default=str(DEFAULT_DATA))
    p_verify.add_argument("--expected",
                          help="optional JSON of expected counts, for a one-off check")
    p_verify.add_argument("--thumbs")
    p_verify.add_argument("--excluded", default=str(DEFAULT_EXCLUDED))
    p_verify.set_defaults(func=_cmd_verify)

    p_thumbs = sub.add_parser("thumbs", help="render the thumbnail set")
    p_thumbs.add_argument("--tile-index", required=True, help="canonical tile index CSV")
    p_thumbs.add_argument("--tile-root", required=True, help="root the tile index paths resolve under")
    p_thumbs.add_argument("--extract", required=True, help="corpus extract directory")
    p_thumbs.add_argument("--out", required=True, help="thumbnail output directory")
    p_thumbs.add_argument("--px", type=int, default=256)
    p_thumbs.add_argument("--quality", type=int, default=75)
    p_thumbs.add_argument("--workers", type=int, default=8)
    p_thumbs.add_argument("--limit", type=int, help="stop after N assets (for a smoke run)")
    p_thumbs.add_argument("--only-missing", action="store_true", default=True)
    p_thumbs.add_argument("--fallback-preview", action="append", default=[],
                          help="directory of <id>.png previews; repeatable")
    p_thumbs.add_argument("--fallback-images", help="directory of <asset-id>/ raster folders")
    p_thumbs.add_argument("--max-source-mb", type=int, default=400,
                          help="skip fallback rasters larger than this")
    p_thumbs.add_argument("--only-ids", help="file of asset ids to (re-)render, one per line")
    p_thumbs.add_argument("--normalize-tiles", action="store_true",
                          help="stretch canonical tiles too; for repairing flat thumbnails")
    p_thumbs.set_defaults(func=_cmd_thumbs)

    p_pack = sub.add_parser("pack", help="bundle the thumbnails into a release archive")
    p_pack.add_argument("--out", required=True, help="thumbnail directory to pack")
    p_pack.add_argument("--archive", required=True, help="archive path to write (.tar.gz)")
    p_pack.add_argument("--release-tag", default="directory-thumbs-v1")
    p_pack.set_defaults(func=_cmd_pack)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
