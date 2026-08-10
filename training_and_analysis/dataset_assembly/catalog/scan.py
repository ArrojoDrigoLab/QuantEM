"""Run repository scanners and write candidate records.

    python -m catalog.scan --source empiar --out candidates/
    python -m catalog.scan --all --out candidates/ --limit 500

Each scanner queries one public repository's API and returns candidate records.
Candidates are written as JSONL, one file per source; scanner errors are written
alongside so a failing adapter is recorded rather than silently dropped. A source
that hands back a cursor also gets a `<source>.cursor.json` for the next sweep.

Nothing here decides eligibility. Run `catalog.triage` next.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .jsonl import write_jsonl
from .sources import SOURCE_NAMES, run_scanner


def scan_source(source: str, out_dir: Path, *, root: Path, since: str | None,
                query: str | None, limit: int) -> dict:
    result = run_scanner(
        source, root=root, since=since, query=query, limit=limit, cursor=None)

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / f"{source}.candidates.jsonl"
    write_jsonl(candidates_path, (c.to_dict() for c in result.candidates))

    summary = {
        "source": source,
        "candidates": len(result.candidates),
        "errors": len(result.errors),
        "cursor_complete": result.cursor_complete,
        "candidates_path": str(candidates_path),
    }
    if result.errors:
        errors_path = out_dir / f"{source}.errors.jsonl"
        write_jsonl(errors_path, (e.to_dict() for e in result.errors))
        summary["errors_path"] = str(errors_path)
    if result.cursor:
        (out_dir / f"{source}.cursor.json").write_text(
            json.dumps(result.cursor, indent=2, sort_keys=True), encoding="utf8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--source", choices=SOURCE_NAMES, help="one repository to scan")
    pick.add_argument("--all", action="store_true", help="scan every repository in turn")
    ap.add_argument("--out", required=True, help="directory for candidate JSONL")
    ap.add_argument("--root", default=".",
                    help="working directory for scanners that cache downloads")
    ap.add_argument("--since", help="ISO date; restrict to records modified on or after")
    ap.add_argument("--query", help="free-text query, where the API supports one")
    ap.add_argument("--limit", type=int, default=0, help="max records per source; 0 for no limit")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).expanduser()
    root = Path(args.root).expanduser()
    sources = list(SOURCE_NAMES) if args.all else [args.source]

    summaries, failed = [], 0
    for source in sources:
        try:
            summary = scan_source(source, out_dir, root=root, since=args.since,
                                  query=args.query, limit=args.limit)
        except Exception as exc:  # a dead API should not stop the sweep
            print(f"{source:16s} FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
            continue
        summaries.append(summary)
        note = f", {summary['errors']} errors" if summary["errors"] else ""
        more = "" if summary["cursor_complete"] else "  (more available)"
        print(f"{source:16s} {summary['candidates']:6d} candidates{note}{more}")

    (out_dir / "scan_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf8")
    total = sum(s["candidates"] for s in summaries)
    print(f"\n{total} candidates from {len(summaries)} source(s) -> {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
