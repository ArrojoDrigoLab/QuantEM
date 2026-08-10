"""Deterministic eligibility triage over scanned candidates.

    python -m catalog.triage --in candidates/ --out triaged/

Runs the keyword rule engine (`catalog.eligibility`) over every candidate and
splits the result three ways:

    eligible.jsonl    the rule engine is confident the record qualifies
    ineligible.jsonl  the rule engine is confident it does not
    review.jsonl      undecided; these go to `catalog.classify` for LLM review

Each output row carries the candidate alongside the rule engine's verdict, which
`catalog.classify` passes to the model as the deterministic triage block.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .eligibility import classify_candidate
from .jsonl import read_jsonl, write_jsonl
from .models import Candidate


def triage(rows) -> tuple[list, list, list, Counter]:
    eligible, ineligible, review = [], [], []
    counts = Counter()
    for row in rows:
        candidate = Candidate.from_dict(row)
        verdict = classify_candidate(candidate)
        status = str(verdict.get("eligibility_status"))
        record = {"candidate": candidate.to_dict(), "deterministic_triage": verdict}
        counts[status] += 1
        if verdict.get("needs_codex_review") or status == "uncertain":
            counts["routed_to_review"] += 1
            review.append(record)
        elif status == "eligible":
            eligible.append(record)
        else:
            ineligible.append(record)
    return eligible, ineligible, review, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="candidate JSONL file, or a directory of *.candidates.jsonl")
    ap.add_argument("--out", required=True, help="directory for the split files and the run summary")
    args = ap.parse_args(argv)

    src = Path(args.inp).expanduser()
    files = sorted(src.glob("*.candidates.jsonl")) if src.is_dir() else [src]
    if not files:
        raise SystemExit(f"no candidate files under {src}")

    rows = [row for f in files for row in read_jsonl(f)]
    eligible, ineligible, review, counts = triage(rows)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "eligible.jsonl", eligible)
    write_jsonl(out_dir / "ineligible.jsonl", ineligible)
    write_jsonl(out_dir / "review.jsonl", review)
    (out_dir / "triage_summary.json").write_text(
        json.dumps(dict(counts), indent=2, sort_keys=True), encoding="utf8")

    print(f"{len(rows)} candidates from {len(files)} file(s)")
    print(f"  eligible    {len(eligible):6d}")
    print(f"  ineligible  {len(ineligible):6d}")
    print(f"  review      {len(review):6d}   -> catalog.classify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
