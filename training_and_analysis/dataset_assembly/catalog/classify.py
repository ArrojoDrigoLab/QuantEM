"""LLM triage of candidates the deterministic rule engine could not settle.

    python -m catalog.classify --in triaged/review.jsonl --out classified.jsonl

Reads rows written by `catalog.triage`, renders the classification prompt for
each, calls the model, validates the response against the output schema, and
writes one classification per candidate.

The model is invoked as a subprocess with the prompt on stdin and the schema
supplied as `--output-schema`, so the response is structurally constrained. The
prompt is loaded from `../prompts/01_dataset_classification.txt` rather than
embedded here — that file is the prompt of record.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .jsonl import read_jsonl, write_jsonl
from .schema import REVIEW_RESPONSE_SCHEMA, validate_classification

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "01_dataset_classification.txt"

DEFAULT_MODEL = os.environ.get("EM_CATALOG_LLM_MODEL", "gpt-5.5")
DEFAULT_BINARY = os.environ.get("EM_CATALOG_LLM_BIN", "codex")
DEFAULT_REASONING = os.environ.get("EM_CATALOG_LLM_REASONING", "medium")
DEFAULT_TIMEOUT = 900


def render_prompt(template: str, **values) -> str:
    """Substitute the {placeholders}.

    str.replace, not str.format: the prompt embeds literal JSON braces that
    format() would try to interpret.
    """
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def build_prompt(template: str, row: dict) -> str:
    candidate = row.get("candidate") or row
    triage = row.get("deterministic_triage") or {}
    evidence = row.get("evidence") or []
    dumps = lambda v: json.dumps(v, ensure_ascii=True, indent=2, sort_keys=True)  # noqa: E731
    return render_prompt(
        template,
        candidate_json=dumps(candidate),
        deterministic_triage_json=dumps(triage),
        escalation_context_json=dumps(row.get("escalation_context") or {}),
        evidence_json=dumps(evidence),
        candidate_id=json.dumps(candidate.get("candidate_id") or candidate.get("source_record_id") or ""),
        output_schema_json=dumps(REVIEW_RESPONSE_SCHEMA),
    )


def call_model(prompt: str, *, binary: str, model: str, reasoning: str,
               timeout: int, search: bool = False) -> dict:
    """Run the model CLI with a constrained output schema. Returns the parsed object."""
    with tempfile.TemporaryDirectory() as tmp:
        schema_path = Path(tmp) / "schema.json"
        answer_path = Path(tmp) / "answer.json"
        schema_path.write_text(json.dumps(REVIEW_RESPONSE_SCHEMA), encoding="utf8")

        command = [
            binary, "exec",
            "--model", model,
            "--config", f"model_reasoning_effort={reasoning}",
            "--output-schema", str(schema_path),
            "--output-last-message", str(answer_path),
        ]
        if search:
            command.append("--search")
        command.append("-")

        completed = subprocess.run(
            command, input=prompt, text=True, capture_output=True, timeout=timeout)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{binary} exited {completed.returncode}: "
                f"{(completed.stderr or '').strip()[:400]}")
        if not answer_path.exists():
            raise RuntimeError(f"{binary} wrote no response")
        return json.loads(answer_path.read_text(encoding="utf8"))


def classify_row(row: dict, template: str, **model_kwargs) -> dict:
    candidate = row.get("candidate") or row
    candidate_id = candidate.get("candidate_id") or candidate.get("source_record_id") or ""
    try:
        response = call_model(build_prompt(template, row), **model_kwargs)
    except Exception as exc:
        return {"candidate_id": candidate_id, "status": "error",
                "error": f"{type(exc).__name__}: {exc}"}

    problems = validate_classification(response, context=f"candidate {candidate_id}")
    if problems:
        return {"candidate_id": candidate_id, "status": "invalid",
                "problems": problems, "response": response}
    return {"candidate_id": candidate_id, "status": "ok", "classification": response}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="review.jsonl from catalog.triage")
    ap.add_argument("--out", required=True, help="classification JSONL to write")
    ap.add_argument("--prompt", default=str(PROMPT_PATH), help="prompt template to render")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--binary", default=DEFAULT_BINARY, help="model CLI to invoke")
    ap.add_argument("--reasoning", default=DEFAULT_REASONING,
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--search", action="store_true",
                    help="allow the model web access; off for the first pass")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows; 0 for all")
    args = ap.parse_args(argv)

    template = Path(args.prompt).expanduser().read_text(encoding="utf8")
    rows = list(read_jsonl(Path(args.inp).expanduser()))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("nothing to classify", file=sys.stderr)
        return 1

    model_kwargs = dict(binary=args.binary, model=args.model, reasoning=args.reasoning,
                        timeout=args.timeout, search=args.search)
    results, counts = [], {"ok": 0, "invalid": 0, "error": 0}
    for i, row in enumerate(rows, 1):
        result = classify_row(row, template, **model_kwargs)
        counts[result["status"]] += 1
        results.append(result)
        verdict = (result.get("classification") or {}).get("eligibility_status", result["status"])
        print(f"[{i}/{len(rows)}] {result['candidate_id'][:52]:52s} {verdict}")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, results)
    print(f"\n{counts['ok']} classified, {counts['invalid']} invalid, "
          f"{counts['error']} errored -> {out_path}")
    return 0 if not counts["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
