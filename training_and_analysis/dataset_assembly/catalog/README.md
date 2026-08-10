# `catalog/`

Discovery of public EM datasets: querying each repository's API, triaging candidates with a
deterministic rule engine, and referring the undecided ones to an LLM.

---

## Usage

```bash
python -m catalog.scan --all --out candidates/           # query the repositories
python -m catalog.triage --in candidates/ --out triaged/ # deterministic rule engine
python -m catalog.classify --in triaged/review.jsonl --out classified.jsonl
```

Each stage reads the previous stage's output directory. `scan` needs network access;
`triage` is offline; `classify` invokes a model CLI.

## Files

| File | What it is |
|---|---|
| `scan.py` | Runs one or all scanners, writes candidates as JSONL |
| `triage.py` | Applies the rule engine, splits into eligible / ineligible / review |
| `classify.py` | Renders the classification prompt, calls the model, validates the response |
| `eligibility.py` | The deterministic keyword rule engine |
| `schema.py` | Classification output schema and its validator |
| `models.py` | The `Candidate` record and `ScannerError` |
| `http.py` | Shared GET/POST with rate-limit backoff |
| `jsonl.py` | JSONL read/write |
| `sources/` | One module per repository |
| `tests/` | Offline tests against recorded API responses |

## Repositories

| Module | Covers |
|---|---|
| `sources/empiar.py` | EMPIAR |
| `sources/openorganelle.py` | OpenOrganelle (Janelia) |
| `sources/bossdb.py` | BossDB |
| `sources/webknossos.py` | webKnossos, including institutional instances |
| `sources/zenodo_dump.py` | Zenodo, via its published full-history metadata dump |
| `sources/portals.py` | FigShare (search and OAI-PMH), Zenodo, BioStudies, Dryad, Mendeley via DataCite |

`portals.py` also carries handlers for DataCite at large, Dataverse and Hugging Face. Those were
searched and contributed no datasets to the published corpus; they are retained as part of the
documented search coverage.

## The three stages

**Scan.** Each scanner queries one API and returns `Candidate` records with whatever the repository
exposes: title, landing and download URLs, DOIs, declared modality, sample description, file formats
and the raw metadata payload. Scanners hold no state and apply no eligibility judgement. Adapter
failures become `ScannerError` records rather than exceptions, so one dead API does not lose a sweep.

**Triage.** `eligibility.py` runs an ordered set of rules over each candidate's text: exclusion
patterns first, then source-specific rules, then a matrix of qualifying-modality against
intracellular-context terms. It emits `eligible`, `ineligible` or `uncertain`, with a confidence and
the reasons. Records it cannot settle are routed onward.

**Classify.** Undecided candidates go to the model with the rule engine's verdict attached. The
prompt is loaded from
[`../prompts/01_dataset_classification.txt`](../prompts/01_dataset_classification.txt) — that file is
the prompt of record, not a copy — and the response is constrained by
[`../prompts/01_dataset_classification.schema.json`](../prompts/01_dataset_classification.schema.json)
and validated before it is written.

Model, binary and reasoning effort are set by `--model`, `--binary` and `--reasoning`, or by
`EM_CATALOG_LLM_MODEL`, `EM_CATALOG_LLM_BIN` and `EM_CATALOG_LLM_REASONING`.

## Tests

```bash
cd ..            # dataset_assembly/
python -m unittest discover -s catalog/tests -t .
```

Every test is offline: transports are patched or injected, and API responses come from recorded
fixtures. No network, no database.

## Notes

- `sources/openorganelle.py` inlines OpenOrganelle's public anonymous API key, which is published in
  their own web frontend. It is not a secret and grants only public read access. Override with
  `OPENORGANELLE_ANON_KEY`.
- The catalog itself is not distributed. This is the code that discovers and triages candidates; the
  resulting dataset inventory is published with the manuscript and browsable in
  [`dataset_directory/`](../../../dataset_directory/).
