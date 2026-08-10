# `literature_screen/`

The bibliographic screen for data outreach: finding published EM studies whose underlying
image data was not publicly deposited.

---

## Usage

```python
from literature_screen.literature import scan_literature

result = scan_literature(since="2011-01-01", limit=500)
```

Queries four bibliographic sources — PubMed E-utilities, OpenAlex, Crossref and Europe PMC — using
the query groups in `sources.json`, then keeps records whose title, abstract, journal or linked data
URLs match at least one term from the modality (`tem`, `sbf_sem`, `fib_sem`, `volume_em`) or context
(`ultrastructure`, `organelle`) term sets. For PubMed it additionally resolves article metadata and
linked data repositories, so a record carries whatever deposition signal the index exposes.

Records with a public repository deposit become candidate public datasets; records without one are
the outreach targets.

`since` sets the publication cutoff. Without it each source falls back to the rolling
`date_window_days` in `sources.json` (3650 days), and `full_history=True` removes the window
entirely.

## Files

| File | What it is |
|---|---|
| `literature.py` | The scanner: per-source adapters, relevance filtering, data-link extraction |
| `sources.json` | Query configuration: query groups, term sets, per-source limits, date window |
| `tests/` | Offline tests against recorded API responses |

The scanner reads the `query_groups`, `query_terms` and `per_source` blocks of `sources.json`.
`query_terms` carries the modality, ultrastructure/organelle context, data-availability,
repository-name and dataset term sets; `source_journal_groups` records the curated journal groups.

## Tests

```bash
cd ..            # dataset_assembly/
python -m unittest discover -s literature_screen/tests -t .
```

Offline — every transport is patched and responses come from recorded fixtures.
