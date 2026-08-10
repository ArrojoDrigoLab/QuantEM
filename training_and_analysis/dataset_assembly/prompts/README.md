# `prompts/`

The LLM prompts used to build the EM image corpus. One file per distinct call, each holding the
complete prompt as sent.

| File | Call | Model | Determines |
|---|---|---|---|
| `01_dataset_classification.txt` | repository candidate triage | GPT-5.5 | eligibility — is this intracellular biological EM with retrievable image data |
| `02_zenodo_second_pass.txt` | Zenodo uncertain-entry review | GPT-5.5 | eligibility for candidates the first pass could not resolve |
| `03_asset_metadata.txt` | per-dataset metadata research | GPT-5.5 | resolution, organ, tissue, modality |
| `04_organ_tissue_vocabulary.txt` | organ/tissue vocabulary standardization | Claude Opus 4.6–4.8 | mapping of free-text tissue values onto a controlled vocabulary |

`01` runs after the deterministic keyword rule engine, on candidates the rule engine could not settle.
`03` and `04` run on datasets already accepted into the corpus and produce the facets exposed in the
dataset directory.

## Placeholders

`{name}` marks runtime substitution. Each file is otherwise verbatim.

| Placeholder | Holds |
|---|---|
| `{candidate_json}` | the scraped candidate record: source, record id, title, URLs, DOIs, file formats, raw repository metadata |
| `{deterministic_triage_json}` | the rule engine's verdict and matched keywords |
| `{escalation_context_json}` | prior-pass outcome, present only on re-review |
| `{evidence_json}` | summaries of files fetched to a temporary directory for inspection |
| `{output_schema_json}` | the JSON Schema the response is validated against |
| `{candidate_id}`, `{record_url}`, `{api_url}` | identifiers for the item under review |
| `{dataset_id}`, `{experiment_name}`, `{dataset_name}` | dataset identity |
| `{lookup_kind}`, `{lookup_url}`, `{source_urls}` | where to look |
| `{dataset_dois}`, `{publication_dois}` | known metadata |
| `{assets_json}` | per-asset current tags and which groups still need values |

`01_dataset_classification.txt` holds four parts concatenated in the order the call sends them — a
system line, the classifier contract, the runner decision rules, and the data blocks.
`catalog.classify` reads the whole file as one template and substitutes the placeholders into it.
