# `dataset_assembly/`

Assembly of the EM image corpus from three streams — public repositories, an author outreach
campaign, and newly released in-house data — and the tiling of assembled assets into the tiles used
for foundation-model pretraining.

---

## Contents

| Item | What it is |
|---|---|
| [`prompts/`](prompts/) | The LLM prompts, verbatim, one file per call |
| [`catalog/`](catalog/) | Repository scanners, the deterministic keyword rule engine, and the LLM triage runner |
| [`literature_screen/`](literature_screen/) | The bibliographic screen behind the outreach campaign |
| [`tiling/`](tiling/) | Tile geometry, tissue-content filtering, per-source capping, normalization, and the manifest |

Each has its own README. One environment covers all four:

```bash
conda env create -f environment.yml
```

`catalog/classify.py` additionally needs a model CLI on PATH; see [`catalog/README.md`](catalog/README.md).
Every script takes its data roots as arguments — there is no path configuration file.

## The three streams

**Public repositories.** WebKnossos, BossDB, OpenOrganelle, EMPIAR, BioImage Archive, Zenodo, Dryad,
and FigShare are queried through their APIs. Candidate entries are triaged first by a deterministic
keyword rule engine, then by a large language model assessing whether the entry contains EM images
of biological origin with intracellular features — excluding surface morphology, viral particles,
and other non-intracellular image types. Candidates are manually reviewed before addition.
Result: 501 datasets, 12,863 2D images, 1,715 3D acquisitions.

**Outreach.** A bibliographic screen identifies published EM studies whose underlying images are not
publicly deposited, grouped by corresponding author. Authors are contacted with a request to deposit
their data publicly and contribute it. Result: 117 datasets, 2,780 2D images, 4 3D acquisitions,
deposited at the EMBL BioImage Archive.

**In-house.** Newly released acquisitions: 37 datasets, 464 2D images, 1 3D acquisition.

## Tiling

Assets are cut into minimally-overlapping 2048 × 2048 tiles. Each candidate tile is scored for
tissue content and rejected below threshold, so blank resin, support film, and empty space do not
enter training. Tiles are rescaled to uint8 using intensity percentiles estimated per source for 2D
assets and per plane for 3D volumes, and each source is capped so large 3D volumes cannot dominate
the corpus by pixel count. Selection is deterministic
under a fixed seed. Per-dataset counts are in the
published dataset inventory, browsable in [`dataset_directory/`](../../dataset_directory/).

During pretraining each tile is randomly kept at native resolution or downscaled, then randomly
cropped to the model's context size before augmentation, giving high view diversity from a fixed set
of base images.

Parameters are in [`tiling/RULES.md`](tiling/RULES.md); the code is authoritative.

## Data

The full catalog list and associated tiles from each source are in the Supplementary Table accompanying the manuscript. The underlying images are public and listed with
their repository URLs and DOIs in the published dataset inventory. Every script
takes its data roots as arguments.
