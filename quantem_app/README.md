# `quantem_app/`

The QuantEM standalone application: organelle segmentation, interactive proofreading, guided
fine-tuning, and quantitative analysis of electron microscopy images — offline, on one machine.

> 🚧 **Under construction.** Some areas are complete, some are staged but not yet adapted, and
> some are not yet written; the module map under [Layout](#layout) marks what is still in
> progress.

QuantEM ships pretrained models for **mitochondria, endoplasmic reticulum, nuclei, and lipid
droplets**, in two sizes — QuantEM (ViT-B, ~86 M parameters) and OmniEM (ViT-L, ~302 M) — so it can
run on a laptop or make use of a GPU when one is present.

---

## Installing

Three channels, one product: the pip wheel, the conda package and the desktop installer all
deliver the same application — pick whichever fits the machine. **None of them contains model
weights**; those are one further download, made by the app itself — see [Models](#models).

### pip

```bash
pip install quantem
quantem
```

`quantem` starts the local server and opens the application in your browser — or in a native
window with `pip install "quantem[desktop]"` instead. Nothing leaves your machine. Requires
Python 3.12 or 3.13; the released wheel ships the built frontend inside the package, so there is
nothing to compile and no node toolchain involved.

PyTorch installs from PyPI as an ordinary dependency (the CPU build, on most platforms). To use
an NVIDIA GPU, first install the CUDA build of `torch`/`torchvision` following
[pytorch.org](https://pytorch.org/get-started/locally/), then `pip install quantem` — pip keeps
the torch you chose.

Everything QuantEM writes lives in one directory. `quantem --data-dir PATH …` chooses it, and so
does `quantem … --data-dir PATH` after the subcommand; `$QUANTEM_DATA_DIR` sets it for a whole
shell.

### conda

The conda package is `noarch` and built from [`packaging/conda/meta.yaml`](packaging/conda/meta.yaml);
it is published per release. Until it reaches a public channel, pip inside a conda environment
gives the identical product:

```bash
conda create -n quantem python=3.13
conda activate quantem
pip install quantem
quantem
```

### From a checkout (development)

```bash
conda env create -f environment.yml
conda activate quantem-app
pip install -e ".[dev,desktop]"
(cd frontend && npm install && npm run build)
quantem
```

Development is the only channel that needs node: released artifacts carry the frontend already
built.

### Desktop installer

Signed installers for Windows and macOS are the second channel. Until code-signing certificates are
in place, installers are **unsigned**, which means one extra click on first launch:

- **Windows** — SmartScreen shows *"Windows protected your PC"* → **More info** → **Run anyway**.
- **macOS** — Gatekeeper blocks the first open → **System Settings → Privacy & Security** →
  **Open Anyway**.

Updates after that first launch apply silently. Institutionally managed Macs may block unsigned
applications outright regardless of user action; use the Python package on those machines.

---

## Models

Model weights ship in **no install channel** — wheel, conda package and desktop installer alike
are ~2 GB of application, not ~7 GB of weights. The weights live in the
[`ArrojoeDrigoLab/quantem`](https://huggingface.co/ArrojoeDrigoLab/quantem) Hugging Face
repository and are downloaded at runtime, once, into your user data directory — verified by
SHA-256 before a pack becomes installed. Open the **Models** screen in the app and install the
packs you need; that is the whole step. The terminal form is identical in effect:

```bash
quantem models install quantem:mito     # <family>:<organelle>, e.g. omniem:er
quantem models list                     # what is installed, and whether it can run
```

| Family | Encoder | Organelles | Tile |
|---|---|---|---|
| QuantEM | ViT-B, 86 M | mito, ER, nucleus, lipid droplet | 512 px (patch 16) |
| OmniEM | ViT-L, 302 M | mito, ER, nucleus, lipid droplet | 518 px (patch 14) |

### Installing the models offline

The in-app Hugging Face download is the normal path. Working on an air-gapped machine, or
archiving the exact weights behind a paper? Download the release bundle for your QuantEM version
on any machine, unzip it anywhere, and point QuantEM at that directory:

```bash
quantem models install ~/Downloads/quantem-models-0.1.0
quantem models list                 # what is installed, and whether it can run
```

Both accept `--data-dir PATH`, before or after the subcommand, to use a data directory other than
the default. Without the console script on your `PATH`:

```bash
QUANTEM_DATA_DIR=/where/you/want/it \
    python -m quantem.registry.install bundle ~/Downloads/quantem-models-0.1.0 --all
```

Nothing about this needs network access, a research checkout, or a path that exists only on the
maintainer's machine. Every file in the bundle is listed in its `MANIFEST.json` with a SHA-256;
each one is re-hashed and checked against that value **before** the pack becomes installed, and
you can check the download on its own first:

```bash
python -m quantem.registry.release verify ~/Downloads/quantem-models-0.1.0
```

The default threshold is **0.5** for every organelle and both families — the same setting behind
every benchmark in the paper, so a number you reproduce here is comparable to a published one.
Guided fine-tuning can calibrate a threshold against your own annotations.

### Self-contained encoders

The released checkpoints are bare `state_dict`s, and the QuantEM family's encoder architecture
comes from Meta's `dinov3`, which QuantEM does not redistribute. So every pack in a release
bundle carries an **exported TorchScript encoder** (`encoder_ts.pt`), traced *after* that pack's
own adapters and fine-tuned blocks are in place. Once it is installed the pack runs with no
`dinov3` and no `timm` present at all. The OmniEM family never needs `dinov3` in any case: its
architecture comes from `timm`.

Each export is verified against the eager model before it is written — all eight currently
reproduce it to `max|diff| = 0.00e+00`, and an export that did not would abort the build rather
than ship.

### Building a release bundle (maintainers)

One command, run once per release on a machine that has the architecture code:

```bash
export QUANTEM_DINOV3_PATH=/path/to/facebookresearch/dinov3
python -m quantem.registry.release build \
    --out dist/quantem-models-0.1.0 --release 0.1.0 \
    --heads-root   <dir of <organelle>_<family>/head.pt> \
    --weights-root <dir of <run_id>/checkpoint_index.json> \
    --search-dir   <dir holding the encoder checkpoint files>
python -m quantem.registry.release verify dist/quantem-models-0.1.0
```

That directory is the artifact that goes to Hugging Face and Zenodo. It contains, per pack, a
`pack.json`, `head.pt`, `resolved_config.yaml`, `checkpoint_index.json` and the exported
`encoder_ts.pt`; at the top, a `MANIFEST.json` with a SHA-256 for every file and a
`MANIFEST.json.sha256` beside it. It records no path from the build machine.

The three roots have no defaults — the training outputs live somewhere different on every build
box, and a default that only resolves on one computer is exactly how the one documented way to
obtain these models came to be a command nobody else could run. `$QUANTEM_HEADS_ROOT`,
`$QUANTEM_WEIGHTS_ROOT` and `$QUANTEM_ENCODER_SEARCH_DIRS` set them once per shell.

---

## Layout

```
quantem_app/
├─ src/quantem/
│  ├─ core/           Django project: settings, urls, config, middleware
│  ├─ assets/         images, renditions, OME-NGFF pyramids, volume readers
│  ├─ jobs/           DB-backed job queue and worker pool (no broker)
│  ├─ segmentation/   segmentations, objects, ROIs, overlay pyramids, editing API
│  ├─ seg_core/       segmenter base class, extraction, rasterisation
│  ├─ inference/      in-process model inference          ← being written
│  ├─ finetune/       guided fine-tuning + threshold sweep ← being written
│  ├─ analysis/       morphometrics, compartments, spatial stats, export ← being written
│  └─ registry/       model download, checksum, cache      ← being written
├─ frontend/          React + TypeScript, Viv/deck.gl viewer
├─ pyproject.toml     dependency and tooling truth
└─ environment.yml    conda development environment
```

## Requirements

- Python 3.13 (3.12 also supported)
- ~2 GB disk for the application, plus ~7 GB for the model release bundle and the same again for
  the installed copy (you can delete the unzipped bundle afterwards)
- A GPU is **optional**. Inference and guided fine-tuning both run on CPU; CUDA and Apple-Silicon
  MPS are used automatically when available.
- No PostgreSQL, no PostGIS, no SpatiaLite, no Redis, no GDAL. The application stores everything in
  a single SQLite file in your user data directory — back it up by copying that one file.
- **Image formats in v1: TIFF (including OME-TIFF) and PNG**, as single files or a directory of
  slices. Pixel size is read from OME-XML or TIFF tags where present and can always be set by hand.

## How this maps to the paper

| Paper | Here |
|---|---|
| Released models, 4 organelles × 2 encoder sizes | `inference/`, `registry/` |
| Sliding-tile inference, 25 % overlap, Hann blending | `inference/` |
| Guided fine-tuning, Dice-maximising threshold sweep | `finetune/` |
| Interactive proofreading | `frontend/src/features/segmentation`, `segmentation/overlay_ngff` |
| Compartment assignment, distances, Monte Carlo nulls | `analysis/` |

## Licence

BSD-3-Clause — see [`LICENSE`](LICENSE). Third-party components and their licences are in
[`NOTICE`](NOTICE).
