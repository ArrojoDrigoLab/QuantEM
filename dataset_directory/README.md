# `dataset_directory/`

A browsable, filterable directory of the QuantEM electron microscopy corpus.

**Live site: [arrojodrigolab.github.io/QuantEM/dataset_directory/](https://arrojodrigolab.github.io/QuantEM/dataset_directory/)**

## Contents

| Item | What it is |
|---|---|
| [`site/`](site/) | The directory itself — HTML, CSS and ES modules. No build step, no framework. |
| [`exporter/`](exporter/) | The pipeline that turns a corpus extract into the site's data files |
| [`data/`](data/) | The published artifacts, committed so they are reviewable in a diff |
| [`serve.py`](serve.py) | A local server that assembles the site the way the deployed one is |

```bash
python serve.py          # then open http://localhost:45175/
```

## What you can do with it

Filter the whole corpus by **organ**, **kingdom**, **species**, **in-plane resolution** and
**imaging modality** — the same axes the manuscript uses to describe corpus composition — plus
dimensionality and source repository, and free-text search over dataset, experiment and image names.

Organ and kingdom are browsable two levels deep: expand an organ for its tissue contexts, or a
kingdom for its species. Neither is a strict hierarchy in this corpus — a species can occur under
more than one kingdom, and a tissue context under more than one organ — so a value can appear under
several parents, with a different count under each.

Selecting several values within one facet is a union; selecting across facets narrows. A dataset is
listed if **any** of its images match, and the number beside it is how many of its own images
matched — so a fifty-image dataset reads *50 assets* unfiltered, *17 assets of 50* under a filter,
and disappears when none of its images match. Every filter state is in the address bar, so any view
can be linked or cited.

## The corpus

The site reports what the corpus actually holds — every total, vocabulary size and per-facet count
shown on the page is computed from the published data at build time. Nothing is transcribed by hand
into this document, because the corpus keeps growing and a number written down here would be wrong
the moment it did.

The manuscript's supplementary tables are the citable snapshot of the sources; this directory is the
browsable, current view of the same material, and carries each dataset's provenance as a link to the
repository that serves it. Where the two differ, the directory is simply newer.

## Data

**No image data is stored or redistributed here.** Every dataset links to the repository that serves
it, and reuse terms are set by each depositor — this directory does not restate them, because the
source repository is authoritative and current in a way a cached copy would not be.

Thumbnails are 256 px derivatives shown for identification and browsing. Each is a representative
crop rather than a whole-image reduction: an electron micrograph shrunk whole is a uniform grey
rectangle, whereas one tile of it at 256 px still resolves membranes and organelles. They are built
from the tile corpus already cut and scored for pretraining, taking each asset's
highest-tissue-content tile, and falling back to the asset's own raster where no tile survives.

**Every image and volume in the corpus has a thumbnail.** A volume published only in a proprietary
container that no open tool can read is represented by EM of the same subject from the same source
folder rather than by its own file; it is the only such substitution in the set. Where the tile
corpus had no coverage, the image was fetched from the repository that publishes it — using HTTP
range requests to pull single members out of multi-gigabyte archives and single planes out of remote
volumes, rather than downloading whole datasets.

Every thumbnail is checked as it is actually viewed: measured for contrast, dynamic range and tone
after encoding, not just assumed good from its source. Nothing flat, nothing black, nothing blank.

Datasets whose deposition is still in progress have no link yet and are shown as *deposition
pending*. Adding an accession as one completes is a one-line edit to
[`exporter/dataset_links.json`](exporter/dataset_links.json) and a rebuild.

## Licence

The site and exporter code are BSD-3-Clause (see [`LICENSE`](LICENSE)). The metadata compilation in
[`data/`](data/) is CC BY 4.0. **No licence is asserted over the thumbnails or the underlying
images** — each remains subject to the terms set by its depositor. See
[`DATA_LICENSE.md`](DATA_LICENSE.md).

## Rebuilding

See [`exporter/README.md`](exporter/README.md). The short version:

```bash
export PYTHONPATH=exporter/src
python -m quantem_directory build  --extract <extract-dir> --urls <urls.csv> --out data
python -m quantem_directory thumbs --tile-index <tiles.csv> --tile-root <root> \
                                   --extract <extract-dir> --out thumbs
python -m quantem_directory build  --extract <extract-dir> --urls <urls.csv> --out data --thumbs thumbs
python -m quantem_directory verify --data data
```
