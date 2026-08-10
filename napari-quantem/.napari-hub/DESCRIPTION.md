# QuantEM — organelle segmentation for electron microscopy

Pretrained vision-transformer models for segmenting **mitochondria, endoplasmic reticulum, nuclei
and lipid droplets** in EM images, with the tools to correct the output and adapt the models to your
own data.

Eight models ship: each organelle in two sizes — QuantEM ViT-B (86M parameters) and OmniEM ViT-L
(302M). They were trained on the largest curated collection of intracellular EM datasets assembled
to date, spanning many species, tissue contexts and EM modalities — browsable in the
[QuantEM dataset directory](https://arrojodrigolab.github.io/QuantEM/dataset_directory/).

## Try it in three clicks

**File → Open Sample → QuantEM** loads a small example image that ships with the plugin. Open
**QuantEM: Segment**, enter the pixel size, and run.

## What you get

- **Segment** — 2-D images or stacks, using the exact sliding-window procedure behind the published
  benchmarks: 512 px windows, 25 % overlap, Hann blending. Every result carries a provenance record
  recording the model, threshold, pixel sizes and tiling used to produce it.
- **Fine-tune** — correct the segmentation inside a region you draw, and adapt the model to it.
  Only the decoder head trains, which the manuscript found works about as well as heavier methods
  and takes seconds on a GPU rather than minutes.
- **Proofread** — split, merge, delete, filter by size, drop border-touching objects, and
  morphological clean-up, on top of napari's own painting tools.
- **Measure** — per-object morphometrics and per-image summaries into the layer's feature table,
  with CSV export. Physical units when you supply a pixel size, pixels otherwise — the column names
  always say which.
- **Batch** — run a folder and write labels plus one combined measurements CSV.

## Two things worth knowing

**Pixel size is never guessed.** The plugin does not read it from file metadata and does not infer
it. If you supply one, images are resampled to the resolution each model was trained at (8 nm/px for
mitochondria and lipid droplets, 25 nm/px for nuclei; ER runs at native resolution). If you leave it
blank, nothing is rescaled and the widget tells you so.

**Fine-tuning asks you to draw the region you actually annotated,** and this is not a formality.
Inside that region your labels are taken as complete; outside it they are treated as *unknown*,
never as background. Without it, correcting a few objects in one corner of a large image would
teach the model that every organelle you did not label is background.

## Models

Model weights are downloaded on first use and cached — they are far too large for a Python package.
Nothing is fetched until you agree: the picker shows the size up front, and a dialog lists every
file, its size, its source and its licence first. Each file is verified against a published SHA-256
after download and on every subsequent use.

Encoders are shared where the models genuinely share them, so a second organelle usually costs far
less than the first. Offline and air-gapped installs are supported by pointing
`QUANTEM_MODEL_DIR` at a directory of pre-downloaded files.

The plugin is BSD-3-Clause. The weights are licensed separately, on Hugging Face.

## Citing

Acree *et al.*, *QuantEM: An optimized platform of vision transformer-based models for segmentation
and analysis of electron microscopy data.*
