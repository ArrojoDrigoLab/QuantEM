# napari-quantem

Organelle segmentation for electron microscopy in [napari](https://napari.org), using the QuantEM
and OmniEM vision-transformer models.

## Installing

If you do not already have napari:

```bash
pip install "napari-quantem[all]"
```

If napari is already installed then just get the plugin:

```bash
pip install napari-quantem
```

PyTorch is a dependency. The CPU-version is smaller to download, but it will run faster on
versions of Pytorch with CUDA / GPU-enabled, especially for fine-tuning. 

## What it contains

Eight pretrained models — **mitochondria, endoplasmic reticulum, nucleus and lipid droplets**, each
in two sizes (QuantEM ViT-B, 86M parameters; OmniEM ViT-L, 302M) — plus the tools to correct their
output and adapt them to your own data.

| Widget | |
|---|---|
| **Segment** | Run a model over a 2-D image or a stack. Sliding 512 px windows at 25 % overlap with Hann blending — the same procedure used for the published benchmarks. |
| **Fine-tune** | Correct the segmentation inside a region you draw, then adapt the model to it. Only the decoder head trains, which the manuscript found works about as well as heavier methods and takes seconds rather than minutes. |
| **Proofread** | Split, merge, delete, filter by size, remove border-touching objects, and morphological clean-up — uses napari's own painting tools. |
| **Measure** | Per-object morphometrics and per-image summaries into the layer's feature table, plus CSV export. |
| **Batch** | Run a folder of images and write labels and one combined measurements CSV. |
| **Models** | What is cached, what a download would cost, and what hardware is available. |

## Using it

**File → Open Sample → QuantEM → Mouse pancreatic islet (SEM, 5 nm/px)** loads a small example image that ships with the plugin. Open **QuantEM: Segment**, optionally set the pixel size to **5**, and run. (The model itself downloads on first use; see *Model files* below.)

## Model files

Weights are downloaded once, on first use, and cached.

Encoder parameters are shared where the models share them, so a second organelle model usually is much smaller to download than the first. 

## Pixel size

The plugin does not guess pixel size. If you supply one,
images are resampled to the resolution the model was trained at (8 nm/px for mitochondria and lipid
droplets, 25 nm/px for nuclei; ER runs at native resolution). If you leave it blank, nothing is
rescaled.

Supplying a pixel size usually has moderate to insignificant effects when native resolution is ~0.5x-2x the resampling defaults (8nm for mitochondria and lipid droplets, 25nm for nuclei). Large upsampling to this resolution can have variable results, and outputs should be examined; downsampling into this range is likely to improve results for very high resolution images. 

## Fine-tuning: draw the region you actually annotated

The fine-tuning widget asks for a **reviewed region** — a rectangle or polygon you draw. Inside it, labels are taken as complete. Outside it, they are treated as *unknown*, not as background.

The recommended loop is to segment, look for areas where the model is wrong, draw a region around one of
those places, fix the labels inside it, and adapt. With three or more regions you can run a
leave-one-region-out check for a genuinely held-out accuracy figure. Larger regions will give better results. 

## Citing

Acree C, *et al.* *QuantEM: An optimized platform of vision transformer-based models for segmentation and analysis of electron microscopy data.* bioRxiv 2026.
[https://www.biorxiv.org/content/10.64898/2026.08.06.743293v1](https://www.biorxiv.org/content/10.64898/2026.08.06.743293v1)

## Licence

BSD-3-Clause — see [`LICENSE`](LICENSE).
