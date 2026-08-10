# Draft README section for the Hugging Face model repository

**Status: report-only.** The `ArrojoeDrigoLab/quantem` repository is not writable from
this machine, so this file is the drafted addition, left here for the repository owner
to paste into the model card (the repo's `README.md`). Nothing in the application reads
this file.

**Why it is needed.** The model card currently describes the artifacts but not the one
supported way to consume them: the QuantEM application. A reader who lands on the HF
page first has no line telling them the weights are useless loose and that
`pip install quantem` + one command (or one click) downloads, verifies and installs
them. The section below closes that loop in both directions — the app's Models screen
and CLI already point at this repository by URL.

Everything below the rule is the proposed text, written for the HF audience. The pack
ids, commands and file names are the real ones the shipped application uses
(`quantem.registry.hf` pins this repository; `quantem models install` is the CLI).

---

## Using these models

These weights are the pretrained organelle segmentation packs for
**[QuantEM](https://github.com/ArrojoeDrigoLab/QuantEM)**, a desktop application for
segmenting, proofreading and quantifying electron microscopy images — offline, on one
machine. The files in this repository are raw artifacts (safetensors plus a JSON model
card per pack); the supported way to run them is through the application, which
downloads, **verifies every file against its published SHA-256** at a pinned revision,
and installs them into its own model cache.

### Install the application

```bash
pip install quantem
quantem
```

`quantem` starts the local server and opens the application (Python 3.12/3.13; the CPU
build of PyTorch installs as an ordinary dependency — install a CUDA build of torch
first to use an NVIDIA GPU). No model weights ship with the application; they are
downloaded on demand from this repository.

### Get the models

Either route fetches from this repository and verifies before installing:

- **In the app** — open the **Models** screen and click **Download** on a pack. The
  screen shows what each pack costs to download and whether this machine can run it.
- **From the terminal** —

  ```bash
  quantem models install --all          # all eight packs
  quantem models install quantem:mito   # or name packs individually
  quantem models list                   # what is installed, what can run
  ```

The eight packs are `quantem:mito`, `quantem:er`, `quantem:nucleus`, `quantem:ld`
(ViT-B, laptop-friendly) and `omniem:mito`, `omniem:er`, `omniem:nucleus`, `omniem:ld`
(ViT-L). Three QuantEM packs share the `quantem-vitb-trunk.safetensors` encoder and the
OmniEM packs share `omniem-vitl.safetensors`, so installing a family costs one copy of
its trunk, not four.

### Offline machines

Download a QuantEM model release bundle on a connected machine
(see the QuantEM releases page), copy it over, unzip it, and:

```bash
quantem models install <the directory you unzipped the release into>
```

Every file is re-hashed against the bundle's `MANIFEST.json` before the pack counts as
installed.

### Loading the files by hand

Each `<family>-<organelle>.json` card records the architecture, the inference contract
(tile size, working resolution, threshold) and the head artifact's size and SHA-256;
the `requires` list names the trunk file a head needs. Nothing stops you loading the
safetensors yourself, but the contract in the card — resampling to the pack's working
resolution, tiling, thresholding — is what the application implements, and results
without it will not match the published ones.

### Licence

The repository code of QuantEM is MIT; **these weights are not covered by it** — see
the NOTICE file distributed with the application. The OmniEM packs' ViT-L base encoder
is upstream EM-DINO (bioRxiv 10.1101/2025.04.13.648639) and carries its own licence.
