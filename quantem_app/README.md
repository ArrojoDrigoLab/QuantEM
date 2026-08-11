# `quantem_app/`

The QuantEM standalone application: organelle segmentation, interactive proofreading, guided
fine-tuning, and quantitative analysis of electron microscopy images — offline, run locally.

QuantEM ships pretrained models for **mitochondria, endoplasmic reticulum, nuclei, and lipid
droplets**, in two sizes — QuantEM (ViT-B, ~86 M parameters) and OmniEM (ViT-L, ~302 M).

## Installing

The pip wheel and the desktop installers deliver the same application. 
Model weights are not included in the install and are downloaded on-demand and cached from 
inference calls. [Models](#models).

### pip

Requires Python 3.12 or greater. If your machine has a CUDA capable-GPU, first install the 
CUDA build of `torch`/`torchvision` following [pytorch.org](https://pytorch.org/get-started/locally/),
and it will be used by the app. CUDA-enabled builds run much faster, especially when fine-tuning. 

```bash
pip install quantem-app
quantem-app
```

`quantem-app` starts the local server and opens the application in your browser, or in a native
window with `pip install "quantem-app[desktop]"`. 

Everything QuantEM writes lives in one directory. `quantem-app --data-dir PATH …` chooses it, and so
does `quantem-app … --data-dir PATH` after the subcommand; `$QUANTEM_DATA_DIR` sets it for a whole
shell.


### From a checkout (development)

```bash
conda env create -f environment.yml
conda activate quantem-app
pip install -e ".[dev,desktop]"
(cd frontend && npm install && npm run build)
quantem-app
```

Development is the only channel that needs node: released artifacts carry the frontend already
built.

### Desktop installer (Windows)

Download the installer from the [latest release](../../releases/latest) and run it. Windows may
warn that it does not recognise the publisher: click **More info**, then **Run anyway**. If you
would rather check the download first, run `certutil -hashfile <file> SHA256` and compare the
result with `SHA256SUMS.txt` on the same release page.

**Where to install it.** The installer asks you to choose a folder, and everything QuantEM saves
goes inside it: downloaded models, fast-viewing copies of your images, and your segmentation
results. That can reach several gigabytes, and more for large images. To move everything later, 
uninstall, reinstall to the new folder, and copy the `data` folder across.

### Desktop installer (macOS)

Download the `.dmg` from the [latest release](../../releases/latest) and drag QuantEM into your
Applications folder. The first time you open it macOS will say it cannot be verified — click
**Done**, then choose **Open Anyway** in **System Settings → Privacy & Security**; it opens
normally every time after that.

### Storage directory

Everything QuantEM writes — the database, imported images, model packs, caches, exports and logs —
lives in one directory:

| install | location |
|---|---|
| Windows installer | a `data` folder inside the directory you chose at install time |
| pip | inside the Python environment you installed into |
| macOS | `~/Library/Application Support/QuantEM` |

`$QUANTEM_DATA_DIR` overrides all of them, and `--data-dir PATH` overrides it for one command.

## Models

Model weights live in the
[`ArrojoeDrigoLab/quantem`](https://huggingface.co/ArrojoeDrigoLab/quantem) Hugging Face
repository and are downloaded at runtime, once, into your user data directory. Open the **Models** screen in the app and install the
packs you need; that is the whole step. The terminal form is identical in effect:

```bash
quantem-app models install quantem:mito     # <family>:<organelle>, e.g. omniem:er
quantem-app models list                     # what is installed, and whether it can run
```

| Family | Encoder | Organelles | Tile |
|---|---|---|---|
| QuantEM | ViT-B, 86 M | mito, ER, nucleus, lipid droplet | 512 px (patch 16) |
| OmniEM | ViT-L, 302 M | mito, ER, nucleus, lipid droplet | 518 px (patch 14) |
