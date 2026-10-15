# `quantem_app/`

The QuantEM standalone application: organelle segmentation, interactive proofreading, guided
fine-tuning, and quantitative analysis of electron microscopy images — offline, run locally.

QuantEM ships pretrained models for **mitochondria, endoplasmic reticulum, nuclei, and lipid
droplets**, in two sizes — QuantEM (ViT-B, ~86 M parameters) and OmniEM (ViT-L, ~302 M).

## Working across an experiment

Home orders experiments by most recently opened, with pinned experiments first. **Continue
working** returns to the last page opened in that experiment. Experiment and image sorting
are independent. Within an experiment, filters, selection, page, expanded dataset cards and
scroll position are remembered for the browser session.

The workspace bar provides Home, the current experiment, View, Segment & review, Analysis,
Fine-tune, Models and Tasks. Previous/next image follows the experiment's filtered order and
keeps the current kind of page. Fine-tune opens with the current experiment or image selected;
annotations are required to start training, but not to open the dialog or apply a saved model.

### Measure and export many images

1. Open an experiment's **Images** tab. The workflow table separates image preparation,
   calibration, prediction, review completion and measurement status.
2. Select individual images, **Select visible**, or **Select all matching**. Dataset and
   readiness filters narrow the matching set. Selection persists across filters and includes
   each image only once, even when datasets overlap.
3. Images use their image names as sample labels and are included in summaries by default.
   Optionally assign a shared **Sample ID** to average images from one donor or sample
   together, and a **Group** to compare conditions. Saving replaces both fields;
   leaving Sample ID blank restores the image-name default.
4. Choose **Analyze selection**, select organelles and the measurement area, then **Check
   readiness**. A named analysis mask must exist on each image; an absent or ambiguous mask
   skips that image. Review completion is required by default and is independent of whether
   prediction finished. Uncalibrated images retain pixel measurements.
5. **Run ready images** queues durable tasks. Reopen saved batches, compare a metric across
   images, retry failed items, or cancel remaining work. Matching successful results can be
   reused only when their inputs, settings and identity still match.
6. **Export measurements ZIP** works for complete or partial batches. It includes
   `measurements.csv`, `samples.csv`, `groups.csv`, a manifest, and available per-run
   summaries. **Include object tables** adds each run's `objects.csv`.

Object measurements use current confirmed objects. When using an analysis mask, an object's
centroid must lie inside it; its full outline supplies shape measurements. Global
segmentations provide foreground-area measurements rather than object counts or shapes.
Global reports are saved in analysis history like object reports.

Each image contributes separately unless a shared sample ID is explicitly assigned. Sample
summaries average fields within those shared samples; group statistics then average sample
means with equal weight. Identical image names do not combine images automatically. Failed
items and outdated results remain in the image table and manifest but cannot contribute
current measurements. A single sample has no estimable sample SD or SEM. Different
organelles and measurement areas stay separate. The older image-level group endpoint counts
only the latest successful run per image and refuses incompatible measurement recipes.

Changing calibration, masks, labels, geometry, sample metadata or model-result identity marks
saved measurements outdated. Inputs are checked before and after execution, so a run whose
inputs change while queued or measuring fails with an explanation. A retry uses the original
settings; changed inputs require a new batch.

### Manage and continue fine-tuned models

**Models** lists saved fine-tunes alongside access to pretrained model management. Saved
models can be renamed, archived, restored to the list, applied to images, or used with
**Continue training**. Continuation loads the parent's learned weights and creates a separate
named model. Optionally include the parent's annotated images in the new training scope.
Every validation round starts from that same parent checkpoint.

**Training options** offers quick or extended schedules, custom step counts, learning rate,
and optional validation to choose the duration. These controls work for both pretrained
starting points and continued models. **Training history** in Models and completed runs
shows checkpoint selection and the final refit separately. See [fine-tuning controls](docs/fine-tuning-controls.md)
for annotation coverage, evaluation rules, and API options.

Continued models retain their parent and cumulative training-image exposure. Their “before”
score evaluates the parent. A new holdout split does not undo the parent's prior exposure,
so these scores are not automatically independent validation.

Archiving hides a model from new selections while preserving existing applied uses.
**Delete weights** removes managed weights and retains model identity in existing results.
Deletion is blocked while a model is in use by a task or has saved child versions. Existing
segmentations and measurement records are preserved.

### Recover corrections and choose the right export

**Correction history** saves and restores objects, reviewed areas and masks for a
segmentation. A revision is saved automatically before clearing manual corrections or
rerunning segmentation. Restore first saves the current state, then opens the restored
result for review. Unlock completed segmentations and wait for active tasks before restoring.
Probability previews are not restored. Revision archives live in the application data
directory and are included when that directory is backed up.

**Export image or mask** produces a canonical 8-bit display image or mask PNG; it does not
export original acquisition data or measurement tables. Label values cycle through 1–255
and are not persistent unique object IDs. Use Analysis exports for measurements and object IDs.

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
