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

### Evaluate and apply models across an experiment

Start in the experiment's **Images** table. Organize images into datasets and optionally
record sample IDs and groups, then select a representative subset. Selection persists
across filters and pages; **Select all filtered images** includes every matching image.
Images use their names as sample labels and participate in summaries by default. Assign a
shared **Sample ID** to average images from one donor or sample together, and a **Group** to
compare conditions. Leaving Sample ID blank restores the image-name default.

**Evaluate / Fine-tune** opens on evaluation. **Inspect predictions** applies released or
saved models to the selection without requiring annotations. Open images to inspect the
results. If needed, create ROIs populated with model outputs, correct them, and mark them
done. Train on those examples, then evaluate and repeat. **Score against ROIs** evaluates
completed annotations without training or changing segmentations. Images used in the
model's training history are flagged, but can still be evaluated.

Use **Apply models** to apply released defaults or a saved fine-tune to the shared image
selection. The readiness check explains unavailable images and required downloads before
queueing work. Model defaults follow the same installed-model policy as the image viewer.
Evaluation settings can be carried into application, including the chosen threshold. Saved
applications retain per-image progress; cancel remaining tasks or retry failures without
repeating successful images.

A successful full-image model run is ready for measurement automatically. Completed manual
ROIs override predictions inside those areas. Manual-only coverage can be established by a
completed ROI covering the image or explicit acknowledgement that the entire image has been
manually segmented and includes no model output. Whole-image coverage does not lock editing.

In the labeling viewer, **Model results** retains model and threshold versions. Toggle
predictions or compare two overview overlays side by side. Each side has its own model
selector and opacity control, initially 30%. A check mark identifies the confirmed model.
**Confirm this model**, beside an unconfirmed selection, chooses the canonical model and
threshold while preserving completed manual areas. Its confirmation
checks deletion of other versions by default; clear those checkboxes to retain comparisons.
Only the final **Confirm** starts the change; old outputs are deleted after the replacement
has been successfully built and saved. Overview overlays display
thresholded probabilities; measurements use the extracted full-resolution result.

### Measure and export many images

1. Select images in the experiment table and choose **Measure selected**. All available
   organelles and output files are selected by default. Whole-image model results and
   completed manual segmentations can be measured immediately.
2. Single-image and batch analysis use the same organelle definitions and output choices.
   Available Sample Area and Cells measurements are included by default, along with whole-image
   summaries. Custom area definitions can supply a measurement region; their union is used as
   the denominator. Choose **Check readiness** before starting a batch. Explicitly selected
   areas must exist on each image; missing or ambiguous areas skip that image. Uncalibrated
   images retain pixel measurements. A partial manually annotated ROI never makes its organelle
   eligible for whole-image analysis, even when a smaller measurement region is selected.
3. **Run ready images** queues durable tasks. Reopen saved batches, compare measurements
   across images, retry failed items, or cancel remaining work. Successful results are
   reused only when their inputs, settings and identity still match.
4. **Export measurements ZIP** supports complete and partial batches. By default it contains
   image, sample and group tables, available per-run summaries and object tables, and a
   manifest. File checkboxes control the generated outputs.

Saved measurements and exports remain available after segmentation changes. A yellow flag
in history identifies superseded results, with an explanation on the analysis page.
Regenerating a batch creates a new result and can reuse its unchanged measurements.

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
QUANTEM_POSTGRES_BIN_DIR=/path/to/postgresql/bin quantem-app
```

`quantem-app` starts the local server and opens the application in your browser, or in a native
window with `pip install "quantem-app[desktop]"`. 

Everything QuantEM writes lives in one directory. `quantem-app --data-dir PATH …` chooses it, and so
does `quantem-app … --data-dir PATH` after the subcommand; `$QUANTEM_DATA_DIR` sets it for a whole
shell.

QuantEM starts its own PostgreSQL 17.11 server inside that directory. It binds only to loopback,
uses per-install random credentials, and stops with the application. PostgreSQL is bundled by the
desktop installers; it is not a machine-wide service and does not require Docker or a separately
administered database.

The pip and conda developer channels do not duplicate the server payload. A checkout or
environment launch must point `QUANTEM_POSTGRES_BIN_DIR` at the `bin/` directory of the pinned
runtime staged by `packaging/postgresql/stage_runtime.py`. Explicitly setting
`QUANTEM_DATABASE_ENGINE=sqlite` keeps the legacy backend available for tests and migration
comparisons; it is never selected as a fallback when PostgreSQL cannot start.


### From a checkout (development)

```bash
conda env create -f environment.yml
conda activate quantem-app
pip install -e ".[dev,desktop]"
(cd frontend && npm install && npm run build)
python packaging/postgresql/stage_runtime.py \
  --download-cache .scratch/postgresql-download \
  --output .scratch/postgresql-runtime
export QUANTEM_POSTGRES_BIN_DIR="$PWD/.scratch/postgresql-runtime/bin"
quantem-app
```

Development is the only channel that needs node: released artifacts carry the frontend already
built.

### Desktop installer (Windows)

Download the installer from the [latest release](https://github.com/ArrojoDrigoLab/QuantEM/releases/latest) and run it. Windows may
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

Everything QuantEM writes — the private database cluster, imported images, model packs, caches,
exports and logs —
lives in one directory:

| install | location |
|---|---|
| Windows installer | a `data` folder inside the directory you chose at install time |
| pip | inside the Python environment you installed into |
| macOS | `~/Library/Application Support/QuantEM` |

`$QUANTEM_DATA_DIR` overrides all of them, and `--data-dir PATH` overrides it for one command.

### Database migration and recovery

The first launch after upgrading an existing SQLite installation performs the PostgreSQL cutover
automatically, before any worker can claim a job. QuantEM creates and verifies a consistent SQLite
snapshot, advances a disposable copy through every skipped Django migration, initializes the
private PostgreSQL schema, copies and validates every model field, and only then opens the app.
Queued, retrying, or running jobs are retained as records but marked failed; jobs copied from the
old application instance are never resumed.

The startup window reports the current cutover stage, measured record or byte counts, elapsed time,
and a stage ETA once enough work has completed to estimate one. Opaque database checks retain an
indeterminate bar but continue to report liveness and the time since the last measurable advance.
There is no fixed migration deadline: QuantEM keeps monitoring a live sidecar until it finishes or
exits, and shows a non-fatal warning if liveness events stop for five minutes. Large libraries can
therefore take longer than two hours without the loading page abandoning a healthy transaction.

The live SQLite files are moved into that verified snapshot after PostgreSQL commits successfully.
The snapshot is the first **Database upgrade backup** and follows the configured count and age
limits (one backup for 30 days by default). A failed cutover leaves both the original SQLite files
and recovery snapshot intact, and startup stops instead of opening an empty database. A fresh
installation has no SQLite file, so it creates an empty PostgreSQL database without running the
bridge.

The equivalent command remains available for controlled recovery or development against an
inactive data directory; ordinary desktop upgrades do not require it. The destination PostgreSQL
database must contain no user data:

```bash
quantem-app --data-dir PATH database migrate-sqlite --source PATH/quantem.sqlite3
```

The bridge copies primary keys, relationships, WKB geometry, JSON, timestamps and file references,
resets PostgreSQL sequences, and compares canonical field content after the copy. The explicit
command retains its source SQLite file and verified snapshot.

After migration, PostgreSQL becomes the writable database. Opening the retained SQLite snapshot
later does not include any edits made in PostgreSQL. Moving back therefore requires an explicit
data transfer; QuantEM never silently falls back to SQLite.

Create a PostgreSQL backup with:

```bash
quantem-app --data-dir PATH database backup --output PATH/quantem.dump
```

The adjacent JSON manifest records the archive SHA-256 and database version. Restore is
non-destructive and always targets a new, inactive database:

```bash
quantem-app --data-dir PATH database restore PATH/quantem.dump \
  --target-database quantem_restore_20261028
```

Inspect the restored database before selecting it for a later controlled cutover. The active
database is not dropped or overwritten.

PostgreSQL minor releases keep the same on-disk format. A QuantEM update may replace the bundled
17.x binaries after taking a verified pre-migration backup, then restart the existing cluster.
Whenever an app launch detects pending database-schema migrations, QuantEM automatically creates
and verifies that backup before changing the schema. By default it retains only the newest
pre-migration backup and removes it after 30 days. Both limits are editable under **Settings →
Database upgrade backups**; cleanup runs at startup, after a new migration backup, and when the
policy is saved. This policy does not remove backups created explicitly with `database backup`.
A future PostgreSQL major upgrade is a separate release operation: back up with the old QuantEM
version, create a fresh cluster with the new major version, restore and validate it, and retain the
old cluster until acceptance. QuantEM refuses to start a cluster whose major version differs from
the bundled server; it does not attempt an in-place major upgrade or SQLite fallback.

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
