# `benchmarking/`

Evaluation of the final QuantEM and OmniEM models against publicly available segmentation models
across held-out datasets, and the label-efficiency comparison for fine-tuning.

---

## Usage

```bash
conda env create -f environment.yml        # quantem-benchmarking: transcode + aggregation
```

**External comparators** — each model runs in its own environment from `comparators/envs/`, with
weights downloaded from the sources in the table below (not redistributed here). One adapter per
model family; every adapter takes the crops root, the weights location, and the output root as
arguments and writes one per-crop CSV per model and organelle:

```bash
conda env create -f comparators/envs/empanada.yml
python comparators/run_empanada.py mito \
    --seg-root <held-out source crops> --weights <weights dir> --results-root <results root>
```

Weight sources per model are in [`comparators/WEIGHTS.md`](comparators/WEIGHTS.md).

**Internal models** — the benchmark tiles are transcoded into the layout the training harness
reads, then each arm is trained and evaluated through the harness. Run from
`training_and_analysis/`, with [`../segmentation_training/`](../segmentation_training/) importable;
[`internal_models/RUN.md`](internal_models/RUN.md) lists all eight launches:

```bash
python benchmarking/internal_models/tiles_to_harness.py \
    --src-root <benchmark tiles root> --org nucleus --dst-root <harness data root>

python -m segmentation_training.harness.run_seg \
    --config benchmarking/internal_models/configs/nucleus_omniem.yaml \
    --data-root <harness data root> --run-dir <encoder run dir> --output-dir <arm run dir>
```

**Aggregation** — per-crop CSVs to summaries and Dice matrices, then the combined
internal-plus-external leaderboards, in the `quantem-benchmarking` environment:

```bash
python aggregate.py        --results-root <results root>
python make_tables.py      --data <results root>
python compile_combined.py --ext-dir <results root>/tables \
    --results-root <internal arm run dirs> --out-dir <results root>/combined
```

Every root is an explicit command-line argument.

## Contents

| Item | What it is |
|---|---|
| `harness/` | Shared crop loading, region-masked scoring, and metrics — every external model scores through it |
| `comparators/` | One adapter per external model, run through that model's published implementation |
| `comparators/WEIGHTS.md` | Where each model's weights are obtained |
| `comparators/envs/` | The per-comparator environment files |
| `internal_models/` | The nucleus/LD arm configurations, the tile transcode, the CEM-group build, and the launch record |
| `aggregate.py` | Per-crop CSVs to per-dataset and per-organelle summaries and Dice matrices |
| `make_tables.py` | The per-organelle model-by-dataset tables |
| `compile_combined.py` | Merges internal harness results with the external matrices into one leaderboard per organelle |
| `label_efficiency/` | The fine-tuning-versus-annotation-count experiment; scores under `label_efficiency/results/` |
| `results/` | Per-crop, per-dataset, and per-organelle scores, and the combined tables |
| `environment.yml` | The aggregation environment (`quantem-benchmarking`) |

## Evaluation protocol

Every model — internal and external — is scored identically, on the held-out test sets. Held-out membership is defined by the `benchmark_<organelle>.csv` splits built in
[`../segmentation_dataset/`](../segmentation_dataset/), which also lists every source with its
accession.

Internal inference is sliding-tile with
overlap and Hann blending; external models tile or resample internally per their published
pipelines. 

**Macro Dice** is the unweighted mean of per-dataset mean Dice, so every dataset counts equally
regardless of its crop count.

## External models

| Model | Organelles | Weights source | Environment file |
|---|---|---|---|
| MitoNet (empanada) | mitochondria | Zenodo 6861565 (`MitoNet_v1`) | `comparators/envs/empanada.yml` |
| NucleoNet (empanada) | nucleus | Zenodo 18142651 | `comparators/envs/empanada.yml` |
| DropNet (empanada) | lipid droplet | Zenodo 15298854 | `comparators/envs/empanada.yml` |
| micro-SAM | mitochondria, ER, nucleus, lipid droplet | `vit_l_em_organelles` (BioImage.IO) | `comparators/envs/microsam.yml` |
| Incasem | mitochondria, ER | `s3://asem-project` (`1847_mito_CF`, `1841_er_CF`) | `comparators/envs/incasem.yml` |
| DeepContact | mitochondria, ER | figshare 19845940 | `comparators/envs/deepcontact_mito.yml`, `comparators/envs/deepcontact_er.yml` |
| OrgSegNet | mitochondria, nucleus | published OrgSegNet repository (`OrgSegNet_iter_Version1.pth`) | `comparators/envs/orgsegnet.yml` |

External models run through their published implementations and default configurations. Models
whose published pipelines resolution-normalize keep that behavior: DeepContact maps input to
~10 nm/px, Incasem requires 5 nm/px (each crop is replicated along z to adapt the 3D FIB-SEM
models to 2D input, resampled, and contrast-equalized per its published requirements), and
OrgSegNet applies its fixed 768x512 configuration resize. All other models receive the crops at
native pixel resolution. The empanada models are additionally run at their default resampled
resolution, and both settings are reported. micro-SAM runs automatic instance segmentation with
its trained decoder and serves as the general EM comparator across all four organelles.

## Internal models

QuantEM (ViT-B) and OmniEM (ViT-L) are the internal encoders. Each benchmark arm trains a
segmentation head on the encoder through the
[`../segmentation_training/`](../segmentation_training/) harness on the benchmark train/val tiles,
then is evaluated once on the held-out test split:

| Arm | Configuration |
|---|---|
| Mitochondria (QuantEM, OmniEM) | [`../segmentation_training/configs/released_models/`](../segmentation_training/configs/released_models/) `mitochondria_{quantem,omniem}.yaml` |
| ER (QuantEM, OmniEM) | released_models `er_{quantem,omniem}.yaml` |
| Nucleus (QuantEM, OmniEM) | `internal_models/configs/nucleus_{quantem,omniem}.yaml` |
| Lipid droplet (QuantEM, OmniEM) | `internal_models/configs/ld_{quantem,omniem}.yaml` |

## Label efficiency

The fine-tuning comparison measures Dice as a function of the number of manually annotated crops
supplied, on in-house immuno-EM ground truth. The score tables are under `label_efficiency/results/`.

