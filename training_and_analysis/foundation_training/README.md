# `foundation_training/`

Self-supervised pretraining of the QuantEM EM encoder, and the comparison of candidate encoders.

---

## Usage

```bash
conda env create -f environment.yml
conda activate quantem-foundation-training

# upstream DINOv3 at the pinned commit
third_party/fetch_dinov3.sh <dinov3 dir> && pip install <dinov3 dir>

# tiles -> shards
python -m em_ssl.tools.build_shards --manifest <manifest.jsonl> --tile-root <tiles> \
    --output-root <bundle>/shards/em_tiles_v1 --shard-prefix em_tiles_v1 \
    --samples-per-shard 1000 --seed 1337 --balance-shards-by-source --min-side 512

# shards -> pretraining
torchrun --nproc_per_node=2 -m em_ssl.training.train_dinov3_em \
    --config configs/pretraining/quantem_vitb_512.yaml \
    --data-root <bundle> --output-dir <run dir>
```

Upstream DINOv3 (Siméoni et al., 2025, arXiv:2508.10104) is a prerequisite, installed at the pinned
FINO-branch commit in [`third_party/dinov3.pin`](third_party/dinov3.pin) and obtained from Meta.

`--dry-run` builds the model on the meta device and validates the single-channel stem without
training; for a FINO arm it also instantiates the guide heads on CPU to check the translated
`guide:` block.

`--data-root` is the bundle directory; the config's `data.shard_prefix` picks which shard set inside
it to read, so the prefix given to the builder and the one in the config must agree. The released
encoder reads `em_tiles_v1`. 

The tile manifest comes from [`../dataset_assembly/tiling/`](../dataset_assembly/tiling/). Run its
`join_asset_metadata.py` first if metadata conditioning
is being reproduced: pixel size and modality are per-asset facts that the tiles do not carry.

Pretraining the released encoder to step 674999 took roughly 15 GPU-days on H100-class hardware, plus
about 1 TB for the shard bundle.

## Model weights

Model weights must be referenced explicitly. 

- **The QuantEM encoder** — `--run-dir <encoder run dir>`, the directory holding `checkpoint_index.json`,
  which pretraining writes and everything downstream loads through.
- **Published baselines** — put each encoder's weight file under `<weights root>/<encoder name>/`,
  then build a checkpoint index for each so they load through the same path:

  ```bash
  python -m foundation_baselines.register_external_encoders --weights-root <weights root>
  ```

  [`configs/encoder_comparison/`](configs/encoder_comparison/) gives the expected filename,
  architecture and normalisation for each.

## Contents

| Item | What it is |
|---|---|
| `em_ssl/training/train_dinov3_em.py` | Pretraining entrypoint |
| `em_ssl/integration/dinov3_patch.py` | Runtime patches adapting upstream DINOv3 to single-channel EM |
| `em_ssl/integration/config_translation.py` | Translates the experiment schema into DINOv3 configuration keys |
| `em_ssl/config/{schema,resolve}.py` | The experiment contract, and the configuration emitter |
| `em_ssl/transforms/` | Multi-crop EM augmentation and grayscale-safe primitives |
| `em_ssl/data/` | Shard reading and writing, manifest streaming, corpus filtering |
| `em_ssl/fino/` | Metadata conditioning: the factor definitions, the masked guide losses, and the graft onto the upstream guided meta-architecture |
| `em_ssl/tools/` | Shard building, corpus intensity statistics, metadata coverage and diagnostics |
| `em_ssl/utils/checkpoint_index.py` | The encoder handoff that segmentation training loads through |
| `encoder_evaluation/` | The decoder probe every encoder and pretraining recipe was compared with, its configurations, and the reported metrics |
| `foundation_baselines/` | Loaders for the published encoders compared against |
| `configs/` | Every configuration reported in the manuscript |
| `third_party/dinov3.pin` | The pinned upstream commit |

## The encoder

Pretraining used a constant learning rate, so the
`max_steps` in the config is a soft upper bound rather than the training length. Segmentation performance
improved through roughly 650,000 steps before plateauing. **The released checkpoint is the teacher
export at step 674999, from the stable phase.**

## Configurations

Every configuration here is an **experiment definition**: only what this project sets, on top of
upstream DINOv3's own defaults. Anything a file does not mention is upstream's default at the pinned
commit and is not restated, so these files describe the recipe without redistributing Meta's
configuration. Both the trainer and `em_ssl.config.resolve` perform the same merge, so an experiment
file plus the pin fully determines the run.

| Group | What it holds |
|---|---|
| [`configs/pretraining/`](configs/pretraining/) | The released QuantEM encoder |
| [`configs/encoder_comparison/`](configs/encoder_comparison/) | The published encoders compared against: EMCellFound, DINOv2 ViT-L, DINOv3 ViT-L, OmniEM |
| [`configs/ablations/`](configs/ablations/) | The pretraining-recipe arms |
| [`configs/metadata_conditioning/`](configs/metadata_conditioning/) | Continued pretraining under FINO |

To see one arm's complete effective configuration — every upstream default included, exactly as the
run saw it — merge it against upstream locally. This needs DINOv3 installed, and writes a file that
is mostly Meta's defaults, so it is a local step rather than something published here:

```bash
python -m em_ssl.config.resolve --config configs/pretraining/quantem_vitb_512.yaml \
    --out <somewhere>/quantem_vitb_512.resolved.yaml --world-size 2
```

Each arm's `experiment` key below is its row in the corresponding results table under
[`encoder_evaluation/results/`](encoder_evaluation/results/).

| Configuration | Arm | `experiment` |
|---|---|---|
| `ablations/base_512_seed1.yaml` | base at 512 px, first run | `base_512_seed1` |
| `ablations/base_512_seed2.yaml` | base at 512 px, second run | `base_512_seed2` |
| `ablations/gram_anchoring_512.yaml` | Gram anchoring from step 0 | `gram_anchoring_512` |
| `ablations/large_context_full_1024.yaml` | large context, full: a single stage at 1024 px | `large_context_full_1024` |
| `ablations/large_context_ramp_light_512_768.yaml` | large context, light ramp: 512 px then 768 px | `large_context_ramp_light_512_768` |
| `ablations/large_context_ramp_heavy_512_768_1024.yaml` | large context, heavy ramp: 512, then 768, then 1024 px | `large_context_ramp_heavy_512_768_1024` |
| `metadata_conditioning/baseline_continued.yaml` | control, no guide | `baseline_continued` |
| `metadata_conditioning/fino_scale_preserve.yaml` | scale, M+ | `fino_scale_preserve` |
| `metadata_conditioning/fino_scale_suppress.yaml` | scale, M− | `fino_scale_suppress` |
| `metadata_conditioning/fino_modality_preserve.yaml` | modality, M+ | `fino_modality_preserve` |
| `metadata_conditioning/fino_modality_suppress.yaml` | modality, M− | `fino_modality_suppress` |

The metadata-conditioning arms continue from the base arm's teacher export, so set
`dinov3.student.resume_from_teacher_chkpt` before launching. The method is described in
[`em_ssl/fino/`](em_ssl/fino/).

Multi-stage arms train one crop stage per launch, warm-started from the previous stage's teacher
export. One configuration holds the whole schedule, so the stage is selected at launch and the
stages are run in order:

```bash
torchrun --nproc_per_node=2 -m em_ssl.training.train_dinov3_em \
    --config configs/ablations/large_context_ramp_light_512_768.yaml --stage-index 1 \
    --data-root <bundle> --output-dir <run dir> \
    --warm-start <run dir>/eval/<step>/teacher_checkpoint.pth
```

`--warm-start` transfers the backbone only; the DINO and iBOT heads reinitialise at each stage
boundary, as they do upstream.

## Evaluation

[`encoder_evaluation/`](encoder_evaluation/) holds the decoder probe every encoder and recipe was
compared with, the probe configurations, and the reported metrics.
