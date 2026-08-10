# `segmentation_training/`

Training the organelle segmentation heads on the foundation encoders, and the experiments behind the
released configurations.

---

## Usage

```bash
conda env create -f environment.yml
conda activate quantem-segmentation-training

python -m segmentation_training.harness.run_seg \
    --config segmentation_training/configs/released_models/mitochondria_quantem.yaml \
    --data-root <ground-truth tiles> --run-dir <encoder run dir> --output-dir <run dir>
```

`dataprep/` builds the `(EM, mask)` tiles the harness reads, from the annotated sources and their
per-organelle split CSVs:

```bash
python -m segmentation_training.dataprep.build_dataset \
    --corpus-root <annotated sources> --out <ground-truth tiles> \
    --organelles mito er --splits train val test --target-nm <nm/px>
```

Arms at different input scales read separately built roots, one per `--target-nm`.

Run from `training_and_analysis/`, the directory holding this package. Data root, encoder run
directory and output directory are always arguments; a configuration names which encoder an arm used
but never a path. Ground-truth tiles come from
[`../segmentation_dataset/`](../segmentation_dataset/); encoders from
[`../foundation_training/`](../foundation_training/), which must be importable — the encoder is loaded
through its `checkpoint_index`.

## Contents

| Item | What it is |
|---|---|
| `models/` | Necks, decoders, losses, image-style conditioning, and the shared feature contract |
| `hooks/encoder_adaptation.py` | Frozen, LoRA (static and conditional), last-N-blocks and full fine-tuning |
| `harness/` | Training loop, tiled evaluation, dataset, metrics, test-time support |
| `experiments/` | Input scale and multi-organelle, which need more than the harness |
| `configs/` | One directory per reported experiment, with its arms and its reported metrics |
| `EXPERIMENTS.md` | The experiments and where their configurations and metrics are |
| `tests/` | Plumbing tests over a synthetic corpus and a mock encoder |

## Registries

**Decoders** — `upernet`, `dpt`, `nnunet_convnext_unet`, `pspnet`, `deeplabv3plus`, `unet`,
`panoptic_deeplab`, `affinity_mws`, `mask2former_query_hf`. `dodnet` registers itself on import from
`experiments/multi_organelle/`.

**Necks** — `naive_1x1`, `resnet34_detail`.

**Adaptation** — `frozen`, `lora` (plain or convolutional, per `adapt_params.conv`), `lora_ln`
(`lora` plus trainable LayerNorm affines), `cond_lora` (a LoRA bottleneck modulated per image or per
source), `last_n`, `full`. `conv_lora` is accepted as an alias of `lora`.

## Notes

A decoder with a native instance head has its matching instance-loss term appended automatically, so
a configuration listing only `dice_bce` trains on more terms than it names.

Regions empty in both prediction and ground truth are excluded from every mean, which can change the
denominator between arms.

This directory holds the experiments that
compare segmentation architectures and training settings, not the final benchmarking. 

