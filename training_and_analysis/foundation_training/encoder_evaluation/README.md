# `encoder_evaluation/`

How the encoders were compared. A segmentation decoder is trained on a candidate encoder's features and
scored on the held-out split, with everything except the encoder held fixed, so a difference in Dice is
the encoder's. This is the evaluation half of foundation training: the pretraining recipes are chosen
here, not from the pretraining loss.

Metrics are Dice for both organelles, and panoptic quality for mitochondria where the arm produces
instances. Reported values are in [`results/`](results/).

## Two evaluation regimes

The reported experiments were not all evaluated the same way, and the configurations differ.

**Probe evaluation** — a UPerNet decoder on the encoder, 4,000 steps, batch 32, evaluated by sliding
window over full annotated regions. This is the standardised setting the later comparisons use, and
everything reported per checkpoint or per pretraining recipe runs under it.

| Configuration | Encoder |
|---|---|
| [`configs/probe_upernet_frozen.yaml`](configs/probe_upernet_frozen.yaml) | frozen |
| [`configs/probe_upernet_frozen_cached.yaml`](configs/probe_upernet_frozen_cached.yaml) | frozen, with the feature cache |
| [`configs/probe_upernet_lora.yaml`](configs/probe_upernet_lora.yaml) | LoRA, rank 8 |
| [`configs/probe_upernet_last4.yaml`](configs/probe_upernet_last4.yaml) | last four blocks |

The cached variant forwards the encoder over the training set once per checkpoint and organelle rather
than every step. Caching is only valid with a frozen encoder — cached features are static — so the two
adapted configurations disable it and are otherwise identical to the frozen one.

The adapted probe configurations use the same adaptation mechanism and the same learning rates as the
[encoder-adaptation experiment](../../segmentation_training/configs/encoder_adaptation/) in
segmentation training, so the two are directly comparable.

**Linear probe** — [`configs/probe_linear.yaml`](configs/probe_linear.yaml) and
[`configs/probe_linear_label_efficiency.yaml`](configs/probe_linear_label_efficiency.yaml): a
capacity-minimal linear head instead of UPerNet, so the score depends as little as possible on decoder
capacity. This is the setting behind the label-efficiency curves, whose four label fractions are set in
the second of those two files.

[`configs/final_comparison_upernet.yaml`](configs/final_comparison_upernet.yaml) and
[`configs/final_comparison_unet.yaml`](configs/final_comparison_unet.yaml) fix one decoder across
QuantEM and the public foundation encoders, and add the context sweep for the RoPE encoders.

## Reported results

| File | Experiment | Organelles |
|---|---|---|
| [`results/encoder_comparison.csv`](results/encoder_comparison.csv) | The public encoders, frozen and at each adaptation level | ER, mito |
| [`results/quantem_adaptation.csv`](results/quantem_adaptation.csv) | QuantEM at each adaptation level | ER, mito |
| [`results/pretraining_ablation.csv`](results/pretraining_ablation.csv) | Pretraining recipe: two base runs, Gram anchoring, three context-size schedules | ER, mito |
| [`results/metadata_ablation.csv`](results/metadata_ablation.csv) | Continued pretraining while preserving or suppressing scale and modality | ER, mito |
| [`results/checkpoint_progression.csv`](results/checkpoint_progression.csv) | The released encoder across training, at the checkpoints a decoder was trained on | ER, mito |
| [`results/label_efficiency.csv`](results/label_efficiency.csv) | Decoder trained on 1 to 100 per cent of the labels, per encoder | mito |

These are the reported tables: one tidy row per arm and organelle, assembled from the per-head records
the harness writes under `--out`. The harness's own roll-up is wider and uses its internal column names
(`macro_dice`, `dice_ci_lo`, `n_evaluated`, `label_fraction`); these files rename to the reported form
and add the arm labels used in the figures.

Column meanings: `dice_lo` / `dice_hi` are bootstrap confidence bounds on the macro Dice. `pq` is
panoptic quality, reported for mitochondria only where the arm produces instances, so it is empty on ER
rows. `n_eval` is the number of scored crops — crops where the prediction and the ground truth are both
empty are excluded, since Dice is undefined there, so the count varies slightly between arms.

More encoder checkpoints were exported during pretraining than had decoders trained on them, so
`checkpoint_progression.csv` covers the subset that was evaluated; its rows come from the LoRA-adapted
probe, matching the reported curve.

**The encoder-comparison and QuantEM-adaptation rows come from the segmentation training harness**, not
from this probe: a resnet34 detail neck and the organelle's decoder, Dice+BCE, 10,000 steps at batch 8,
encoder learning rate 1e-3 for LoRA and 1e-4 for last-4. Those configurations are in
[`../../segmentation_training/configs/encoder_adaptation/`](../../segmentation_training/configs/encoder_adaptation/),
one per organelle and adaptation level; the encoder is whichever run directory `encoder.run_dir` points
at, so the same file reproduces the row for any of the compared encoders. Everything else here is this
probe.

## Running

Every number this probe produced came from the batch orchestrator, which scans a root of encoder runs
and dispatches one head per (checkpoint × organelle × decoder × label fraction) across the available
GPUs. It is idempotent, so an interrupted sweep resumes, and it writes both the per-head records and
the rolled-up summary that the files in `results/` are derived from:

Run these from the `foundation_training/` directory, so the config paths resolve:

```bash
python -m encoder_evaluation.train_decoders_and_test \
    --runs-root <encoder runs> --derived-root <ground-truth tiles> --out <results dir> \
    --config encoder_evaluation/configs/probe_upernet_frozen.yaml \
    --decoders upernet --organelles mito er --n-checkpoints 999
```

Add `--native-tile-size` to probe each encoder at the crop size it was pretrained at, taken from its
manifest, rather than at the config's `tile_size`. The context-size arms are compared that way, so a
768 or 1024 arm is fed the window its receptive field expects.

`harness/run_probe.py` is the single-run entry point behind it, useful for one encoder at a time:

```bash
python -m encoder_evaluation.harness.run_probe \
    --run-dir <encoder run dir> --derived-root <ground-truth tiles> \
    --config encoder_evaluation/configs/probe_upernet_frozen.yaml \
    --organelles mito er --output-dir <results dir>
```

`--run-dir` points at a directory holding `checkpoint_index.json`; the probe selects evenly spaced
checkpoints from it, so one invocation produces a progression curve.

## Ground truth

The probe scores against manually annotated mitochondria and ER. `--derived-root` is a folder of
`(EM, mask)` crops with a `manifest.jsonl`, built from the annotated sources by:

```bash
python -m encoder_evaluation.dataprep.build_dataset \
    --corpus-root <annotated sources> --out <ground-truth tiles> \
    --organelles mito er --splits train test
```

`--corpus-root` is a directory of annotated source datasets plus a `splits/` folder of per-organelle
split CSVs; [`../../segmentation_dataset/`](../../segmentation_dataset/) is the source inventory. The
builder re-reads those CSVs and the dataset manifests every run and skips anything absent from disk, so
it builds whatever subset is present.

**This is not the benchmark dataset.** The corpus used here is a subset of the one behind the final
benchmarks: CEM is absent, several sources are present in smaller versions, and the splits are drawn
differently and are not whole-source holdouts. Every comparison within this folder used the same training and test data, so the
arms are comparable with each other; they are not directly comparable with the benchmark numbers.
