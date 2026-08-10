# Multi-organelle

One head per organelle against a single head shared across organelles and conditioned on an organelle
code. Three seeds per arm, on both encoders.

Two arms are compared:

| Arm | Configuration |
|---|---|
| Per-organelle adapters — each organelle gets its own adapter set, neck and decoder | `multi_perorg_er.yaml`, `multi_perorg_mito.yaml` |
| Shared DoDNet — one organelle-conditioned head, width 32, unbalanced sampling | `multi_dodnet_mid32.yaml` |

The configurations are encoder-parameterised: the same file runs on either encoder, selected by
`--run-dir`, which is why they name neither.

Reported metrics: [`results/multi_organelle.csv`](results/multi_organelle.csv) — Dice for ER and
mitochondria on each encoder, with the seed standard deviation, and panoptic quality where the arm
produces instances.

## Running

```bash
python -m segmentation_training.experiments.multi_organelle.run_multi_organelle dodnet \
    --organelles mito er --data-roots <data_roots.json> --run-dir <encoder run dir> \
    --mid-channels 32 --balance raw --seed 0

python -m segmentation_training.experiments.multi_organelle.run_multi_organelle per-organelle \
    --organelle mito --data-root <ground-truth tiles> --run-dir <encoder run dir> --seed 0
```

The runner also exposes a dynamic-head width sweep and a balanced-sampling variant; `--help` covers
those flags.
