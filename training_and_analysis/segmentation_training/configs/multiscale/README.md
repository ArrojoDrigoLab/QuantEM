# Multiscale and two-scale

Two ways of giving the model more than one scale, both measured against the same model at a single
scale. Neither is a configuration file: multi-scale fusion changes only what happens at inference, and
the two-scale model builds its configuration from the shared template at launch.

OmniEM only. Neither mechanism was run on QuantEM.

Reported metrics: [`results/scale_mechanism_omniem.csv`](results/scale_mechanism_omniem.csv). The `base_dice` and
`base_pq` columns are the single-scale model each arm is compared against, from
[`../input_scale/`](../input_scale/).

## Multi-scale test-time fusion

The trained head is applied to each test region resampled to 0.75×, 1× and 1.5×, and the foreground
probability maps are averaged. No retraining.

```bash
python -m segmentation_training.experiments.scale.run_scale multiscale --organelle mito --data-root <native-scale tiles> --head <head.pt> --config <resolved_config.yaml> --run-dir <encoder run dir> --scales 0.75 1.0 1.5 --fuse mean
```

Three arms were reported: mitochondria at native scale and at 16 nm/px, and ER at 4 nm/px. The
`--data-root` and the head select the arm; nothing else changes between them.

## Two-scale co-input

A fine view at the arm's scale and a co-centered coarse view covering four times the field, fused
through the shared encoder by cross-attention. This one trains.

```bash
python -m segmentation_training.experiments.scale.run_scale two-scale --organelle mito --data-root <native-scale tiles> --run-dir <encoder run dir> --fuse xattn --coarse-factor 2
```

Same three arms. For mitochondria the coarse view is a co-centered window at twice the pixel size; for
ER it is a four-times-larger field at 8 nm/px against a 4 nm/px fine view. A single seed.
