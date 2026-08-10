# Configurations

One directory per reported experiment, each holding its arms and, in `results/`, the metrics reported
for it. See [`../EXPERIMENTS.md`](../EXPERIMENTS.md) for the list.

No configuration carries a path. The encoder run directory, the ground-truth data root and the output
directory are supplied at launch:

```bash
python -m segmentation_training.harness.run_seg \
    --config segmentation_training/configs/decoder/mito_quantem_affinity_mws.yaml \
    --data-root <ground-truth tiles> --run-dir <encoder run dir> --output-dir <run dir>
```

The test-time arms run against an already-trained head:

```bash
python -m segmentation_training.harness.tta \
    --config segmentation_training/configs/test_time_support/mito_omniem_pixel_similarity_gt.yaml \
    --run-dir <encoder run dir> --head <head.pt> --data-root <ground-truth tiles> \
    --output-dir <run dir> --split test
```

`multiscale/` and `multi_organelle/` build their configurations at launch; see their own READMEs.
