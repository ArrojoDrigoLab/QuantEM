# Test-time adaptation

Three support arms on QuantEM, varying where the support comes from and how it is combined with the
first-pass prediction. Each arm has an ER and a mitochondria configuration, named by the `er_` and
`mito_` prefixes. The unadapted reference is carried in the reported metrics as its own row for
context.

| Arm | Support | Combination |
|---|---|---|
| `*_support_inferred_gated_uncertainty` | first-pass prediction | uncertainty gating |
| `*_support_inferred_gated_residual` | first-pass prediction | residual |
| `*_support_gt_uncertainty` | manual annotation | uncertainty gating |

Reported metrics: [`results/tta_quantem.csv`](results/tta_quantem.csv).

```bash
python -m segmentation_training.harness.tta \
    --config segmentation_training/configs/test_time_adaptation/er_quantem_support_gt_uncertainty.yaml \
    --run-dir <encoder run dir> --head <head.pt> --data-root <ground-truth tiles> \
    --output-dir <run dir> --split test
```
