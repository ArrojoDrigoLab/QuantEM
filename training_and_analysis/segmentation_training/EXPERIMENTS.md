# Experiments

One directory per reported experiment under `configs/`, each holding its arms and, in `results/`, the
metrics reported for it.

| Experiment | Configurations | OmniEM | QuantEM | Results |
|---|---|---|---|---|
| Encoder adaptation | `encoder_adaptation/` | 4 arms | 4 arms | `adaptation.csv` |
| Learning rate | `learning_rate/` | 4 rates | — | `lr_sweep_omniem.csv` |
| Input scale | `input_scale/` | 4 per organelle | 3 per organelle | `scale_omniem_er.csv`, `scale_omniem_mito.csv`, `scale_quantem.csv` |
| Multiscale and two-scale | `multiscale/` | 2 mechanisms | — | `scale_mechanism_omniem.csv` |
| Decoder | `decoder/` | 6 ER, 9 mitochondria | 6 ER, 9 mitochondria | `decoder.csv` |
| Neck | `neck/` | 4 decoders × 2 necks | 4 decoders × 2 necks | `neck_adapted.csv` |
| Loss function | `loss_function/` | 3 arms, ER | 3 arms, ER | `loss_native_adapted_er.csv` |
| Image-style conditioning | `image_style_conditioning/` | 4 arms | 4 arms | `conditioning.csv` |
| Test-time support | `test_time_support/` | 8 arms | — | `support_omniem.csv` |
| Test-time adaptation | `test_time_adaptation/` | — | 3 arms | `tta_quantem.csv` |
| Multi-organelle | `multi_organelle/` | 2 arms | 2 arms | `multi_organelle.csv` |
| Released models | `released_models/` | 2 heads | 2 heads | `released_models.csv` |

Test-time support arms are scored at tile scale, so only their delta against the same model with
support disabled is comparable with the other experiments.

Multiscale, two-scale and multi-organelle build their configurations at launch; see
`experiments/` and the READMEs in those two config directories.

## Frozen against best configuration

| Encoder | Organelle | Frozen | Best | Configuration of the best arm |
|---|---|--:|--:|---|
| OmniEM | mitochondria | 0.467 | 0.746 | `multi_organelle/multi_perorg_mito.yaml` |
| OmniEM | ER | 0.104 | 0.516 | `loss_function/er_omniem_native_dice_bce_cldice_skeleton_recall.yaml` |
| QuantEM | mitochondria | 0.470 | 0.723 | `input_scale/scale_mito_native_quantem.yaml` |
| QuantEM | ER | 0.008 | 0.458 | `input_scale/scale_er_native_quantem.yaml` |

Frozen values are the frozen rows of `encoder_adaptation/results/adaptation.csv`. The best values are
single runs; the tables above report the seed means where an arm was repeated.
