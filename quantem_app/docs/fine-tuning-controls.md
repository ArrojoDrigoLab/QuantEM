# Fine-tuning controls

Completed areas provide dense supervision: confirmed objects are foreground,
and the rest of each completed area is background. Every rasterized completed
pixel is eligible, including small areas and areas containing no objects.
Unannotated context never enters cross-entropy, Dice, or validation scores.

Head training reads half a model tile of surrounding image context on each
side, clipped to the image boundary. Small areas are centered in their training
window. Larger areas use overlapping windows with a final window aligned to
each edge. Downsampling conditions labels on annotated source pixels, so a
small completed region cannot disappear or inherit an unannotated background
label. The preview reports contributing areas and background-only areas.

The dialog offers quick and extended automatic schedules, plus an explicit
step budget. Quick uses 20 steps per tile bounded to 300–600 steps; extended
raises the ceiling to 20,000. Both automatic schedules allow at least one
complete pass through the tiles. Sampling shuffles each complete pass instead
of drawing with replacement. Explicit budgets that cannot visit every training
tile are rejected.

Optional validation selects the duration by masked validation loss. An inner
split holds back 20% of the training images, independently of any outer test
images. The model is then reset to its initial head and refitted on every
training area for the selected duration, with at least one complete pass.
Validation needs two training images; when unavailable, the requested fixed
duration is used and the saved trace states why. Final test images do not
choose checkpoints, stopping time, or thresholds. Cross-validation continues
to save the final fold's model; its excluded images are recorded in the result.

Training history, validation history, selected step, stopping status, split
identities, and visited-tile counts are retained in the adapter's parameters
and displayed in the completed run. Negative-only data can train suppression
but cannot estimate foreground recall; it retains the default threshold.

## API

`POST /api/finetune/runs/` accepts the following optional fields in addition to
the existing scope and training-mode fields:

| Field | Default | Accepted values |
| --- | --- | --- |
| `schedule` | `quick` | `quick`, `extended` |
| `steps` | automatic | integer 1–20,000, sufficient for one complete pass |
| `lr` | 0.0001 | finite number 0.0000001–0.01 |
| `weight_decay` | 0.0001 | finite number 0–1 |
| `seed` | 0 | integer 0–4,294,967,295 |
| `validation_fraction` | 0 | number 0–0.5; 0 disables selection |
| `eval_interval` | 100 | integer 1–20,000 |
| `patience` | 0 | integer 0–1,000; 0 disables early stopping |
| `min_delta` | 0.0001 | minimum improvement in validation loss, 0–1 |

The progress budget includes both selection and refit when enabled. Early
stopping may finish below that budget. `training_rounds` in the run-detail
response preserves actual steps and both histories after job-log cleanup.
