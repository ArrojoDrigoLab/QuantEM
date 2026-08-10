# `experiments/`

Two reported experiments need more than the training harness. Each is a package with its own runner;
everything else runs through `harness/run_seg.py` on a configuration from [`../configs/`](../configs/).

| Package | What it holds |
|---|---|
| `scale/` | Input scale: the rescaled-dataset sweep, multi-scale test-time fusion, and the two-scale co-input model |
| `multi_organelle/` | One head per organelle against a shared organelle-conditioned head |
| `common/` | The shared adapted-base loader, config template, evaluation loop and reporting they share |

Every runner takes the encoder run directory, the ground-truth data root and the output directory as
arguments.
