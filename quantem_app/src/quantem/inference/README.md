# `quantem.inference`

Model inference, in process: tiling, blending, resampling, device selection and
a model cache.

| Module | What it does |
| --- | --- |
| `tiling.py` | Splits a region into overlapping tiles and blends them back with a 2-D Hann window. `BandBlender` accumulates by row band so peak memory does not scale with the region. |
| `resample.py` | Converts between the image's pixel size and a model's canonical scale. The probability field returns to the image's own pixel grid before anything is thresholded. |
| `device.py` | Chooses CUDA, MPS or CPU, honours an override, and falls back rather than failing. |
| `postprocess.py` | Threshold, closing, hole filling, connected components, minimum area. |
| `specs.py` | Per-organelle constants for the released models. |
| `engine.py` | Loads and caches models, plans tiles, runs the forward pass, reduces logits to a foreground probability. |
| `encoders.py` | Builds an encoder from an exported artifact, from `timm`, or from `dinov3`. |
| `export.py` | Exports an encoder to TorchScript at build time and checks it against the eager one. |
| `segmenter.py` | The segmenter the application calls. |
| `_fig3/` | The segmentation architecture: neck, decoder, adapters, head loader. |

Torch is imported inside functions, not at module scope, except in `encoders.py`,
`export.py` and `_fig3/`, which are reached only from `engine.build_module`. An
application that never runs a model does not pay for torch.

## Encoders

A released pack is a bare `state_dict`, so loading one needs matching
architecture code to load it into.

The neck, decoder and head loader are in `_fig3/` and stay eager, which keeps a
head inspectable and fine-tunable.

Encoders are handled per pack rather than per architecture, because no pack runs
its base encoder unmodified — adapters are forward hooks inside the blocks, and
some packs replace the last blocks or the whole backbone. The OmniEM ViT-L builds
through `timm`. The QuantEM ViT-B builds through `timm` as well when the pack's
index says so, and otherwise loads a self-contained TorchScript encoder exported
at build time.

Two loads refuse rather than guess, because a wrong answer here is plausible
instead of loud: `_fig3/load_head.py` rejects a partial state-dict load, and
`engine._check_contract` rejects an encoder whose input scaling, patch size or
pack id disagrees with the spec.

## Instance splitting

Objects come from connected components, so two touching organelles are one
object. The `affinity_mws` decoders also emit affinities on
`SegModel.aux_logits`, which a mutex watershed could resolve; nothing in the
application reads them today.
