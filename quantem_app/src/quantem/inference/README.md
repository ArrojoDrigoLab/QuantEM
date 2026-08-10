# `quantem.inference` — in-process model inference

Everything runs in this process: tiling, blending, resampling, a model cache and
device selection. There is no subprocess, no second interpreter and no
research-tree import.

## What is implemented here

| Module | Status |
| --- | --- |
| `tiling.py` | **Complete.** 25% overlap, `stride = round(t * (1 - overlap))`, last window flush to the edge, 2-D Hann as `outer(hann(t), hann(t))` plus a `1e-3` floor, weighted accumulate then normalise. Geometry is identical to `fig3/harness/evaluate.py::predict_region`. Adds `BandBlender`: bounded-memory row-band accumulation so peak cost is `2 x tile x width x 4` bytes instead of the whole region. |
| `resample.py` | **Complete.** `pixel_size_nm -> canonical_nm` with area-averaged downsampling, bilinear upsampling, and mask return by NEAREST. Threshold at model scale, then upsample the mask. |
| `device.py` | **Complete.** Three-way `cuda \| mps \| cpu`, env override, graceful fallback, per-device autocast policy. Without MPS every Mac would silently run a ViT-L on CPU. |
| `postprocess.py` | **Complete.** threshold → closing(disk r) → fill holes → label → min-area. Default threshold 0.5 for every organelle and both families. |
| `specs.py` | **Complete.** The eight released models: per-organelle constants merged with `quantem/registry/manifest.py::ARCHITECTURE`. |
| `engine.py` | **Complete.** Cache, eviction, resampling, padding, tile plan, blending, normalisation and the softmax/foreground reduction, plus `build_module`, which rebuilds the pack's architecture and loads the head into it. Still drivable with `forward=` injected. |
| `_fig3/` | **Complete.** The segmentation architecture: neck, decoder, `SegModel`, LoRA adapters, config schema, head loader. See its `__init__.py` for the module-graph contract. |
| `encoders.py` | **Complete.** Three-tier encoder construction: exported artifact, `timm`, `dinov3`. |
| `export.py` | **Complete.** Build-time TorchScript export, verified against the eager encoder. |
| `segmenter.py` | **Complete against the contract.** Runs end to end. |

Nothing imports torch at module import time *except* `encoders.py`, `export.py`
and `_fig3/`, which are only reached from inside `engine.build_module`;
`device.py` and `engine.py` import it inside functions so Django startup does not
pay for it.

## The architecture question, and how it was answered

The released checkpoints are bare `state_dict`s. Loading one requires an object
with the right parameter names and shapes to load it *into*: a ViT-B/16 or
ViT-L/14 encoder, one of two necks (`naive_1x1`, `resnet34_detail`), one of
three decoders (`affinity_mws`, `upernet`, `dpt`), plus the adapter wiring
(`last_n`, `full`, `lora8`) — see the table in `quantem/registry/manifest.py`.

One option was to ship *no* architecture code and require a TorchScript export
of the whole pack. What is implemented is the useful half of that, split by who
owns the code:

* **The neck, decoder and head loader are ours.** They live in `_fig3/`, trimmed
  to the arms the eight packs actually use (2 necks, 3 decoders, 3 adapt modes —
  the full cross-product the packs span, and nothing else). Keeping them eager
  costs no licence surface, keeps a head inspectable and fine-tunable, and means
  the export step is not on the critical path for anyone.
* **The encoder is where the packaging problem really was**, and it is what
  `export.py` exports. The OmniEM ViT-L needs only `timm`, already a dependency,
  so that family runs out of the box. The QuantEM ViT-B needs Meta's `dinov3`,
  which QuantEM does not redistribute and does not depend on — so that family is
  exported once, on a machine that has it, and afterwards loads from a
  self-contained `encoder_ts.pt` that nothing else can reach into.

The export is per **pack**, not per encoder, because no pack runs its base
encoder unmodified: LoRA adapters are forward hooks inside the blocks,
`last_n` replaces the last four blocks and `full` replaces the whole backbone.
`export.py`'s docstring has the details.

The rejected alternative was vendoring the *encoder* code too. That would put a
research codebase's torch/timm constraints into a desktop app and redistribute a
package we have no licence to redistribute.

Two guards exist because the failure mode here is a plausible wrong answer
rather than a crash: `_fig3/load_head.py` refuses a partial state-dict load
(upstream only warned), and `engine._check_contract` refuses an encoder whose
input scaling, patch size or pack id disagrees with the spec.

## Known limitation: instance splitting

Post-processing is connected components. Two touching mitochondria stay one
object. The `affinity_mws` decoders emit affinities designed to be resolved by a
mutex watershed; they are exposed on `SegModel.aux_logits` (`[B, 10, t, t]` in
`[0, 1]`, one channel per offset) and a test pins them there, but the watershed
itself is not vendored — it needs `elf`/`affogato`. It is one component away
from being fixed.

## Open items

* `pixel_size_nm` is not yet threaded through. `quantem.segmentation.organelle_tasks`
  builds the segmenter kwargs and must pass the asset's numeric pixel size;
  without it a model with a `canonical_nm` (mito/LD 8 nm, nucleus 25 nm) runs at
  native scale. There is a `TODO(quantem)` at the constructor.
* Remote model download is a stub (`registry/install.py`). The eight packs are
  local files and the app is offline; `install_pack_from_manifest` documents what
  the fetch must do when it lands.
* Full-image streaming is plumbed but not enabled: `predict_region_streaming`
  and `BandBlender` are ready, and `BaseSegmenter.predict_from_image_file` is the
  hook, but band-wise labeling needs seam stitching (an instance crossing a band
  boundary must not become two). `supports_image_file_prediction` stays False
  until that exists.

## Reproducibility notes

* Overlap 0.25, threshold 0.5 and the Hann floor `1e-3` match `fig3`'s
  `EvalSpec` defaults, which is what every published number was produced with.
* Encoder normalisation (`ENCODER_NORM` in the manifest) appears in none of the
  eight `resolved_config.yaml` files. Inference is not reproducible from those
  YAMLs alone, which is why the values are pinned in the manifest and read here.
* Downsampling uses `cv2.INTER_AREA`. The pipeline that *built the training
  crops* (`fig3/dataprep/resample.py`) used `scipy.ndimage.zoom(order=1)`, i.e.
  bilinear in both directions. For upsampling the two agree; for downsampling
  `INTER_AREA` antialiases properly but is not bit-identical to training-time
  preprocessing. `resample.to_model_scale(..., downscale_interpolation=cv2.INTER_LINEAR)`
  reproduces the training behaviour exactly if a comparison ever needs it.
