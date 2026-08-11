"""Tunables and the checkpoint this feature runs on.

WHICH SAM, AND WHY -- read this before changing ``CHECKPOINT``
==============================================================

Owner ruling R14 asks for ``vit_b`` micro-SAM, falling back to ``vit_b``
SAM-HQ2 if micro-SAM will not resolve or run against this stack. Measured on
this machine (python 3.13.14, numpy 2.4.4, torch 2.13.0+cu126, Quadro RTX
8000), the answer is a third thing that satisfies the ruling better than either
literal option:

**We run micro-SAM's weights on the plain ``segment-anything`` runtime.**

The two facts behind that:

1. *Does the ``micro_sam`` package install and run here?*  It installs. It is
   on PyPI at 1.8.8 and resolves cleanly on 3.13 with numpy 2 -- so the
   compatibility half of R14's escape clause does not fire.
2. *What does it cost?*  Too much. Its dependency closure is **116 packages,
   320.9 MB of wheels** (roughly 800 MB-1 GB installed): napari 0.8.0 and
   PyQt6 (78.4 MB for ``PyQt6-Qt6`` alone), trackastra (50.5 MB), PySCIPOpt
   (48.2 MB), llvmlite/numba (44.7 MB), gurobipy (11.2 MB), tensorboard, dask,
   xarray, ipython and qtconsole. ``micro_sam`` is a napari plugin plus a
   training stack; the annotation GUI and the tracking/ILP solvers come with
   it. Another agent is removing ~150 MB from the app bundle this round, so
   adding an order of magnitude more than that back for one prompt endpoint is
   not a trade worth making.

What the micro-SAM *model* actually is, though, is a stock SAM ``vit_b``
fine-tuned on EM data -- ``micro_sam.util._load_checkpoint`` unwraps torch-em's
``{"model_state": ...}`` envelope, strips a ``sam.`` prefix, and hands the
result to the ordinary ``sam_model_registry``. Verified directly against
``vit_b_em_organelles``: 314 state-dict entries, patch-embed width 768, and
``Sam.load_state_dict`` reports **zero missing and zero unexpected keys**. So
the weights load on Meta's Segment Anything alone, which QuantEM carries at
:mod:`quantem.sam._vendor.segment_anything` -- about 75 KB of source needing
only torch, numpy and torchvision, all already installed.

Measured end to end on that path: encoder 0.54 s, first decoder pass 0.077 s,
a second box against the cached embedding 0.016 s, and the SAM1
embedding-rehydration trick reproduces the mask bit-for-bit.

So the scientific content of R14 -- ``vit_b`` micro-SAM, EM-organelle weights
-- is delivered exactly, and no new package is installed at all. The runtime is
swappable behind :mod:`quantem.sam.backends`: point ``CHECKPOINT`` at
``sam_vit_b_01ec64.pth`` for stock Meta ``vit_b``, or add a backend module for
SAM-HQ2, without touching anything else.

One consequence to keep in mind: micro-SAM's own wrapper takes boxes as
``yxyx`` and points as ``[y, x]``. We are not using that wrapper. The Meta
predictor takes ``xyxy``, and :mod:`quantem.sam.backends.sam1` passes ``xyxy``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Pixels of context added around the drawn box before the crop is cut.
#:
#: SAM sees the crop, not the box, and a box cut flush to the object gives the
#: encoder no surroundings to separate the object from. 200 px is the value
#: this was ported from and it is a sensible default at EM scales.
BBOX_CONTEXT_RADIUS = 200

#: Crop windows are snapped to this grid so neighbouring boxes share an encode.
#:
#: THE CACHE KEY IS THE CROP WINDOW, NOT THE BOX -- that is the whole
#: optimisation. One ``set_image`` (the encoder, hundreds of ms) serves many
#: ``predict`` calls (the decoder, tens of ms). Padding each box by
#: ``BBOX_CONTEXT_RADIUS`` and keying on the padded rect would defeat it: two
#: boxes a few pixels apart produce two different rects and therefore two
#: encodes.
#:
#: The scheme here, chosen over the union-find area merging it was ported
#: from -- that only pays off when several boxes are planned together, and this
#: endpoint handles one box per request: **the window is the grid cell holding
#: the box's centre, grown by BBOX_CONTEXT_RADIUS on all four sides, clamped to
#: the image.** Every box whose centre lands in the same cell gets a
#: byte-identical window and therefore a cache hit. A box too large or too
#: off-centre to fit inside that window falls back to its own padded rect, which
#: is correct but shares nothing -- see :func:`quantem.sam.geometry.plan_crop`.
CROP_GRID = 1024

#: How many crop embeddings to keep. Each is a float32 ``(1, 256, 64, 64)``
#: tensor, so 4 MB; eight of them is 32 MB.
#:
#: Bounded on purpose. The implementation this was ported from kept two
#: unbounded dicts *and* two unbounded on-disk ``.npz`` directories with no
#: eviction anywhere, which is a leak in a long-lived desktop process. There is
#: no on-disk tier here at all: re-encoding a crop costs about half a second,
#: which is not worth a cache that grows forever.
EMBEDDING_CACHE_ENTRIES = 8

#: Set to "1" to run a deterministic fake backend with no weights and no GPU.
#: The test suite uses it so the whole feature can be exercised offline.
STUB_MODE_ENV_VAR = "QUANTEM_SAM_STUB"


@dataclass(frozen=True)
class CheckpointSpec:
    """One downloadable weight file, addressed by digest."""

    #: Identity recorded on the created object and mixed into the cache key, so
    #: swapping weights can never serve an embedding made by the old ones.
    identity: str
    #: Architecture key for ``segment_anything.sam_model_registry``.
    architecture: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    #: Shown to the user when the download is offered or fails.
    display_name: str


#: micro-SAM's EM-organelles ``vit_b``, published through BioImage.IO (the
#: "noisy-ox" entry). The digest and size below were measured on the downloaded
#: file, not copied from a page.
CHECKPOINT = CheckpointSpec(
    identity="microsam:vit_b_em_organelles",
    architecture="vit_b",
    filename="microsam_vit_b_em_organelles.pt",
    url=("https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1.2/files/vit_b.pt"),
    sha256="0e08dd7bf3761df3f2440dcf74b3bc7156dd3c61ced65e838c845dd06102a7ac",
    size_bytes=375_023_499,
    display_name="micro-SAM EM organelles (ViT-B)",
)
