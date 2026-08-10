"""In-process model inference.

Inference runs in this process — no subprocess bridge, no second interpreter.
See README.md in this directory for module status and what is still missing.

Layout::

    specs.py        which models exist and what they expect
    device.py       cuda | mps | cpu selection
    resample.py     native <-> canonical nm/px
    tiling.py       sliding windows, 25% overlap, Hann blending
    engine.py       load a pack once, run it over a region
    encoders.py     build the foundation encoder: exported | timm | dinov3
    export.py       build-time TorchScript export (a CLI, not a runtime path)
    _fig3/          the vendored architecture the released heads load into
    postprocess.py  probability map -> instance labels
    segmenter.py    the BaseSegmenter implementation the app resolves

Only :mod:`quantem.inference.segmenter` touches Django; everything else is
numpy/torch and can be exercised standalone. Nothing re-exported here imports
torch at module import time -- ``encoders``, ``export`` and ``_fig3`` do, and are
reached only from inside :func:`quantem.inference.engine.build_module`.
"""

from .postprocess import DEFAULT_THRESHOLD
from .specs import FAMILIES, MODEL_SPECS, ORGANELLES, ModelSpec, OrganelleSpec
from .tiling import DEFAULT_OVERLAP

__all__ = [
    "DEFAULT_OVERLAP",
    "DEFAULT_THRESHOLD",
    "FAMILIES",
    "MODEL_SPECS",
    "ORGANELLES",
    "ModelSpec",
    "OrganelleSpec",
]
