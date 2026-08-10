"""The segmentation architecture the released checkpoints load into.

This is the code that produced the published numbers. The released ``head.pt``
files are bare ``state_dict``s, so **the module graph here is part of the model**:
if a layer is renamed or a fusion order changes, the checkpoint either fails to
load or -- worse -- loads into the wrong tensors and silently produces a
different segmentation. Treat the parameter names and the forward arithmetic as
frozen. See ``../README.md``.

What was trimmed
----------------
Everything inference does not execute: training loops, losses, augmentation,
evaluation harnesses, campaign/sweep scripts, the heavy H100-only arms
(Mask2Former / MaskDINO / ViT-Adapter / ViT-CoMer, which need detectron2 or
mmcv), the E1 style-conditioning stack and the E2c conditional adapters. None of
the eight released packs use any of it: all eight carry ``conditioner: None``
and were trained with ``cond.enabled: false``.

What survives is exactly the four-level pipeline the packs declare::

    ENCODER (see ../encoders.py) -> NECK -> DECODER -> logits

with two necks (``naive_1x1``, ``resnet34_detail``), three decoders
(``affinity_mws``, ``upernet``, ``dpt``) and three encoder-adaptation modes
(``last_n``, ``full``, ``lora``). That is the full cross-product the eight packs
span; nothing else is reachable.

This package is private (leading underscore). Import :mod:`quantem.inference`
instead.
"""

from .base import STRIDES, ConvGNAct, SegModel, resize_to
from .decoders import build_decoder
from .load_head import build_and_load_head, build_segmodel, inspect_head
from .necks import build_neck
from .schema import DecoderSpec, EncoderSpec, HeadConfig, NeckSpec, load_head_config

__all__ = [
    "STRIDES",
    "ConvGNAct",
    "DecoderSpec",
    "EncoderSpec",
    "HeadConfig",
    "NeckSpec",
    "SegModel",
    "build_and_load_head",
    "build_decoder",
    "build_neck",
    "build_segmodel",
    "inspect_head",
    "load_head_config",
    "resize_to",
]
