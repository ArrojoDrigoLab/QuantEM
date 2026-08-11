# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0.
#
# QUANTEM: this file is not upstream's -- upstream's also imports
# SamAutomaticMaskGenerator, which is not vendored. Every other file in this
# directory is upstream, verbatim.
"""Segment Anything (Meta Platforms, Inc.), Apache-2.0.

From https://github.com/facebookresearch/segment-anything at commit
``dca509fe793f601edb92606367a655c15ac00fdf``, trimmed to what box prompting
uses: the ViT-B encoder and ``SamPredictor``. The automatic mask generator,
its ``utils/amg.py`` helpers and the ONNX export path are not included.

Two changes from upstream: this file, and ``predictor.py`` line 10, where
``from segment_anything.modeling import Sam`` became a relative import so it
resolves under this package name.

Needs ``torch``, ``numpy`` and ``torchvision`` -- the last via
``utils/transforms.py``, which is on the prompt path.
"""

from .build_sam import (
    build_sam,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    sam_model_registry,
)
from .predictor import SamPredictor

__all__ = [
    "SamPredictor",
    "build_sam",
    "build_sam_vit_b",
    "build_sam_vit_h",
    "build_sam_vit_l",
    "sam_model_registry",
]
