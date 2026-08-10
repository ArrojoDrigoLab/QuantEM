"""Canonical constants for QuantEM inference.

Every value here matches the released artifacts and the manuscript. Nothing is re-derived at
runtime.

Why these values
----------------
``IGNORE_INDEX`` / ``BACKGROUND`` / ``FOREGROUND``
    255 doubles as the loss and metric ignore index; it is also mmsegmentation's default and
    originates in Cityscapes' ``out of roi`` void class, so it survives round-tripping through
    uint8 TIFFs.

``QUANTEM_MEAN`` / ``QUANTEM_STD``
    The *pinned* pretraining values (0.583175 / 0.244468), deliberately not the recomputed
    corpus values (0.580959 / 0.242007). The pretraining config pins the former on purpose, and
    inference must match what the encoder was trained under.

``OMNIEM_*``
    The dataset hands the OmniEM encoder a raw [0, 1] single-channel tile (mean 0 / std 1);
    channel replication and the per-channel normalisation below happen *inside* the encoder's
    preprocess.
"""

from __future__ import annotations

# --- derived-mask encoding ------------------------------------------------------------------
BACKGROUND = 0
FOREGROUND = 1
IGNORE_INDEX = 255

# --- normalisation --------------------------------------------------------------------------
QUANTEM_MEAN = 0.583175
QUANTEM_STD = 0.244468

# What the dataset feeds an OmniEM encoder: raw [0, 1].
OMNIEM_DATASET_MEAN = 0.0
OMNIEM_DATASET_STD = 1.0
# What the OmniEM encoder applies internally, after replicating 1 -> 3 channels.
OMNIEM_ENCODER_MEAN = 0.595446
OMNIEM_ENCODER_STD = 0.211906

# --- tiled inference ------------------------------------------------------------------------
# Ported from segmentation_training/harness/evaluate.py lines 33-147.
DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP = 0.25
DEFAULT_FG_THRESHOLD = 0.5
DEFAULT_INSTANCE_MIN_SIZE = 16

#: Hann window floor. Load-bearing: without it, a pixel covered by exactly one window whose
#: Hann weight is 0 at the border gets zero total weight and the blend divides by ~0.
HANN_FLOOR = 1e-3

#: Padding for partial tiles. 0-pad ("honest border"), never reflect.
PAD_MODE = "constant"

#: Feature-pyramid strides the necks emit, relative to the input grid.
STRIDES = (4, 8, 16, 32)

# --- encoder-side ---------------------------------------------------------------------------
#: Prefix (non-patch) token counts. QuantEM = 1 CLS + 4 storage/register tokens; OmniEM
#: (DINOv2 ViT-L/14) = 1 CLS. Wrong values silently corrupt LoRA feature splitting.
N_PREFIX_TOKENS = {"quantem": 5, "omniem": 1}
