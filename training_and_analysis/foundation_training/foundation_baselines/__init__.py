"""Public EM / vision foundation-model baselines as frozen feature extractors.

These are the external comparison encoders used in the encoder and decoder comparisons against
publicly available methods. The QuantEM encoders load through the ``dinov3`` framework in
``encoder_evaluation`` and ``segmentation_training``; this package adds only the external ViTs
(EMCF-MAE, Meta-DINOv3, natural-image DINOv2 ViT-L and OmniEM/EM-DINO), which load via ``timm`` and
a local weight file and expose the same tap interface — a list of ``[B, C, H/p, W/p]`` grids from
``timm``'s uniform ``forward_intermediates`` API.

Nothing here trains or fine-tunes: the encoders are frozen and only the downstream decoder trains.
Weights live under an untracked ``foundation_weights/``
and are registered into per-encoder ``checkpoint_index.json`` files by
``register_external_encoders.py`` so both harnesses load them with their existing
manifest-driven code path.
"""

from .external_vit import (
    REGISTRY,
    ExternalEncoderSpec,
    build_external_backbone,
    extract_intermediates,
    load_state_dict_any,
    round_to_patch,
)

__all__ = [
    "REGISTRY",
    "ExternalEncoderSpec",
    "build_external_backbone",
    "extract_intermediates",
    "load_state_dict_any",
    "round_to_patch",
]
