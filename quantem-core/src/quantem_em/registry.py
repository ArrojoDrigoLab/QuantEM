"""The eight released QuantEM / OmniEM organelle models.

Every field below was read from the staged ``resolved_config.yaml`` files and cross-checked
against ``training_and_analysis/segmentation_training/configs/released_models/*.yaml``, which
carry identical values. Verified 2026-08-06.

Encoder-sharing is **not** uniform, and the artifact split reflects that (measured from the eight
``head.pt`` files):

* OmniEM uses LoRA, so its ViT-L is untouched and genuinely shared by all four heads.
* QuantEM mito/nucleus/LD fine-tune the **last four blocks**, which therefore differ per organelle;
  only blocks 0-7 + embeddings + final norm are shared (the "trunk").
* QuantEM ER is ``adapt: full`` — it replaces the entire encoder, so it needs no trunk at all.
"""

from __future__ import annotations

from .constants import (
    OMNIEM_DATASET_MEAN,
    OMNIEM_DATASET_STD,
    OMNIEM_ENCODER_MEAN,
    OMNIEM_ENCODER_STD,
    QUANTEM_MEAN,
    QUANTEM_STD,
)
from .spec import EncoderSpec, ModelSpec

# --- encoders ---------------------------------------------------------------------------------

QUANTEM_VITB = EncoderSpec(
    family="quantem",
    # timm >= 1.0.20, in timm/models/eva.py. The *_qkvb variant is required: our checkpoint was
    # trained from scratch with mask_k_bias=False, so its k-bias is live (max |k| = 7.0 at block 5)
    # -- unlike Meta's distilled releases, whose all-zero biases motivated the plain variant.
    timm_model="vit_base_patch16_dinov3_qkvb",
    patch_size=16,
    embed_dim=768,
    depth=12,
    in_chans=1,
    n_prefix_tokens=5,  # 1 CLS + 4 storage tokens
    dataset_mean=QUANTEM_MEAN,
    dataset_std=QUANTEM_STD,
    strip_prefix="backbone.",
    checkpoint_key="teacher",
    rope_periods_bf16=True,  # reference stores periods as bf16; timm regenerates fp32
    # DINOv3's `layernorm` default. timm's dinov3 entries hard-code 1e-5 (DINOv3's
    # `layernormbf16`), which is right for Meta's released weights and wrong for ours.
    norm_eps=1e-6,
)

OMNIEM_VITL = EncoderSpec(
    family="omniem",
    timm_model="vit_large_patch14_dinov2.lvd142m",
    patch_size=14,
    embed_dim=1024,
    depth=24,
    in_chans=3,  # 1 -> 3 channel replication happens in preprocess
    n_prefix_tokens=1,  # CLS only
    dataset_mean=OMNIEM_DATASET_MEAN,
    dataset_std=OMNIEM_DATASET_STD,
    encoder_mean=OMNIEM_ENCODER_MEAN,
    encoder_std=OMNIEM_ENCODER_STD,
    strip_prefix="vit.",
)

# --- per-organelle post-processing ------------------------------------------------------------
# min_area: owner ruling 2026-08-06 -- 100 px for every organelle except nucleus, 500 px for
# nucleus. Fixed, and deliberately NOT exposed in the UI.
_POST = {
    "mito": (100, 3),
    "er": (100, 1),
    "ld": (100, 2),
    "nucleus": (500, 12),
}

_LORA_R8 = {"rank": 8, "conv": False}
_LAST4 = {"n": 4}


def _omniem(organelle, arm, neck, decoder, canonical_nm, task):
    min_area, close_radius = _POST[organelle]
    return ModelSpec(
        model_id=f"omniem/{organelle}",
        organelle=organelle,
        arm_name=arm,
        encoder=OMNIEM_VITL,
        neck=neck,
        decoder=decoder,
        adapt="lora",
        adapt_params=dict(_LORA_R8),
        canonical_nm=canonical_nm,
        task=task,
        min_area=min_area,
        close_radius=close_radius,
        trunk_artifact="omniem-vitl",
        model_artifact=f"omniem-{organelle}",
    )


def _quantem(organelle, arm, neck, decoder, adapt, canonical_nm, task):
    min_area, close_radius = _POST[organelle]
    # adapt="full" replaces the whole encoder -> the model artifact is self-contained.
    trunk = None if adapt == "full" else "quantem-vitb-trunk"
    return ModelSpec(
        model_id=f"quantem/{organelle}",
        organelle=organelle,
        arm_name=arm,
        encoder=QUANTEM_VITB,
        neck=neck,
        decoder=decoder,
        adapt=adapt,
        adapt_params=dict(_LAST4) if adapt == "last_n" else {},
        canonical_nm=canonical_nm,
        task=task,
        min_area=min_area,
        close_radius=close_radius,
        trunk_artifact=trunk,
        model_artifact=f"quantem-{organelle}",
    )


REGISTRY: dict[str, ModelSpec] = {
    m.model_id: m
    for m in (
        # -- mitochondria: 8 nm/px, instance head ------------------------------------------
        _omniem("mito", "F4v2_omni_cem", "naive_1x1", "affinity_mws", 8.0, "instance"),
        _quantem("mito", "F4v2_qem_cem", "naive_1x1", "affinity_mws", "last_n", 8.0, "instance"),
        # -- endoplasmic reticulum: NATIVE resolution, semantic head -----------------------
        _omniem("er", "F4_omni_er", "resnet34_detail", "dpt", None, "semantic"),
        _quantem("er", "F4_qem_er", "resnet34_detail", "upernet", "full", None, "semantic"),
        # -- nucleus: 25 nm/px --------------------------------------------------------------
        _omniem("nucleus", "F4_omni_nuc", "naive_1x1", "affinity_mws", 25.0, "instance"),
        _quantem("nucleus", "F4_qem_nuc", "naive_1x1", "affinity_mws", "last_n", 25.0, "instance"),
        # -- lipid droplets: 8 nm/px --------------------------------------------------------
        _omniem("ld", "F4_omni_ld", "naive_1x1", "affinity_mws", 8.0, "instance"),
        _quantem("ld", "F4_qem_ld", "naive_1x1", "affinity_mws", "last_n", 8.0, "instance"),
    )
}

#: UI defaults, per owner ruling 2026-08-06: QuantEM for mitochondria, OmniEM elsewhere.
DEFAULT_MODEL_FOR_ORGANELLE = {
    "mito": "quantem/mito",
    "er": "omniem/er",
    "nucleus": "omniem/nucleus",
    "ld": "omniem/ld",
}

ORGANELLE_LABELS = {
    "mito": "Mitochondria",
    "er": "Endoplasmic reticulum",
    "nucleus": "Nucleus",
    "ld": "Lipid droplets",
}


def list_models() -> list[ModelSpec]:
    return list(REGISTRY.values())


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return REGISTRY[model_id]
    except KeyError:
        raise KeyError(f"unknown model id {model_id!r}; known ids: {sorted(REGISTRY)}") from None
