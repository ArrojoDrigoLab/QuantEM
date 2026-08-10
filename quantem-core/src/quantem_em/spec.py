"""``ModelSpec`` — the frozen description of one released segmentation model.

A ``ModelSpec`` is *data*, not a parsed training config. The eight staged
``resolved_config.yaml`` files are 145 lines each, of which about twelve matter at inference; the
rest describe data roots, augmentation, the optimiser, bootstrap CIs and 60+ conditioning fields
that no released model uses (``cond.enabled: false`` in all eight). They are transcribed to
literals once, at packaging time, and **nothing reads a training config at runtime**.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EncoderSpec:
    """How to build and normalise one of the two base encoders."""

    family: str  # "quantem" | "omniem"
    timm_model: str  # timm.create_model name
    patch_size: int
    embed_dim: int
    depth: int
    in_chans: int  # what the timm model is built with
    n_prefix_tokens: int  # 1 CLS + register/storage tokens
    dataset_mean: float  # fed to normalize_em
    dataset_std: float
    encoder_mean: float | None = None  # applied inside the encoder, after channel replication
    encoder_std: float | None = None
    strip_prefix: str | None = None  # checkpoint key prefix to strip
    checkpoint_key: str | None = None  # sub-dict of the checkpoint to read
    #: Install the RoPE period buffer as true bfloat16, which is what drives the whole rotary
    #: computation's precision in both implementations (QuantEM only).
    rope_periods_bf16: bool = False
    #: LayerNorm epsilon. **Not cosmetic.** DINOv3 offers ``layernorm`` (1e-6) and
    #: ``layernormbf16`` (1e-5); our encoder trained with the 1e-6 default, while timm's
    #: ``vit_base_patch16_dinov3`` hard-codes 1e-5 to match Meta's released checkpoints. Building
    #: with timm's value puts the wrong epsilon in all 25 LayerNorms and shifts every block.
    norm_eps: float | None = None

    @property
    def replicates_channels(self) -> bool:
        return self.in_chans == 3


@dataclass(frozen=True)
class ModelSpec:
    """One released organelle model = encoder + neck + decoder + adaptation + inference contract."""

    model_id: str  # "quantem/mito"
    organelle: str  # "mito" | "er" | "nucleus" | "ld"
    arm_name: str  # the Fig S5 arm, e.g. "F4v2_omni_cem"
    encoder: EncoderSpec

    neck: str  # "naive_1x1" | "resnet34_detail"
    decoder: str  # "affinity_mws" | "upernet" | "dpt"
    neck_out_channels: int = 256
    feature_layers: str = "last4"
    apply_encoder_norm: bool = True

    adapt: str = "frozen"  # "lora" | "last_n" | "full"
    adapt_params: dict = field(default_factory=dict)

    tile_size: int = 512  # rounded up to a patch multiple at runtime (512 -> 518 for p14)
    overlap: float = 0.25
    fg_threshold: float = 0.5
    instance_min_size: int = 16
    num_classes: int = 2
    task: str = "instance"  # "instance" | "semantic"

    #: nm/px the model was trained at. ``None`` = native resolution (ER).
    canonical_nm: float | None = None

    #: Post-processing defaults.
    close_radius: int = 3
    min_area: int = 60

    #: Weight artifacts this model needs, in load order. A trunk may be shared with sibling
    #: models; a model with no trunk is self-contained (QuantEM ER replaces the whole encoder).
    trunk_artifact: str | None = None
    model_artifact: str = ""

    def effective_tile(self) -> int:
        """Tile size rounded up to a whole number of patches. 512 -> 512 (p16) | 518 (p14)."""
        p = self.encoder.patch_size
        return ((self.tile_size + p - 1) // p) * p

    def stride(self) -> int:
        """Sliding-window step. overlap 0.25 -> 384 (p16) | 389 (p14)."""
        return max(1, round(self.effective_tile() * (1.0 - self.overlap)))

    @property
    def family(self) -> str:
        return self.encoder.family

    @property
    def resamples(self) -> bool:
        return self.canonical_nm is not None
