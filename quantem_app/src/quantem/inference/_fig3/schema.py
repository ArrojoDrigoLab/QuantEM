"""The subset of the training config that inference has to reproduce.

The full training config has 8 nested sections and ~130 fields covering the data
pipeline, augmentation, optimiser, losses, evaluation metrics and the style
conditioning study. Exactly four of them change what a forward pass computes, and
only those are kept:

* which encoder blocks are tapped (``encoder.feature_layers``),
* whether the encoder's final LayerNorm is applied per tap
  (``encoder.apply_encoder_norm``),
* how the encoder was adapted (``encoder.adapt`` / ``adapt_params``), which
  decides what extra parameters exist for the head to load into,
* the neck and decoder type and params, and ``data.num_classes``.

Unknown keys are dropped, exactly as the research loader does, so the released
``resolved_config.yaml`` files load here verbatim -- they are read as shipped,
never edited. Note ``encoder.tile_size`` is deliberately **not** read: those
YAMLs say 512 for both families because 512 is the requested *base* tile, while
the OmniEM ViT-L/14 actually runs at 518 (``round_to_patch(512, 14)``). The
authoritative tile is :data:`quantem.registry.manifest.ARCHITECTURE`, surfaced
as ``ModelSpec.tile_size``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _known(cls: type, raw: dict | None) -> dict:
    """Keep only keys that are declared fields of ``cls`` (drop the rest)."""
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in (raw or {}).items() if k in allowed}


@dataclass
class EncoderSpec:
    """The base encoder and how its feature taps are read."""

    run_dir: str | None = None
    checkpoint_step: int | None = None
    feature_layers: Any = "last4"  # "last4" | "last1" | explicit block indices
    apply_encoder_norm: bool = True  # apply the encoder's final LayerNorm per tap
    adapt: str = "frozen"  # frozen | lora | lora_ln | last_n | full
    adapt_params: dict = field(default_factory=dict)

    def resolved_layers(self, depth: int) -> list[int]:
        """Concrete 0-based block indices for this architecture depth.

        ``last4`` on a ViT-B (depth 12) is ``[8, 9, 10, 11]``; on a ViT-L
        (depth 24) it is ``[20, 21, 22, 23]``. The neck concatenates the taps in
        this order, so the order is load-bearing.
        """
        fl = self.feature_layers
        if isinstance(fl, (list, tuple)):
            return [int(i) for i in fl]
        if fl == "last1":
            return [depth - 1]
        if fl == "last4":
            return [depth - 4, depth - 3, depth - 2, depth - 1]
        raise ValueError(f"feature_layers must be 'last1', 'last4', or a list; got {fl!r}")

    @classmethod
    def from_dict(cls, d: dict | None) -> EncoderSpec:
        return cls(**_known(cls, d))


@dataclass
class NeckSpec:
    type: str = "naive_1x1"
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> NeckSpec:
        return cls(**_known(cls, d))


@dataclass
class DecoderSpec:
    type: str = "upernet"
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> DecoderSpec:
        return cls(**_known(cls, d))


@dataclass
class HeadConfig:
    """Everything from a ``resolved_config.yaml`` that shapes the module graph."""

    name: str = ""
    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    neck: NeckSpec = field(default_factory=NeckSpec)
    decoder: DecoderSpec = field(default_factory=DecoderSpec)
    num_classes: int = 2
    config_path: str | None = None

    @property
    def neck_out_channels(self) -> int:
        return int((self.neck.params or {}).get("out_channels", 256))

    @classmethod
    def from_dict(cls, raw: dict | None) -> HeadConfig:
        raw = raw or {}
        data = raw.get("data") or {}
        return cls(
            name=str(raw.get("name", "")),
            encoder=EncoderSpec.from_dict(raw.get("encoder")),
            neck=NeckSpec.from_dict(raw.get("neck")),
            decoder=DecoderSpec.from_dict(raw.get("decoder")),
            num_classes=int(data.get("num_classes", 2)),
        )


def load_head_config(path: str | Path | None) -> HeadConfig:
    """Load a released ``resolved_config.yaml``. ``None`` yields the defaults."""
    if path is None:
        return HeadConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = HeadConfig.from_dict(raw)
    cfg.config_path = str(path)
    return cfg
