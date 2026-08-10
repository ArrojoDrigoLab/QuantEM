"""Segmentation experiment configuration: the dataclass-over-YAML schema and its loader."""

from __future__ import annotations

from .schema import (  # noqa: F401
    DataSpec,
    DecoderSpec,
    EncoderSpec,
    EvalSpec,
    SegConfig,
    LossSpec,
    LossTerm,
    NeckSpec,
    OptimSpec,
    load_seg_config,
)

__all__ = [
    "SegConfig",
    "load_seg_config",
    "EncoderSpec",
    "NeckSpec",
    "DecoderSpec",
    "LossSpec",
    "LossTerm",
    "DataSpec",
    "OptimSpec",
    "EvalSpec",
]
