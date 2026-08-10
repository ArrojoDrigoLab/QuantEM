"""Base encoders. Both families load through timm; no Meta code is redistributed."""

from __future__ import annotations

from ...spec import EncoderSpec


def build_encoder(spec: EncoderSpec, state_dict=None, *, img_size: int = 512, strict: bool = True):
    """Dispatch to the right builder for ``spec.family``."""
    if spec.family == "quantem":
        from .quantem_vit import build_quantem_encoder

        return build_quantem_encoder(spec, state_dict, img_size=img_size, strict=strict)
    if spec.family == "omniem":
        from .omniem_vit import build_omniem_encoder

        return build_omniem_encoder(spec, state_dict, img_size=img_size, strict=strict)
    raise ValueError(f"unknown encoder family {spec.family!r}")
