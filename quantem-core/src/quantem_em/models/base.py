"""Shared building blocks and the assembled ``SegModel``.

Ported from ``training_and_analysis/segmentation_training/models/base.py`` (which is byte-identical
in executable code to the ``fig3`` tree that produced the published numbers). Trimmed to the
inference path: no conditioner, no ``require()`` heavy-dependency helper, no training hooks.

Contract, preserved exactly so necks and decoders stay swappable:

* Encoder taps -> list of ``[B, embed_dim, H/p, W/p]`` patch grids, one per selected block, ascending.
* Neck ``forward(feats, image) -> list`` of one map per stride in ``STRIDES`` (4, 8, 16, 32).
* Decoder ``forward(pyramid, out_hw) -> [B, num_classes, H, W]`` dense logits. Instance decoders put
  their extra tensors on ``self.aux_logits`` and still return semantic logits here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spec import EncoderSpec

STRIDES: tuple[int, ...] = (4, 8, 16, 32)


class ConvGNAct(nn.Sequential):
    """conv -> GroupNorm -> GELU. GroupNorm, never BatchNorm: these run at batch size 1."""

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 32):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
            nn.GroupNorm(min(groups, cout), cout),
            nn.GELU(),
        )


def resize_to(x: torch.Tensor, hw) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    return F.interpolate(x, size=tuple(hw), mode="bilinear", align_corners=False)


class Encoder(nn.Module):
    """Uniform tap interface over a timm backbone, for either family.

    Mirrors the research harness's ``FrozenEncoder.features(x, layers)`` so the necks and decoders
    below are unchanged from the versions that produced the published numbers. The differences from
    that class are all removals: no MAE branch, no SAM3 branch, no ``dinov3`` import.
    """

    def __init__(self, backbone: nn.Module, spec: EncoderSpec, apply_encoder_norm: bool = True):
        super().__init__()
        self.backbone = backbone
        self.spec = spec
        self.apply_encoder_norm = bool(apply_encoder_norm)
        self.patch_size = spec.patch_size
        self.embedding_dim = spec.embed_dim
        self.depth = spec.depth
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    # -- input contract ------------------------------------------------------
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt a normalised single-channel tile to what this backbone expects.

        QuantEM is natively 1-channel and the dataset already applied its EM statistics, so this is
        a no-op. OmniEM is a 3-channel DINOv2: the dataset hands it a raw ``[0, 1]`` tile
        (``dataset_mean=0``, ``dataset_std=1``) and the channel replication plus the EM-specific
        per-channel normalisation happen here — matching ``external_vit.preprocess``.
        """
        s = self.spec
        if s.in_chans != 1 and x.shape[1] == 1:
            x = x.repeat(1, s.in_chans, 1, 1)
        if s.encoder_mean is not None:
            x = (x - s.encoder_mean) / s.encoder_std
        return x

    def features(
        self, x: torch.Tensor, layers: list[int], grad: bool = False
    ) -> list[torch.Tensor]:
        """Tapped feature grids, ascending by block index.

        ``grad=True`` keeps the graph so encoder-side adapters (LoRA) receive gradients; the base
        weights stay frozen regardless.
        """
        if grad:
            return self._extract(x, layers)
        with torch.no_grad():
            return self._extract(x, layers)

    def _extract(self, x: torch.Tensor, layers: list[int]) -> list[torch.Tensor]:
        layers = sorted(layers)
        x = self.preprocess(x)
        feats = self.backbone.forward_intermediates(
            x,
            indices=layers,
            norm=self.apply_encoder_norm,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        return list(feats)

    def resolved_layers(self, feature_layers: str = "last4") -> list[int]:
        """``"last4"`` -> the last four block indices. All eight released models use last4."""
        if feature_layers != "last4":
            raise ValueError(
                f"unsupported feature_layers {feature_layers!r} (released models use 'last4')"
            )
        return list(range(self.depth - 4, self.depth))


class SegModel(nn.Module):
    """Encoder + neck + decoder. The single module the inference loop drives."""

    def __init__(
        self,
        encoder: Encoder,
        neck: nn.Module,
        decoder: nn.Module,
        layers: list[int],
        encoder_trainable: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.layers = list(layers)
        self.encoder_trainable = bool(encoder_trainable)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.features(image, self.layers, grad=self.encoder_trainable)
        pyramid = self.neck(feats, image)
        return self.decoder(pyramid, out_hw=image.shape[-2:])

    @property
    def aux_logits(self):
        """Instance decoders stash affinities here during ``forward``.

        Written as a side effect, so a ``SegModel`` is **not thread-safe**: use one instance per
        worker.
        """
        return getattr(self.decoder, "aux_logits", []) or []

    def head_parameters(self):
        """Neck + decoder only — what head-only fine-tuning trains."""
        yield from self.neck.parameters()
        yield from self.decoder.parameters()
