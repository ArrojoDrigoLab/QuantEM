"""Decoders: neck pyramid -> dense semantic logits ``[B, num_classes, H, W]``.

Ported verbatim from ``segmentation_training/models/decoders.py``, keeping only the three arms the
released models use. The other six dense decoders, the panoptic-deeplab arm, and the detectron2-backed
Mask2Former / MaskDINO wrappers are dropped along with their dependencies.

Instance decoders honour the same semantic-logit return contract and put their extra tensors on
``self.aux_logits``, so post-processing and metrics work uniformly across arms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ConvGNAct, resize_to


class _PPM(nn.Module):
    """Pyramid Pooling Module: adaptive-pool the coarsest map at several bin sizes, fuse."""

    def __init__(self, in_channels: int, out_channels: int, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(ConvGNAct(in_channels, out_channels, k=1) for _ in bins)
        self.bins = tuple(bins)
        self.project = ConvGNAct(in_channels + len(bins) * out_channels, out_channels, k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        outs = [x]
        for b, stage in zip(self.bins, self.stages, strict=True):
            outs.append(resize_to(stage(F.adaptive_avg_pool2d(x, output_size=b)), hw))
        return self.project(torch.cat(outs, dim=1))


class UPerNet(nn.Module):
    """PPM on the coarsest level + FPN over the pyramid -> fuse -> seg head. (QuantEM ER.)"""

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256):
        super().__init__()
        self.strides = tuple(strides)
        n = len(self.strides)
        self.ppm = _PPM(in_channels, channels)
        self.laterals = nn.ModuleList(ConvGNAct(in_channels, channels, k=1) for _ in range(n - 1))
        self.fpn_convs = nn.ModuleList(ConvGNAct(channels, channels, k=3) for _ in range(n))
        self.fuse = ConvGNAct(n * channels, channels, k=3)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        self.aux_logits = []
        laterals = [self.laterals[i](pyramid[i]) for i in range(len(pyramid) - 1)]
        laterals.append(self.ppm(pyramid[-1]))
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + resize_to(laterals[i], laterals[i - 1].shape[-2:])
        feats = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        target_hw = feats[0].shape[-2:]
        fused = self.fuse(torch.cat([resize_to(f, target_hw) for f in feats], dim=1))
        return resize_to(self.classifier(fused), out_hw)


class _ResidualConvUnit(nn.Module):
    """DPT RefineNet-style residual conv unit (GN + GELU pre-activation)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DPT(nn.Module):
    """DPT-style reassemble + progressive RefineNet fusion. (OmniEM ER.)"""

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256):
        super().__init__()
        self.strides = tuple(strides)
        n = len(self.strides)
        self.reassemble = nn.ModuleList(ConvGNAct(in_channels, channels, k=1) for _ in range(n))
        self.rcu_in = nn.ModuleList(_ResidualConvUnit(channels) for _ in range(n))
        self.rcu_out = nn.ModuleList(_ResidualConvUnit(channels) for _ in range(n))
        self.head = nn.Sequential(
            ConvGNAct(channels, channels, k=3),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        self.aux_logits = []
        feats = [self.reassemble[i](pyramid[i]) for i in range(len(pyramid))]
        path = None
        for i in range(len(feats) - 1, -1, -1):
            cur = self.rcu_in[i](feats[i])
            if path is not None:
                cur = cur + resize_to(path, cur.shape[-2:])
            path = self.rcu_out[i](cur)
        return resize_to(self.head(path), out_hw)


class _SharedDecoderTrunk(nn.Module):
    """Fuse the pyramid to the finest resolution -> a shared feature map for instance heads."""

    def __init__(self, in_channels: int, strides: tuple, channels: int = 256):
        super().__init__()
        n = len(strides)
        self.proj = nn.ModuleList(ConvGNAct(in_channels, channels, k=1) for _ in range(n))
        self.fuse = ConvGNAct(n * channels, channels, k=3)
        self.refine = ConvGNAct(channels, channels, k=3)

    def forward(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        target_hw = pyramid[0].shape[-2:]
        feats = [resize_to(self.proj[i](pyramid[i]), target_hw) for i in range(len(pyramid))]
        return self.refine(self.fuse(torch.cat(feats, dim=1)))


class AffinityMWS(nn.Module):
    """Affinity head + semantic foreground logit. (Mitochondria, nucleus, lipid droplets.)

    ``self.aux_logits = [affinities]``, ``[B, n_offsets, H, W]`` in [0, 1]. Affinities are
    position-invariant scalar edge weights, so they Hann-blend across window seams correctly.

    v1 does **not** ship mutex-watershed clustering: no published number was produced with it
    (``affogato`` was never installed on any campaign box, and the reference silently fell back to
    connected components). Connected components is what reproduces the paper, so the affinity head
    runs and its output is available, but instances come from CC.
    """

    DEFAULT_OFFSETS = (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),  # short-range attractive
        (0, 3),
        (3, 0),
        (0, 9),
        (9, 0),
        (9, 9),
        (9, -9),  # long-range repulsive
    )
    N_SHORT = 4

    def __init__(
        self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256, offsets=None
    ):
        super().__init__()
        self.strides = tuple(strides)
        self.offsets = tuple(tuple(o) for o in (offsets or self.DEFAULT_OFFSETS))
        self.n_short = self.N_SHORT
        self.sem_trunk = _SharedDecoderTrunk(in_channels, strides, channels)
        self.aff_proj = ConvGNAct(in_channels, channels, k=3)
        self.aff_block = ConvGNAct(channels, channels, k=3)
        self.aff_head = nn.Conv2d(channels, len(self.offsets), 1)
        self.sem_head = nn.Sequential(
            ConvGNAct(channels, channels, k=3), nn.Conv2d(channels, num_classes, 1)
        )
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        sem = self.sem_trunk(pyramid)
        sem_logits = resize_to(self.sem_head(sem), out_hw)
        aff = torch.sigmoid(self.aff_head(self.aff_block(self.aff_proj(pyramid[0]))))
        self.aux_logits = [resize_to(aff, out_hw)]
        return sem_logits


def build_decoder(kind: str, in_channels: int, strides: tuple, num_classes: int) -> nn.Module:
    builders = {"upernet": UPerNet, "dpt": DPT, "affinity_mws": AffinityMWS}
    try:
        cls = builders[kind]
    except KeyError:
        raise ValueError(
            f"unknown decoder {kind!r}; released models use {sorted(builders)}"
        ) from None
    return cls(in_channels, strides, num_classes)
