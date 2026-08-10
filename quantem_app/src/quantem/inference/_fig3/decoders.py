"""Decoders: neck pyramid -> dense semantic logits.

Only the three decoders the released packs use are included, with the parameter
names the checkpoints expect. Not included: ``nnunet_convnext_unet``, ``pspnet``,
``deeplabv3plus``, ``unet`` and ``panoptic_deeplab`` (no released pack selects
them), the detectron2-backed heavy arms, and the eval-only instance
post-processing (``mutex_watershed_postproc``, ``panoptic_instance_postproc``),
which needs ``elf``/``affogato``.

Every decoder consumes the neck pyramid -- ``[B, in_channels, H/s, W/s]``, one
map per stride in ``base.STRIDES``, ascending -- and returns
``[B, num_classes, H, W]`` at ``out_hw``. The instance decoder
(``affinity_mws``) honours that same semantic contract and puts its affinities
on ``self.aux_logits``, which is what lets the engine reduce all eight packs to
a foreground probability the same way.

Pack coverage:

* ``affinity_mws`` -- the six mito / LD / nucleus packs.
* ``upernet`` -- ``quantem:er``.
* ``dpt`` -- ``omniem:er``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ConvGNAct, resize_to
from .schema import DecoderSpec


class _PPM(nn.Module):
    """Pyramid Pooling Module: adaptive-pool the coarsest map at several bins, fuse."""

    def __init__(self, in_channels: int, out_channels: int, bins: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.stages = nn.ModuleList(ConvGNAct(in_channels, out_channels, k=1) for _ in bins)
        self.bins = tuple(bins)
        self.project = ConvGNAct(in_channels + len(bins) * out_channels, out_channels, k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        outs = [x]
        for b, stage in zip(self.bins, self.stages, strict=True):
            pooled = F.adaptive_avg_pool2d(x, output_size=b)
            outs.append(resize_to(stage(pooled), hw))
        return self.project(torch.cat(outs, dim=1))


class UPerNet(nn.Module):
    """PPM on the coarsest level + FPN over the pyramid -> fuse -> seg head."""

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256) -> None:
        super().__init__()
        self.strides = tuple(strides)
        n = len(self.strides)
        self.ppm = _PPM(in_channels, channels)
        # Laterals on all but the coarsest level; the coarsest comes from the PPM.
        self.laterals = nn.ModuleList(ConvGNAct(in_channels, channels, k=1) for _ in range(n - 1))
        self.fpn_convs = nn.ModuleList(ConvGNAct(channels, channels, k=3) for _ in range(n))
        self.fuse = ConvGNAct(n * channels, channels, k=3)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw: tuple[int, int]) -> torch.Tensor:
        self.aux_logits = []
        laterals = [self.laterals[i](pyramid[i]) for i in range(len(pyramid) - 1)]
        laterals.append(self.ppm(pyramid[-1]))
        # Top-down pathway, coarse to fine.
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + resize_to(laterals[i], laterals[i - 1].shape[-2:])
        feats = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        target_hw = feats[0].shape[-2:]
        feats = [resize_to(f, target_hw) for f in feats]
        fused = self.fuse(torch.cat(feats, dim=1))
        logits = self.classifier(fused)
        return resize_to(logits, out_hw)


class _ResidualConvUnit(nn.Module):
    """DPT RefineNet-style residual conv unit (GN + GELU pre-activation)."""

    def __init__(self, channels: int) -> None:
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
    """DPT-style dense decoder: reassemble pyramid levels + progressive fusion."""

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256) -> None:
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

    def forward(self, pyramid: list[torch.Tensor], out_hw: tuple[int, int]) -> torch.Tensor:
        self.aux_logits = []
        feats = [self.reassemble[i](pyramid[i]) for i in range(len(pyramid))]
        path = None
        for i in range(len(feats) - 1, -1, -1):  # coarse -> fine
            cur = self.rcu_in[i](feats[i])
            if path is not None:
                cur = cur + resize_to(path, cur.shape[-2:])
            path = self.rcu_out[i](cur)
        logits = self.head(path)
        return resize_to(logits, out_hw)


class _SharedDecoderTrunk(nn.Module):
    """Fuse the pyramid to the finest resolution -> the shared instance-head map."""

    def __init__(self, in_channels: int, strides: tuple, channels: int = 256) -> None:
        super().__init__()
        n = len(strides)
        self.proj = nn.ModuleList(ConvGNAct(in_channels, channels, k=1) for _ in range(n))
        self.fuse = ConvGNAct(n * channels, channels, k=3)
        self.refine = ConvGNAct(channels, channels, k=3)

    def forward(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        target_hw = pyramid[0].shape[-2:]
        feats = [resize_to(self.proj[i](pyramid[i]), target_hw) for i in range(len(pyramid))]
        fused = self.fuse(torch.cat(feats, dim=1))
        return self.refine(fused)


class AffinityMWS(nn.Module):
    """Affinity head (short attractive + long repulsive) plus a semantic logit.

    ``self.aux_logits = [affinities]``, ``[B, n_offsets, H, W]`` in ``[0, 1]``;
    channel ``c`` is the affinity along ``self.offsets[c]``. Affinities are
    position-invariant scalar edge weights, so they Hann-blend across sliding
    window seams correctly -- a position-dependent auxiliary output would not.

    The returned tensor is the **semantic** logit, which is what the app
    thresholds. Resolving the affinities into instances needs a mutex watershed,
    which is not vendored: splitting two touching mitochondria is still the
    connected-components limitation described in ``../README.md``.
    """

    # (dy, dx). The first N_SHORT are the nearest-neighbour attractive edges;
    # the rest are long-range repulsive mutex edges.
    DEFAULT_OFFSETS = (
        (0, 1), (1, 0), (1, 1), (1, -1),
        (0, 3), (3, 0), (0, 9), (9, 0), (9, 9), (9, -9),
    )
    N_SHORT = 4

    def __init__(
        self,
        in_channels: int,
        strides: tuple,
        num_classes: int,
        channels: int = 256,
        offsets: tuple | None = None,
    ) -> None:
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

    def forward(self, pyramid: list[torch.Tensor], out_hw: tuple[int, int]) -> torch.Tensor:
        sem = self.sem_trunk(pyramid)
        sem_logits = resize_to(self.sem_head(sem), out_hw)
        a = self.aff_block(self.aff_proj(pyramid[0]))
        aff = torch.sigmoid(self.aff_head(a))
        aff = resize_to(aff, out_hw)
        self.aux_logits = [aff]
        return sem_logits


def _b_upernet(spec: DecoderSpec, in_channels: int, strides: tuple, num_classes: int) -> nn.Module:
    p = dict(spec.params)
    return UPerNet(in_channels, strides, num_classes, channels=int(p.get("channels", 256)))


def _b_dpt(spec: DecoderSpec, in_channels: int, strides: tuple, num_classes: int) -> nn.Module:
    p = dict(spec.params)
    return DPT(in_channels, strides, num_classes, channels=int(p.get("channels", 256)))


def _b_affinity(spec: DecoderSpec, in_channels: int, strides: tuple, num_classes: int) -> nn.Module:
    p = dict(spec.params)
    return AffinityMWS(
        in_channels,
        strides,
        num_classes,
        channels=int(p.get("channels", 256)),
        offsets=p.get("offsets"),
    )


DECODERS = {
    "upernet": _b_upernet,
    "dpt": _b_dpt,
    "affinity_mws": _b_affinity,
}


def build_decoder(
    spec: DecoderSpec, in_channels: int, strides: tuple, num_classes: int
) -> nn.Module:
    """Dispatch ``spec.type``; unknown types name the ones that ship."""
    builder = DECODERS.get(spec.type)
    if builder is None:
        raise ValueError(
            f"Unknown decoder type {spec.type!r}. QuantEM ships {sorted(DECODERS)}; the "
            "remaining decoder arms are not included (no released pack uses them)."
        )
    return builder(spec, in_channels, tuple(strides), num_classes)
