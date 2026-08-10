"""Neck registry — frozen encoder taps (all at stride 16) -> multi-scale pyramid at STRIDES.

Every neck honours the contract in ``models/base.py``:

  * ``forward(feats: list[Tensor], image: Tensor|None) -> list[Tensor]`` returns one map per stride in
    ``STRIDES`` (4, 8, 16, 32), each ``[B, out_channels, H/s, W/s]``.
  * ``.out_channels`` (int) and ``.strides`` (tuple == STRIDES) are exposed as attributes so the
    decoder registry can wire itself blind to which neck built the pyramid.

The taps come in at a single stride (16) so a neck's job is (a) fuse the taps into a common channel
width and (b) synthesise the finer/coarser strides — either by pure resampling (``naive_1x1``) or by
folding in a raw-image detail branch that carries stride 2/4/8 high-frequency content
(``resnet34_detail``).

Only torch and this package are imported at module level; torchvision is not imported, because the
ResNet-34 detail stem is hand-rolled, so every neck builds and runs on a CPU-only machine.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config.schema import NeckSpec
from .base import STRIDES, ConvGNAct, resize_to


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _tap_fuse(embed_dim: int, n_taps: int, out_channels: int) -> nn.Module:
    """Concat the ``n_taps`` stride-16 taps on channels -> 1x1 conv -> GN -> GELU -> out_channels."""
    return ConvGNAct(embed_dim * n_taps, out_channels, k=1)


def _pyramid_hw(out_hw_16: tuple[int, int]) -> list[tuple[int, int]]:
    """Given the stride-16 grid (H/16, W/16), the (H/s, W/s) sizes for every stride in STRIDES."""
    h16, w16 = out_hw_16
    sizes = []
    for s in STRIDES:
        f = 16 / s  # >1 upsample (finer), <1 downsample (coarser)
        sizes.append((max(1, round(h16 * f)), max(1, round(w16 * f))))
    return sizes


# --------------------------------------------------------------------------- #
# 1. naive_1x1  (baseline)
# --------------------------------------------------------------------------- #
class Naive1x1Neck(nn.Module):
    """Concat taps -> 1x1 -> single stride-16 map, then resample to each pyramid stride.

    No raw-image branch: the finer levels are pure bilinear upsamples of the stride-16 features (they
    add resolution to the decoder's sampling grid but no new high-frequency content — that is exactly
    the baseline this ablation contrasts ``resnet34_detail`` against).
    """

    def __init__(self, embed_dim: int, n_taps: int, out_channels: int):
        super().__init__()
        self.out_channels = int(out_channels)
        self.strides = STRIDES
        self.fuse = _tap_fuse(embed_dim, n_taps, out_channels)

    def forward(self, feats: list[torch.Tensor], image: torch.Tensor | None = None) -> list[torch.Tensor]:
        x = torch.cat(feats, dim=1)          # [B, embed_dim*n_taps, H/16, W/16]
        x = self.fuse(x)                      # [B, C, H/16, W/16]
        sizes = _pyramid_hw(tuple(x.shape[-2:]))
        return [resize_to(x, hw) for hw in sizes]


# --------------------------------------------------------------------------- #
# 2. resnet34_detail  (raw-image high-frequency branch fused into the pyramid)
# --------------------------------------------------------------------------- #
def _norm2d(kind: str, ch: int) -> nn.Module:
    """GroupNorm / InstanceNorm — never BatchNorm (batch-size robustness; seg runs use tiny batches)."""
    if kind == "instancenorm":
        return nn.InstanceNorm2d(ch, affine=True)
    # default: groupnorm (32 groups, clamped to channels)
    return nn.GroupNorm(min(32, ch), ch)


class _BasicBlock(nn.Module):
    """ResNet basic block (2x conv3x3) with GroupNorm/InstanceNorm and a projection shortcut."""

    def __init__(self, cin: int, cout: int, stride: int, norm: str):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.n1 = _norm2d(norm, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.n2 = _norm2d(norm, cout)
        self.act = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False), _norm2d(norm, cout)
            )
        else:
            self.down = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idn = x if self.down is None else self.down(x)
        y = self.act(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        return self.act(y + idn)


class _ResNet34Stem(nn.Module):
    """Hand-rolled ResNet-34-style stem on the 1-channel raw image, emitting detail feats at
    strides 4, 8, 16, 32 (matching STRIDES). No torchvision, no pretrained weights, GN/IN only.

    Layout (34-depth basic-block counts 3/4/6/3):
      conv7x7 s2 (->s2) -> maxpool s2 (->s4) -> layer1 (s4) -> layer2 (s8) -> layer3 (s16) -> layer4 (s32).
    Returns the layer1..4 outputs, i.e. the STRIDES pyramid, each projected to ``out_channels``.
    """

    def __init__(self, width: int, out_channels: int, norm: str):
        super().__init__()
        w = int(width)
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 7, stride=2, padding=3, bias=False),  # s2
            _norm2d(norm, w),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)          # s4
        self.layer1 = self._make(w, w, 3, stride=1, norm=norm)    # s4
        self.layer2 = self._make(w, w * 2, 4, stride=2, norm=norm)  # s8
        self.layer3 = self._make(w * 2, w * 4, 6, stride=2, norm=norm)  # s16
        self.layer4 = self._make(w * 4, w * 8, 3, stride=2, norm=norm)  # s32
        # per-stride 1x1 projections to the neck channel width (order matches STRIDES = 4,8,16,32)
        self.proj = nn.ModuleList([
            nn.Conv2d(w, out_channels, 1),
            nn.Conv2d(w * 2, out_channels, 1),
            nn.Conv2d(w * 4, out_channels, 1),
            nn.Conv2d(w * 8, out_channels, 1),
        ])

    @staticmethod
    def _make(cin: int, cout: int, n: int, stride: int, norm: str) -> nn.Sequential:
        blocks = [_BasicBlock(cin, cout, stride=stride, norm=norm)]
        for _ in range(1, n):
            blocks.append(_BasicBlock(cout, cout, stride=1, norm=norm))
        return nn.Sequential(*blocks)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = self.pool(self.stem(image))
        c1 = self.layer1(x)   # s4
        c2 = self.layer2(c1)  # s8
        c3 = self.layer3(c2)  # s16
        c4 = self.layer4(c3)  # s32
        return [self.proj[0](c1), self.proj[1](c2), self.proj[2](c3), self.proj[3](c4)]


class ResNet34DetailNeck(nn.Module):
    """Fuse the stride-16 encoder taps with a ResNet-34 detail branch run on the raw image.

    The taps are projected to ``out_channels`` at stride 16 and resampled to every pyramid stride
    (the semantic content). The detail stem supplies genuine stride-2/4/8/16-derived high-frequency
    features at strides 4/8/16/32, fused per level by ``concat+1x1`` (default) or elementwise ``add``.
    """

    def __init__(self, embed_dim: int, n_taps: int, out_channels: int,
                 norm: str = "groupnorm", width: int = 64, fuse: str = "concat"):
        super().__init__()
        self.out_channels = int(out_channels)
        self.strides = STRIDES
        self.fuse_mode = str(fuse)
        self.tap_fuse = _tap_fuse(embed_dim, n_taps, out_channels)
        self.detail = _ResNet34Stem(width=width, out_channels=out_channels, norm=norm)
        if self.fuse_mode == "concat":
            self.mix = nn.ModuleList([
                ConvGNAct(out_channels * 2, out_channels, k=1) for _ in STRIDES
            ])
        else:  # "add"
            self.mix = None

    def forward(self, feats: list[torch.Tensor], image: torch.Tensor | None = None) -> list[torch.Tensor]:
        if image is None:
            raise ValueError("resnet34_detail neck requires the raw `image` (its detail branch).")
        sem16 = self.tap_fuse(torch.cat(feats, dim=1))       # [B, C, H/16, W/16]
        detail = self.detail(image)                           # list per STRIDES, [B, C, H/s, W/s]
        pyramid = []
        for lvl, d in enumerate(detail):
            s = resize_to(sem16, tuple(d.shape[-2:]))         # broadcast semantic map to this stride
            if self.fuse_mode == "concat":
                pyramid.append(self.mix[lvl](torch.cat([s, d], dim=1)))
            else:
                pyramid.append(s + d)
        return pyramid


# --------------------------------------------------------------------------- #
# Registry + factory
# --------------------------------------------------------------------------- #
def _build_naive_1x1(spec: NeckSpec, embed_dim, n_taps, patch_size, out_channels):
    return Naive1x1Neck(embed_dim=embed_dim, n_taps=n_taps, out_channels=out_channels)


def _build_resnet34_detail(spec: NeckSpec, embed_dim, n_taps, patch_size, out_channels):
    p = spec.params or {}
    return ResNet34DetailNeck(
        embed_dim=embed_dim, n_taps=n_taps, out_channels=out_channels,
        norm=str(p.get("norm", "groupnorm")),
        width=int(p.get("width", 64)),
        fuse=str(p.get("fuse", "concat")),
    )


NECKS: dict[str, object] = {
    "naive_1x1": _build_naive_1x1,
    "resnet34_detail": _build_resnet34_detail,
}


def build_neck(spec: NeckSpec, embed_dim: int, n_taps: int, patch_size: int,
               out_channels: int = 256) -> nn.Module:
    """Dispatch ``spec.type`` through ``NECKS``; unknown type -> ValueError listing valid keys."""
    builder = NECKS.get(spec.type)
    if builder is None:
        raise ValueError(
            f"Unknown neck type {spec.type!r}. Valid: {sorted(NECKS)}"
        )
    return builder(spec, embed_dim, n_taps, patch_size, out_channels)
