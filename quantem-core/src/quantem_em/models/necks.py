"""Necks: encoder taps (single stride) -> feature pyramid at STRIDES (4, 8, 16, 32).

Ported verbatim from ``segmentation_training/models/necks.py``, keeping only the two arms the
released models use. The ``vit_adapter`` / ``vit_comer`` arms and their compiled deformable-attention
dependency are dropped.

The ResNet-34 detail stem is hand-rolled — **no torchvision, no pretrained weights** — so this adds
no dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import STRIDES, ConvGNAct, resize_to


def _tap_fuse(embed_dim: int, n_taps: int, out_channels: int) -> nn.Module:
    """Concat the taps on channels -> 1x1 -> GN -> GELU -> out_channels."""
    return ConvGNAct(embed_dim * n_taps, out_channels, k=1)


def _pyramid_hw(out_hw_16: tuple[int, int]) -> list[tuple[int, int]]:
    """Sizes for every stride in STRIDES, given the encoder's tap grid."""
    h16, w16 = out_hw_16
    sizes = []
    for s in STRIDES:
        f = 16 / s
        sizes.append((max(1, round(h16 * f)), max(1, round(w16 * f))))
    return sizes


class Naive1x1Neck(nn.Module):
    """Concat taps -> 1x1 -> one map, resampled to each pyramid stride.

    Used by mitochondria, nucleus and lipid droplets in both families.
    """

    def __init__(self, embed_dim: int, n_taps: int, out_channels: int):
        super().__init__()
        self.out_channels = int(out_channels)
        self.strides = STRIDES
        self.fuse = _tap_fuse(embed_dim, n_taps, out_channels)

    def forward(
        self, feats: list[torch.Tensor], image: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        x = self.fuse(torch.cat(feats, dim=1))
        return [resize_to(x, hw) for hw in _pyramid_hw(tuple(x.shape[-2:]))]


def _norm2d(kind: str, ch: int) -> nn.Module:
    if kind == "instancenorm":
        return nn.InstanceNorm2d(ch, affine=True)
    return nn.GroupNorm(min(32, ch), ch)


class _BasicBlock(nn.Module):
    """ResNet basic block with GroupNorm/InstanceNorm and a projection shortcut."""

    def __init__(self, cin: int, cout: int, stride: int, norm: str):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.n1 = _norm2d(norm, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.n2 = _norm2d(norm, cout)
        self.act = nn.ReLU(inplace=True)
        self.down = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), _norm2d(norm, cout))
            if (stride != 1 or cin != cout)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idn = x if self.down is None else self.down(x)
        y = self.act(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        return self.act(y + idn)


class _ResNet34Stem(nn.Module):
    """ResNet-34-style stem on the raw 1-channel image, emitting strides 4, 8, 16, 32.

    Basic-block counts 3/4/6/3. conv7x7 s2 -> maxpool s2 -> layer1..4.
    """

    def __init__(self, width: int, out_channels: int, norm: str):
        super().__init__()
        w = int(width)
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 7, stride=2, padding=3, bias=False),
            _norm2d(norm, w),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make(w, w, 3, stride=1, norm=norm)
        self.layer2 = self._make(w, w * 2, 4, stride=2, norm=norm)
        self.layer3 = self._make(w * 2, w * 4, 6, stride=2, norm=norm)
        self.layer4 = self._make(w * 4, w * 8, 3, stride=2, norm=norm)
        self.proj = nn.ModuleList(
            [
                nn.Conv2d(w, out_channels, 1),
                nn.Conv2d(w * 2, out_channels, 1),
                nn.Conv2d(w * 4, out_channels, 1),
                nn.Conv2d(w * 8, out_channels, 1),
            ]
        )

    @staticmethod
    def _make(cin: int, cout: int, n: int, stride: int, norm: str) -> nn.Sequential:
        blocks = [_BasicBlock(cin, cout, stride=stride, norm=norm)]
        for _ in range(1, n):
            blocks.append(_BasicBlock(cout, cout, stride=1, norm=norm))
        return nn.Sequential(*blocks)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = self.pool(self.stem(image))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [self.proj[0](c1), self.proj[1](c2), self.proj[2](c3), self.proj[3](c4)]


class ResNet34DetailNeck(nn.Module):
    """Encoder taps fused with a ResNet-34 detail branch run on the raw image.

    Used by both ER models — the thin, high-frequency structure the taps alone cannot resolve.
    """

    def __init__(
        self,
        embed_dim: int,
        n_taps: int,
        out_channels: int,
        norm: str = "groupnorm",
        width: int = 64,
        fuse: str = "concat",
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.strides = STRIDES
        self.fuse_mode = str(fuse)
        self.tap_fuse = _tap_fuse(embed_dim, n_taps, out_channels)
        self.detail = _ResNet34Stem(width=width, out_channels=out_channels, norm=norm)
        self.mix = (
            nn.ModuleList([ConvGNAct(out_channels * 2, out_channels, k=1) for _ in STRIDES])
            if self.fuse_mode == "concat"
            else None
        )

    def forward(
        self, feats: list[torch.Tensor], image: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        if image is None:
            raise ValueError("resnet34_detail neck requires the raw `image` for its detail branch.")
        sem16 = self.tap_fuse(torch.cat(feats, dim=1))
        # The detail stem is defined on a single channel; OmniEM's 3-channel replication happens
        # inside the encoder, so the raw image reaching here is still 1-channel.
        if image.shape[1] != 1:
            image = image[:, :1]
        detail = self.detail(image)
        pyramid = []
        for lvl, d in enumerate(detail):
            s = resize_to(sem16, tuple(d.shape[-2:]))
            pyramid.append(
                self.mix[lvl](torch.cat([s, d], dim=1)) if self.mix is not None else s + d
            )
        return pyramid


def build_neck(kind: str, embed_dim: int, n_taps: int, out_channels: int = 256) -> nn.Module:
    if kind == "naive_1x1":
        return Naive1x1Neck(embed_dim, n_taps, out_channels)
    if kind == "resnet34_detail":
        return ResNet34DetailNeck(embed_dim, n_taps, out_channels)
    raise ValueError(f"unknown neck {kind!r}; released models use 'naive_1x1' or 'resnet34_detail'")
