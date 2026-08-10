"""Necks: stride-16 encoder taps -> a feature pyramid at :data:`STRIDES`.

Only the two necks the released packs use are included, with the parameter names
the checkpoints expect. Not included: the ``conv_lora`` alias builder (no released
pack selects it) and the ``vit_adapter`` / ``vit_comer`` heavy arms, which need
mmcv and CUDA-compiled multi-scale deformable attention.

Pack coverage:

* ``naive_1x1`` -- the six mito / LD / nucleus packs.
* ``resnet34_detail`` -- ``quantem:er`` and ``omniem:er``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import STRIDES, ConvGNAct, resize_to
from .schema import NeckSpec


def _tap_fuse(embed_dim: int, n_taps: int, out_channels: int) -> nn.Module:
    """Concat the stride-16 taps on channels -> 1x1 -> GN -> GELU."""
    return ConvGNAct(embed_dim * n_taps, out_channels, k=1)


def _pyramid_hw(out_hw_16: tuple[int, int]) -> list[tuple[int, int]]:
    """``(H/s, W/s)`` for every stride in :data:`STRIDES`, given the stride-16 grid."""
    h16, w16 = out_hw_16
    sizes = []
    for s in STRIDES:
        f = 16 / s  # >1 finer, <1 coarser
        sizes.append((max(1, round(h16 * f)), max(1, round(w16 * f))))
    return sizes


class Naive1x1Neck(nn.Module):
    """Concat taps -> 1x1 -> one stride-16 map, resampled to each pyramid stride.

    No raw-image branch: the finer levels are bilinear upsamples, which give the
    decoder a finer sampling grid but no new high-frequency content. That is the
    baseline the ``resnet34_detail`` ablation is measured against.
    """

    def __init__(self, embed_dim: int, n_taps: int, out_channels: int) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.strides = STRIDES
        self.fuse = _tap_fuse(embed_dim, n_taps, out_channels)

    def forward(
        self, feats: list[torch.Tensor], image: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        x = torch.cat(feats, dim=1)  # [B, embed_dim * n_taps, H/16, W/16]
        x = self.fuse(x)             # [B, C, H/16, W/16]
        sizes = _pyramid_hw(tuple(x.shape[-2:]))
        return [resize_to(x, hw) for hw in sizes]


def _norm2d(kind: str, ch: int) -> nn.Module:
    """GroupNorm or InstanceNorm -- never BatchNorm (batch size 1 at inference)."""
    if kind == "instancenorm":
        return nn.InstanceNorm2d(ch, affine=True)
    return nn.GroupNorm(min(32, ch), ch)


class _BasicBlock(nn.Module):
    """ResNet basic block (two 3x3 convs) with a projection shortcut."""

    def __init__(self, cin: int, cout: int, stride: int, norm: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.n1 = _norm2d(norm, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.n2 = _norm2d(norm, cout)
        self.act = nn.ReLU(inplace=True)
        self.down: nn.Module | None
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
    """ResNet-34-shaped stem on the 1-channel input, emitting :data:`STRIDES`.

    Hand-rolled rather than torchvision's: no pretrained weights are involved
    (it trains from scratch with the head) and the norm layers are GN/IN, so
    reusing torchvision would only add a dependency and a BatchNorm to strip.

    Layout (34-depth basic-block counts 3/4/6/3): conv7x7 s2 -> maxpool s2 ->
    layer1 (s4) -> layer2 (s8) -> layer3 (s16) -> layer4 (s32).
    """

    def __init__(self, width: int, out_channels: int, norm: str) -> None:
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
        # Per-stride 1x1 projections to the neck width; order matches STRIDES.
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
        c1 = self.layer1(x)   # s4
        c2 = self.layer2(c1)  # s8
        c3 = self.layer3(c2)  # s16
        c4 = self.layer4(c3)  # s32
        return [self.proj[0](c1), self.proj[1](c2), self.proj[2](c3), self.proj[3](c4)]


class ResNet34DetailNeck(nn.Module):
    """Stride-16 encoder taps fused with a ResNet-34 branch run on the raw image.

    The taps carry semantics at stride 16 and are broadcast to every level; the
    detail stem supplies genuine high-frequency content at strides 4/8/16/32.
    This is the ER arm: ER is a thin reticulum whose boundaries live below the
    patch grid, so a pure upsample of stride-16 features cannot resolve them.

    The detail branch reads ``image`` -- the **network input tensor**, already
    normalised. The two families normalise differently (see
    :class:`quantem.inference.encoders.EncoderContract`): the QuantEM/dinov3
    input is standardised EM, the OmniEM/timm input is raw ``[0, 1]``. Feeding
    the wrong one silently shifts this branch's input distribution, which is why
    the contract is asserted at load time rather than assumed.
    """

    def __init__(
        self,
        embed_dim: int,
        n_taps: int,
        out_channels: int,
        norm: str = "groupnorm",
        width: int = 64,
        fuse: str = "concat",
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.strides = STRIDES
        self.fuse_mode = str(fuse)
        self.tap_fuse = _tap_fuse(embed_dim, n_taps, out_channels)
        self.detail = _ResNet34Stem(width=width, out_channels=out_channels, norm=norm)
        self.mix: nn.ModuleList | None
        if self.fuse_mode == "concat":
            self.mix = nn.ModuleList(
                [ConvGNAct(out_channels * 2, out_channels, k=1) for _ in STRIDES]
            )
        else:  # "add"
            self.mix = None

    def forward(
        self, feats: list[torch.Tensor], image: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        if image is None:
            raise ValueError("resnet34_detail neck requires the raw `image` (its detail branch).")
        sem16 = self.tap_fuse(torch.cat(feats, dim=1))  # [B, C, H/16, W/16]
        detail = self.detail(image)
        mix = self.mix  # non-None exactly when fuse_mode == "concat"; built together
        pyramid = []
        for lvl, d in enumerate(detail):
            s = resize_to(sem16, tuple(d.shape[-2:]))
            if mix is not None:
                pyramid.append(mix[lvl](torch.cat([s, d], dim=1)))
            else:
                pyramid.append(s + d)
        return pyramid


def _build_naive_1x1(
    spec: NeckSpec, embed_dim: int, n_taps: int, patch_size: int, out_channels: int
) -> nn.Module:
    return Naive1x1Neck(embed_dim=embed_dim, n_taps=n_taps, out_channels=out_channels)


def _build_resnet34_detail(
    spec: NeckSpec, embed_dim: int, n_taps: int, patch_size: int, out_channels: int
) -> nn.Module:
    p = spec.params or {}
    return ResNet34DetailNeck(
        embed_dim=embed_dim,
        n_taps=n_taps,
        out_channels=out_channels,
        norm=str(p.get("norm", "groupnorm")),
        width=int(p.get("width", 64)),
        fuse=str(p.get("fuse", "concat")),
    )


NECKS = {
    "naive_1x1": _build_naive_1x1,
    "resnet34_detail": _build_resnet34_detail,
}


def build_neck(
    spec: NeckSpec,
    embed_dim: int,
    n_taps: int,
    patch_size: int,
    out_channels: int = 256,
) -> nn.Module:
    """Dispatch ``spec.type``; unknown types name the ones that ship."""
    builder = NECKS.get(spec.type)
    if builder is None:
        raise ValueError(
            f"Unknown neck type {spec.type!r}. QuantEM ships {sorted(NECKS)}; the "
            "heavy arms (vit_adapter, vit_comer) are not included."
        )
    return builder(spec, embed_dim, n_taps, patch_size, out_channels)
