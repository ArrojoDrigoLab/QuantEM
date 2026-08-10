"""The fixed segmentation decoder heads — whichever one is selected is identical across all compared
encoders.

``build_decoder`` dispatches ``ProbeConfig.decoder`` to one of four heads:
  * ``linear`` (the ProbeConfig default): the DINOv2/DINOv3 dense linear probe — concat the selected
    blocks' patch grids -> BatchNorm2d -> 1x1 conv -> bilinear x16 upsample. Capacity-minimal, so the
    metric reflects the *encoder's* feature quality, not the decoder's.
  * ``light_conv``: a small progressive-upsampling conv head for sharper qualitative masks / a
    boundary-sensitive robustness check.
  * ``upernet``: PPM + FPN over a pyramid synthesised from the selected ViT taps — the higher-capacity
    head the cross-encoder comparison is reported with.
  * ``unet``: the same tap-to-pyramid synthesis decoded by a plain UNet path, as a cheaper check on
    whether the decoder choice changes the ranking.

Only what tracks the encoder — the tap width (embedding_dim) and the number of taps — varies; every
other hyperparameter is fixed, so the metric compares encoders rather than heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, patch_size: int = 16,
                 mode: str = "linear"):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.num_classes = num_classes
        if mode == "linear":
            # trained BN over the concatenated features (DINOv3 detail) + per-patch linear classifier
            self.bn = nn.BatchNorm2d(in_channels)
            self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        elif mode == "light_conv":
            self.bn = nn.BatchNorm2d(in_channels)
            chs = [in_channels, 256, 128, 64, 32]
            blocks = []
            for i in range(4):  # 4 x2 upsamples == x16 == patch_size
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(chs[i], chs[i + 1], 3, padding=1, bias=False),
                        nn.GroupNorm(min(32, chs[i + 1]), chs[i + 1]),
                        nn.GELU(),
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    )
                )
            self.up = nn.Sequential(*blocks)
            self.classifier = nn.Conv2d(chs[-1], num_classes, kernel_size=1)
        else:
            raise ValueError(f"Unknown decoder mode {mode!r} (expected linear|light_conv)")

    def forward(self, feats: list[torch.Tensor], out_hw: tuple[int, int] | None = None) -> torch.Tensor:
        x = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)  # [B, sum(Ci), h, w]
        x = self.bn(x)
        if self.mode == "linear":
            logits = self.classifier(x)  # [B, K, h, w]
            H = out_hw[0] if out_hw else x.shape[2] * self.patch_size
            W = out_hw[1] if out_hw else x.shape[3] * self.patch_size
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        else:
            x = self.up(x)
            logits = self.classifier(x)
            if out_hw and tuple(logits.shape[-2:]) != tuple(out_hw):
                logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return logits

# --------------------------------------------------------------------------- #
# UPerNet head (the "competitor" decoder for the cross-encoder comparison)
# --------------------------------------------------------------------------- #
def _resize(x: torch.Tensor, hw) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    return F.interpolate(x, size=tuple(hw), mode="bilinear", align_corners=False)

class _ConvGNAct(nn.Sequential):
    """conv-k -> GroupNorm -> GELU (GN, not BN: batch-size robust for the small probe batches)."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.GroupNorm(min(32, cout), cout),
            nn.GELU(),
        )

class _PPM(nn.Module):
    """Pyramid Pooling Module: adaptive-pool the coarsest map at several bins, fuse (UPerNet's PPM)."""

    def __init__(self, cin: int, cout: int, bins=(1, 2, 3, 6)):
        super().__init__()
        self.bins = tuple(bins)
        self.stages = nn.ModuleList(_ConvGNAct(cin, cout, 1) for _ in bins)
        self.project = _ConvGNAct(cin + len(self.bins) * cout, cout, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        outs = [x]
        for b, st in zip(self.bins, self.stages):
            outs.append(_resize(st(F.adaptive_avg_pool2d(x, b)), hw))
        return self.project(torch.cat(outs, dim=1))

class UPerNetDecoder(nn.Module):
    """UPerNet over ViT taps — the fixed 'competitor' decoder held identical across all compared encoders.

    ViT blocks all emit the same stride (patch grid), so the UPerNet feature pyramid is synthesised the
    standard ViT way (à la SETR/ViT-Det): assign the selected blocks (fine->coarse) to strides
    (4,8,16,32), project + resample each, run PPM on the coarsest, FPN top-down, fuse, classify, upsample
    to ``out_hw``. Structure matches EMCF's own UPerNet head (PPM bins (1,2,3,6) + FPN); GroupNorm here for
    batch-size robustness. Same head, same params, every encoder — so the metric reflects the *encoder*.
    """

    STRIDES = (4, 8, 16, 32)

    def __init__(self, embed_dim: int, n_taps: int, num_classes: int, patch_size: int,
                 channels: int = 256):
        super().__init__()
        self.patch_size = patch_size
        n = len(self.STRIDES)
        # Map the available taps (>=1) onto the n pyramid levels by even spacing (n_taps==4 -> 1:1).
        self._pick = ([round(i * (n_taps - 1) / (n - 1)) for i in range(n)] if n_taps > 1
                      else [0] * n)
        self.proj = nn.ModuleList(_ConvGNAct(embed_dim, channels, 1) for _ in range(n))
        self.ppm = _PPM(channels, channels)
        self.laterals = nn.ModuleList(_ConvGNAct(channels, channels, 1) for _ in range(n - 1))
        self.fpn_convs = nn.ModuleList(_ConvGNAct(channels, channels, 3) for _ in range(n))
        self.fuse = _ConvGNAct(n * channels, channels, 3)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, feats: list[torch.Tensor], out_hw: tuple[int, int] | None = None) -> torch.Tensor:
        if out_hw is None:
            out_hw = (feats[0].shape[2] * self.patch_size, feats[0].shape[3] * self.patch_size)
        H, W = out_hw
        pyramid = []
        for lvl, s in enumerate(self.STRIDES):
            t = self.proj[lvl](feats[self._pick[lvl]])
            pyramid.append(_resize(t, (max(1, round(H / s)), max(1, round(W / s)))))
        laterals = [self.laterals[i](pyramid[i]) for i in range(len(pyramid) - 1)]
        laterals.append(self.ppm(pyramid[-1]))
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + _resize(laterals[i], laterals[i - 1].shape[-2:])
        fpn = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        tgt = fpn[0].shape[-2:]
        fused = self.fuse(torch.cat([_resize(f, tgt) for f in fpn], dim=1))
        return _resize(self.classifier(fused), out_hw)

class _DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(_ConvGNAct(cin, cout, k=3), _ConvGNAct(cout, cout, k=3))

class UNetDecoder(nn.Module):
    """UNet decoder over ViT taps — the second fixed decoder in the encoder comparison.

    Same self-contained ViT->pyramid synthesis as ``UPerNetDecoder`` (assign the selected blocks
    fine->coarse to strides (4,8,16,32), project + resample), but decode with a plain UNet path
    (coarse->fine, double-conv + additive skips) instead of PPM+FPN. Identical across all compared
    encoders, so the metric still reflects the encoder. Lighter than UPerNet, so it also serves as a
    cheaper check on whether the decoder choice changes the ranking at fixed encoders.
    """

    STRIDES = (4, 8, 16, 32)

    def __init__(self, embed_dim: int, n_taps: int, num_classes: int, patch_size: int,
                 channels: int = 256):
        super().__init__()
        self.patch_size = patch_size
        n = len(self.STRIDES)
        self._pick = ([round(i * (n_taps - 1) / (n - 1)) for i in range(n)] if n_taps > 1
                      else [0] * n)
        self.proj = nn.ModuleList(_ConvGNAct(embed_dim, channels, 1) for _ in range(n))
        self.stages = nn.ModuleList(_DoubleConv(channels, channels) for _ in range(n))
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, feats: list[torch.Tensor], out_hw: tuple[int, int] | None = None) -> torch.Tensor:
        if out_hw is None:
            out_hw = (feats[0].shape[2] * self.patch_size, feats[0].shape[3] * self.patch_size)
        H, W = out_hw
        pyramid = []
        for lvl, s in enumerate(self.STRIDES):
            t = self.proj[lvl](feats[self._pick[lvl]])
            pyramid.append(_resize(t, (max(1, round(H / s)), max(1, round(W / s)))))
        x = self.stages[-1](pyramid[-1])                      # coarsest
        for i in range(len(pyramid) - 2, -1, -1):
            x = _resize(x, pyramid[i].shape[-2:]) + pyramid[i]
            x = self.stages[i](x)
        return _resize(self.classifier(x), out_hw)

def build_decoder(embedding_dim: int, n_layers: int, num_classes: int, patch_size: int,
                  mode: str) -> nn.Module:
    if mode == "upernet":
        return UPerNetDecoder(embed_dim=embedding_dim, n_taps=n_layers, num_classes=num_classes,
                              patch_size=patch_size)
    if mode == "unet":
        return UNetDecoder(embed_dim=embedding_dim, n_taps=n_layers, num_classes=num_classes,
                           patch_size=patch_size)
    return SegDecoder(
        in_channels=embedding_dim * n_layers, num_classes=num_classes,
        patch_size=patch_size, mode=mode,
    )
