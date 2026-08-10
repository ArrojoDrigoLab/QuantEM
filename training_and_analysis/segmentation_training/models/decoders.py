"""Segmentation decoder / mask-head registry: the variable compared in the decoder experiment.

Every decoder consumes the neck feature pyramid — a list of ``[B, in_channels, H/s, W/s]`` maps,
one per stride ``s`` in ``base.STRIDES`` (4, 8, 16, 32), ascending — and returns dense semantic logits
``[B, num_classes, H, W]`` at the requested ``out_hw``. This uniform semantic-logit contract is honored
even by the instance decoders (their instance-specific tensors ride on ``self.aux_logits``) so metrics
and the shared CC / Mutex-Watershed instance post-proc work identically across arms.

Registry keys
-------------
Dense (native, CPU-buildable):
  * ``upernet``               — PPM on the coarsest level + FPN over the pyramid + seg head.
  * ``dpt``                   — DPT-style reassemble + RefineNet fusion.
  * ``nnunet_convnext_unet``  — UNet-over-pyramid, ConvNeXt blocks, deep supervision (the reference control).
  * ``pspnet``                — PPM on the coarsest level + a conv seg head.
  * ``deeplabv3plus``         — ASPP on the coarsest level + a low-level skip and a light decoder.
  * ``unet``                  — plain UNet over the pyramid: additive skips, double convs, no deep supervision.

Instance (native, CPU-buildable; semantic fg logit + instance aux):
  * ``panoptic_deeplab``      — bottom-up: semantic + center-heatmap + center-offset regression.
  * ``affinity_mws``          — affinity head (short attractive + long repulsive) + semantic fg;
                                Mutex-Watershed clustering is an eval-only post-proc (elf/affogato,
                                scipy-CC fallback for CPU smoke).

Optional (heavy dependency, imported lazily at build time so the module itself always imports):
  * ``mask2former_query_hf``  — Mask2Former query decoder built on HuggingFace ``transformers``:
                                MSDeformAttn pixel decoder + masked-attention query decoder +
                                Hungarian set loss (see ``heavy.mask2former_hf``).

All native decoders use ``base.ConvGNAct`` / ``base.resize_to`` (GroupNorm, never BatchNorm).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import base
from .base import STRIDES, ConvGNAct, resize_to


# =================================================================================================
# Dense decoders
# =================================================================================================
class _PPM(nn.Module):
    """Pyramid Pooling Module: adaptive-pool the coarsest map at several bin sizes, fuse."""

    def __init__(self, in_channels: int, out_channels: int, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(
            ConvGNAct(in_channels, out_channels, k=1) for _ in bins
        )
        self.bins = tuple(bins)
        self.project = ConvGNAct(in_channels + len(bins) * out_channels, out_channels, k=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        outs = [x]
        for b, stage in zip(self.bins, self.stages):
            pooled = F.adaptive_avg_pool2d(x, output_size=b)
            outs.append(resize_to(stage(pooled), hw))
        return self.project(torch.cat(outs, dim=1))


class UPerNet(nn.Module):
    """UPerNet: PPM on the coarsest level + FPN over the pyramid -> fuse -> seg head -> out_hw."""

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256):
        super().__init__()
        self.strides = tuple(strides)
        n = len(self.strides)
        self.ppm = _PPM(in_channels, channels)
        # Lateral 1x1 on all-but-coarsest (coarsest comes from PPM).
        self.laterals = nn.ModuleList(ConvGNAct(in_channels, channels, k=1) for _ in range(n - 1))
        # FPN smoothing after top-down add.
        self.fpn_convs = nn.ModuleList(ConvGNAct(channels, channels, k=3) for _ in range(n))
        self.fuse = ConvGNAct(n * channels, channels, k=3)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        self.aux_logits = []
        # Build lateral maps: fine..coarse; coarsest through PPM.
        laterals = [self.laterals[i](pyramid[i]) for i in range(len(pyramid) - 1)]
        laterals.append(self.ppm(pyramid[-1]))
        # Top-down pathway (coarse -> fine).
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + resize_to(laterals[i], laterals[i - 1].shape[-2:])
        feats = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        # Fuse all to the finest resolution.
        target_hw = feats[0].shape[-2:]
        feats = [resize_to(f, target_hw) for f in feats]
        fused = self.fuse(torch.cat(feats, dim=1))
        logits = self.classifier(fused)
        return resize_to(logits, out_hw)


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
    """DPT-style dense decoder: reassemble pyramid levels + progressive RefineNet fusion."""

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256):
        super().__init__()
        self.strides = tuple(strides)
        n = len(self.strides)
        self.reassemble = nn.ModuleList(ConvGNAct(in_channels, channels, k=1) for _ in range(n))
        # One fusion block per level (each: RCU on incoming, add upsampled deeper, RCU, upsample).
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
        # Fuse coarse -> fine.
        path = None
        for i in range(len(feats) - 1, -1, -1):
            cur = self.rcu_in[i](feats[i])
            if path is not None:
                cur = cur + resize_to(path, cur.shape[-2:])
            path = self.rcu_out[i](cur)
        logits = self.head(path)
        return resize_to(logits, out_hw)


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: 7x7 depthwise -> GN -> 1x1 expand -> GELU -> 1x1 project, residual."""

    def __init__(self, channels: int, expand: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.pw1 = nn.Conv2d(channels, channels * expand, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * expand, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        return x + y


class NNUNetConvNeXtUNet(nn.Module):
    """UNet decoder over the pyramid with ConvNeXt blocks + deep supervision (aux logits per stage).

    The pyramid (fine..coarse) is the encoder path. Decoding runs coarse -> fine, emitting an auxiliary
    seg logit at every decoder stage into ``self.aux_logits`` (coarsest -> ... , the primary output is
    the finest at ``out_hw``). The train loop deep-supervises against downsampled targets.
    """

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256,
                 blocks_per_stage: int = 2):
        super().__init__()
        self.strides = tuple(strides)
        n = len(self.strides)
        self.num_classes = num_classes
        # Project each pyramid level to the working width.
        self.proj = nn.ModuleList(ConvGNAct(in_channels, channels, k=1) for _ in range(n))
        # Decoder stages fuse the deeper (upsampled) feature with the skip at each level.
        self.up = nn.ModuleList(ConvGNAct(channels, channels, k=3) for _ in range(n - 1))
        self.stages = nn.ModuleList(
            nn.Sequential(*[_ConvNeXtBlock(channels) for _ in range(blocks_per_stage)])
            for _ in range(n)
        )
        # A seg head at every decoder level for deep supervision.
        self.heads = nn.ModuleList(nn.Conv2d(channels, num_classes, 1) for _ in range(n))
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        self.aux_logits = []
        feats = [self.proj[i](pyramid[i]) for i in range(len(pyramid))]
        n = len(feats)
        # Start at the coarsest, walk to the finest.
        x = self.stages[n - 1](feats[n - 1])
        aux: list[torch.Tensor] = [self.heads[n - 1](x)]
        for i in range(n - 2, -1, -1):
            up = resize_to(self.up[i](x), feats[i].shape[-2:])
            x = self.stages[i](up + feats[i])
            aux.append(self.heads[i](x))
        # aux is coarse..fine; the finest map is the primary output.
        primary = resize_to(aux[-1], out_hw)
        # Deep-supervision aux = the coarser heads (all but the finest), kept at their native res.
        self.aux_logits = aux[:-1]
        return primary


# =================================================================================================
# Simple dense baselines (PSPNet / DeepLabv3+ / UNet) — the standard decoder recipes that the final
# segmentation comparison contrasts the resnet34_detail+DPT recipe against, at a fixed encoder. Each
# consumes the neck pyramid (fine..coarse) and returns dense logits at out_hw; GroupNorm throughout.
# =================================================================================================
class PSPNet(nn.Module):
    """PSPNet: Pyramid Pooling Module on the coarsest level + a conv seg head (the classic simple head)."""

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256):
        super().__init__()
        self.strides = tuple(strides)
        self.ppm = _PPM(in_channels, channels)
        self.head = nn.Sequential(ConvGNAct(channels, channels, k=3),
                                  nn.Conv2d(channels, num_classes, kernel_size=1))
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        self.aux_logits = []
        x = self.ppm(pyramid[-1])
        return resize_to(self.head(x), out_hw)


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLabv3+): parallel dilated convs + image-level global context."""

    def __init__(self, in_channels: int, out_channels: int, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(ConvGNAct(in_channels, out_channels, k=1))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                    nn.GroupNorm(min(32, out_channels), out_channels), nn.GELU()))
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                         ConvGNAct(in_channels, out_channels, k=1))
        self.project = ConvGNAct((len(rates) + 1) * out_channels, out_channels, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        outs.append(resize_to(self.global_pool(x), hw))
        return self.project(torch.cat(outs, dim=1))


class DeepLabV3Plus(nn.Module):
    """DeepLabv3+: ASPP on the coarsest level + a low-level (finest) skip + a light decoder."""

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256,
                 low_level_channels: int = 64):  # 64 (not the classic 48): GroupNorm needs /32 channels
        super().__init__()
        self.strides = tuple(strides)
        self.aspp = _ASPP(in_channels, channels)
        self.low_proj = ConvGNAct(in_channels, low_level_channels, k=1)
        self.decoder = nn.Sequential(ConvGNAct(channels + low_level_channels, channels, k=3),
                                     ConvGNAct(channels, channels, k=3))
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        self.aux_logits = []
        x = self.aspp(pyramid[-1])                      # coarse semantic context
        low = self.low_proj(pyramid[0])                 # finest level -> low-level detail
        x = resize_to(x, low.shape[-2:])
        x = self.decoder(torch.cat([x, low], dim=1))
        return resize_to(self.classifier(x), out_hw)


class _DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(ConvGNAct(cin, cout, k=3), ConvGNAct(cout, cout, k=3))


class SimpleUNet(nn.Module):
    """Plain UNet decoder over the pyramid (coarse->fine, additive skips, double-conv). Deliberately
    the *simple* UNet baseline — no ConvNeXt blocks, no deep supervision (that's nnunet_convnext_unet)."""

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256):
        super().__init__()
        self.strides = tuple(strides)
        n = len(self.strides)
        self.proj = nn.ModuleList(ConvGNAct(in_channels, channels, k=1) for _ in range(n))
        self.stages = nn.ModuleList(_DoubleConv(channels, channels) for _ in range(n))
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        self.aux_logits = []
        feats = [self.proj[i](pyramid[i]) for i in range(len(pyramid))]
        x = self.stages[-1](feats[-1])                  # coarsest
        for i in range(len(feats) - 2, -1, -1):
            x = resize_to(x, feats[i].shape[-2:]) + feats[i]
            x = self.stages[i](x)
        return resize_to(self.classifier(x), out_hw)


# =================================================================================================
# Instance decoders (semantic fg logit + instance aux)
# =================================================================================================
class _SharedDecoderTrunk(nn.Module):
    """Fuse the pyramid to the finest resolution -> a shared feature map (used by instance heads)."""

    def __init__(self, in_channels: int, strides: tuple, channels: int = 256):
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


class PanopticDeepLab(nn.Module):
    """Native bottom-up: dense semantic head + instance center-heatmap + center-offset regression.

    ``self.aux_logits = [center_logits, offset_reg]`` where
      * ``center_logits`` [B, 1, H, W]  — per-pixel object-center heatmap logit.
      * ``offset_reg``    [B, 2, H, W]  — (dy, dx) regression from each pixel to its object center
        (in pixel units at ``out_hw`` scale). Grouping to instances is an eval-time post-proc.
    """

    # aux index 1 (offset) is position-dependent: it points to the within-window instance centroid, so it
    # cannot be Hann-averaged across sliding-window seams. predict_region takes each pixel's
    # offset from the window where it is most central instead. aux index 0 (center heatmap) is position-
    # invariant and blends normally.
    AUX_POSITION_DEPENDENT = (1,)

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256):
        super().__init__()
        self.strides = tuple(strides)
        self.sem_trunk = _SharedDecoderTrunk(in_channels, strides, channels)
        self.ins_trunk = _SharedDecoderTrunk(in_channels, strides, channels)
        self.sem_head = nn.Sequential(ConvGNAct(channels, channels, k=3),
                                      nn.Conv2d(channels, num_classes, 1))
        self.center_head = nn.Sequential(ConvGNAct(channels, channels, k=3),
                                         nn.Conv2d(channels, 1, 1))
        self.offset_head = nn.Sequential(ConvGNAct(channels, channels, k=3),
                                         nn.Conv2d(channels, 2, 1))
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        sem = self.sem_trunk(pyramid)
        ins = self.ins_trunk(pyramid)
        sem_logits = resize_to(self.sem_head(sem), out_hw)
        center = resize_to(self.center_head(ins), out_hw)
        offset = resize_to(self.offset_head(ins), out_hw)
        self.aux_logits = [center, offset]
        return sem_logits

    def native_instance_labels(self, aux, fg):
        """True-instance label map from center/offset grouping (eval-only). ``aux=[center_logit, offset]``
        (full-region, accumulated over the sliding window as logits); ``fg`` = semantic foreground ``[H,W]``."""
        center = torch.sigmoid(torch.as_tensor(aux[0]).float())   # logit -> [0,1] (torch/numpy-safe)
        return panoptic_instance_postproc(center, aux[1], fg)


class AffinityMWS(nn.Module):
    """Affinity head (short-range attractive + long-range repulsive) + semantic fg logit.

    ``self.aux_logits = [affinities]`` with ``affinities`` [B, n_offsets, H, W] in [0,1] (sigmoid),
    predicted from the finest pyramid level. Channel ``c`` is the affinity along ``self.offsets[c]``
    (a (dy, dx) displacement). Affinities are position-invariant scalar edge weights, so they Hann-blend
    across sliding-window seams correctly -> no position-dependent aux. Short-range offsets
    are attractive (bind within an object); long-range offsets act as repulsive mutex edges. The
    Mutex-Watershed clustering is an eval-only post-proc
    (see ``mutex_watershed_postproc``); training regresses these affinities directly.

    Offsets convention: list of (dy, dx). The first ``n_short`` are the nearest-neighbour attractive
    edges; the remainder are the long-range repulsive edges.
    """

    # (dy, dx). Short-range = direct + diagonal neighbours; long-range = repulsive mutex edges.
    DEFAULT_OFFSETS = (
        (0, 1), (1, 0), (1, 1), (1, -1),               # short-range attractive
        (0, 3), (3, 0), (0, 9), (9, 0), (9, 9), (9, -9),  # long-range repulsive
    )
    N_SHORT = 4

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, channels: int = 256,
                 offsets=None):
        super().__init__()
        self.strides = tuple(strides)
        self.offsets = tuple(tuple(o) for o in (offsets or self.DEFAULT_OFFSETS))
        self.n_short = self.N_SHORT
        self.sem_trunk = _SharedDecoderTrunk(in_channels, strides, channels)
        # Affinities from the finest pyramid level (index 0).
        self.aff_proj = ConvGNAct(in_channels, channels, k=3)
        self.aff_block = ConvGNAct(channels, channels, k=3)
        self.aff_head = nn.Conv2d(channels, len(self.offsets), 1)
        self.sem_head = nn.Sequential(ConvGNAct(channels, channels, k=3),
                                      nn.Conv2d(channels, num_classes, 1))
        self.aux_logits: list[torch.Tensor] = []

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        sem = self.sem_trunk(pyramid)
        sem_logits = resize_to(self.sem_head(sem), out_hw)
        a = self.aff_block(self.aff_proj(pyramid[0]))
        aff = torch.sigmoid(self.aff_head(a))
        aff = resize_to(aff, out_hw)
        self.aux_logits = [aff]
        return sem_logits

    def native_instance_labels(self, aux, fg):
        """True-instance label map from mutex-watershed on the predicted affinities (eval-only). ``aux=[aff]``
        (full-region, in [0,1]). MWS labels the entire field, so restrict instances to the semantic
        foreground ``fg`` (background pixels -> 0) for a fair comparison against the GT instance map."""
        import numpy as np
        seg = mutex_watershed_postproc(aux[0], self.offsets, n_attractive=self.n_short)
        fg_bool = np.asarray(fg).astype(bool)
        if fg_bool.shape == seg.shape:
            seg = seg + 1          # shift MWS region ids to >=1 so 0 can denote background uniquely
            seg[~fg_bool] = 0      # background -> 0
        return seg.astype(np.int32)


def mutex_watershed_postproc(affinities: torch.Tensor, offsets, n_attractive: int = 4) -> "object":
    """Eval-only Mutex-Watershed instance labelling from predicted affinities.

    Args:
        affinities:  [n_offsets, H, W] (or [B, n_offsets, H, W] -> first batch element used) in [0,1],
                     predicting P(same instance) for every offset, short- and long-range alike.
        offsets:     sequence of (dy, dx) matching the affinity channel order. The first ``n_attractive``
                     channels are the short-range attractive edges; the rest are long-range mutex edges.
        n_attractive: number of leading short-range attractive channels.

    Returns a numpy int32 label map [H, W]. Uses ``affogato`` Mutex-Watershed when available; otherwise
    falls back to a scipy connected-components labelling on a thresholded short-range affinity interior
    map so CPU smoke still yields *some* instance segmentation.

    affogato conventions (those of ``affogato.segmentation.mws``): ``strides`` has length = spatial
    ndim, not n_offsets; edges are processed highest-weight-first, where attractive channels merge and
    mutex channels *separate*. This network predicts P(same) for all channels, so the mutex (long-range)
    channels are inverted to P(split)=1-P(same) before clustering.
    """
    import numpy as np

    aff = affinities
    if hasattr(aff, "detach"):
        aff = aff.detach().to("cpu").float().numpy()
    aff = np.asarray(aff, dtype="float32")
    if aff.ndim == 4:
        aff = aff[0]
    offsets = [tuple(int(v) for v in o) for o in offsets]
    ndim = len(offsets[0])                                   # spatial dims (2 for 2D)
    n_att = int(max(1, min(n_attractive, aff.shape[0])))

    try:  # scoped to the import alone, so a call-time affogato error propagates instead of falling back.
        base.require("affogato", arm="affinity_mws")
        from affogato.segmentation import compute_mws_segmentation  # type: ignore
        _have_mws = True
    except ImportError:
        _have_mws = False

    if _have_mws:
        import os
        w = aff.copy()
        w[n_att:] = 1.0 - w[n_att:]                          # mutex channels: high weight == separate
        # Full-edge MWS (stride 1) is single-threaded and impractically slow on large multi-channel regions.
        # QUANTEM_MWS_STRIDE>1 deterministically subsamples the long-range mutex edges (standard MWS deployment
        # practice); the short-range attractive edges are always kept, so merging is full-res and only the
        # split constraints are thinned -> large speedup, minor separation cost on touching instances.
        stride = max(1, int(os.environ.get("QUANTEM_MWS_STRIDE", "1")))
        seg = compute_mws_segmentation(
            w, offsets, number_of_attractive_channels=n_att,
            strides=[stride] * ndim, randomize_strides=False,
        )
        return seg.astype(np.int32)

    # affogato absent. A run that requires true mutex-watershed sets QUANTEM_REQUIRE_MWS
    # -> fail-closed, so a connected-components result is never reported as Mutex-Watershed.
    import os
    import warnings
    if os.environ.get("QUANTEM_REQUIRE_MWS"):
        raise RuntimeError(
            "affogato (conda-forge) unavailable but QUANTEM_REQUIRE_MWS is set: refusing to fall back to "
            "connected-components. Install with: mamba install -c conda-forge affogato")
    warnings.warn(
        "affogato unavailable: mutex_watershed_postproc is using the connected-components fallback rather "
        "than Mutex-Watershed, so affinity instance metrics are a connected-components floor, not MWS.",
        RuntimeWarning, stacklevel=2)
    from scipy import ndimage  # lazy, no GPU needed.
    mean_aff = aff[:n_att].mean(axis=0)                      # short-range affinity -> interior proxy
    interior = mean_aff > 0.5
    labels, _ = ndimage.label(interior)
    return labels.astype(np.int32)


def panoptic_instance_postproc(center, offset, fg, center_thresh: float = 0.1,
                               nms_kernel: int = 7) -> "object":
    """Eval-only Panoptic-DeepLab instance grouping from the panoptic head's center heatmap + offset field.

    Args:
        center: ``[1,H,W]``/``[H,W]`` object-center heatmap in [0,1] (already sigmoid'd).
        offset: ``[2,H,W]`` (dy,dx) regression from each pixel to its instance center (pixel units).
        fg:     ``[H,W]`` bool/{0,1} semantic-foreground mask (only fg pixels get instance ids).
        center_thresh: min heatmap value for a peak; nms_kernel: max-pool NMS window.

    Returns a numpy int32 label map ``[H,W]`` (0=background). If no center peaks are found, falls back to a
    connected-components labelling of the foreground so eval always yields *some* instance segmentation.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    def _t(a):
        return a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))

    center = _t(center).float()
    if center.dim() == 3:
        center = center[0]
    offset = _t(offset).float()
    fg = _t(fg).bool()
    if fg.dim() == 3:
        fg = fg[0]
    H, W = center.shape

    # 1. Center detection: max-pool NMS -> local maxima above threshold.
    c = center[None, None]
    pooled = F.max_pool2d(c, nms_kernel, stride=1, padding=nms_kernel // 2)
    peaks = (c[0, 0] == pooled[0, 0]) & (center > center_thresh)
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    if ys.numel() == 0 or fg.sum() == 0:
        import warnings
        warnings.warn("panoptic_instance_postproc: no center peaks detected, so the labelling falls back to "
                      "connected components rather than center/offset grouping.", RuntimeWarning, stacklevel=2)
        from scipy import ndimage  # CC fallback (no centers detected)
        lab, _ = ndimage.label(fg.cpu().numpy())
        return lab.astype(np.int32)
    centers = torch.stack([ys.float(), xs.float()], dim=1)          # [C, 2]

    # 2. Each fg pixel votes for a center: voted = (y,x) + offset(y,x); assign to the nearest detected center.
    fy, fx = torch.nonzero(fg, as_tuple=True)                        # [N]
    voted = torch.stack([fy.float() + offset[0][fg], fx.float() + offset[1][fg]], dim=1)  # [N, 2]
    dist = torch.cdist(voted, centers)                              # [N, C]
    assign = dist.argmin(dim=1) + 1                                 # instance ids 1..C
    lab = torch.zeros((H, W), dtype=torch.int32)
    lab[fy, fx] = assign.to(torch.int32)
    return lab.cpu().numpy()


# =================================================================================================
# Registry + factory
# =================================================================================================
def _b_upernet(spec, in_channels, strides, num_classes):
    p = dict(spec.params)
    return UPerNet(in_channels, strides, num_classes, channels=int(p.get("channels", 256)))


def _b_dpt(spec, in_channels, strides, num_classes):
    p = dict(spec.params)
    return DPT(in_channels, strides, num_classes, channels=int(p.get("channels", 256)))


def _b_nnunet(spec, in_channels, strides, num_classes):
    p = dict(spec.params)
    return NNUNetConvNeXtUNet(in_channels, strides, num_classes,
                              channels=int(p.get("channels", 256)),
                              blocks_per_stage=int(p.get("blocks_per_stage", 2)))


def _b_pspnet(spec, in_channels, strides, num_classes):
    p = dict(spec.params)
    return PSPNet(in_channels, strides, num_classes, channels=int(p.get("channels", 256)))


def _b_deeplabv3plus(spec, in_channels, strides, num_classes):
    p = dict(spec.params)
    return DeepLabV3Plus(in_channels, strides, num_classes, channels=int(p.get("channels", 256)),
                         low_level_channels=int(p.get("low_level_channels", 64)))


def _b_unet(spec, in_channels, strides, num_classes):
    p = dict(spec.params)
    return SimpleUNet(in_channels, strides, num_classes, channels=int(p.get("channels", 256)))


def _b_panoptic(spec, in_channels, strides, num_classes):
    p = dict(spec.params)
    return PanopticDeepLab(in_channels, strides, num_classes, channels=int(p.get("channels", 256)))


def _b_affinity(spec, in_channels, strides, num_classes):
    p = dict(spec.params)
    return AffinityMWS(in_channels, strides, num_classes,
                       channels=int(p.get("channels", 256)),
                       offsets=p.get("offsets"))


def _b_m2f_query_hf(spec, in_channels, strides, num_classes):
    # Detectron2-free query decoder via HuggingFace transformers' pure-PyTorch Mask2Former internals
    # (same architecture: MSDeformAttn pixel decoder + masked-attention query decoder + Hungarian
    # matcher + set criterion). The runnable path where detectron2 is unavailable.
    from .heavy.mask2former_hf import build_mask2former_query_hf

    return build_mask2former_query_hf(in_channels, strides, num_classes, params=spec.params)


DECODERS: dict[str, object] = {
    "upernet": _b_upernet,
    "dpt": _b_dpt,
    "nnunet_convnext_unet": _b_nnunet,
    "pspnet": _b_pspnet,
    "deeplabv3plus": _b_deeplabv3plus,
    "unet": _b_unet,
    "panoptic_deeplab": _b_panoptic,
    "affinity_mws": _b_affinity,
    "mask2former_query_hf": _b_m2f_query_hf,
}


def build_decoder(spec, in_channels: int, strides: tuple, num_classes: int) -> nn.Module:
    """Dispatch ``spec.type`` via ``DECODERS``; unknown -> ValueError listing valid keys."""
    builder = DECODERS.get(spec.type)
    if builder is None:
        raise ValueError(
            f"Unknown decoder type {spec.type!r}. Valid keys: {sorted(DECODERS)}"
        )
    return builder(spec, in_channels, tuple(strides), num_classes)
