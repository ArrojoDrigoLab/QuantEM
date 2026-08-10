"""Shared building blocks and the assembled ``SegModel``.

Scoped to inference: the lazy-import helper for the H100-only arms, the style
conditioner wiring and ``trainable_parameters()`` are not included, since nothing
at inference time uses them.

The contract every neck and decoder honours -- do not change it, the released
checkpoints are keyed to it:

* Encoder taps: ``feats`` = list of ``[B, embed_dim, H/p, W/p]`` patch grids, one
  per selected block in **ascending block order**. Ascending matters: the neck
  concatenates them on the channel axis, so a different order silently permutes
  the input to a trained 1x1 convolution.
* Neck: ``forward(feats, image) -> list[Tensor]``, one map per stride in
  :data:`STRIDES`, each ``[B, neck_channels, H/s, W/s]``. ``image`` is the
  normalised network input; the ``resnet34_detail`` neck runs a high-frequency
  branch on it.
* Decoder: ``forward(pyramid, out_hw) -> [B, num_classes, H, W]``. Instance
  decoders put their non-semantic outputs on ``self.aux_logits`` and still return
  semantic logits here, so the foreground reduction in
  :mod:`quantem.inference.engine` is uniform across all eight packs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Canonical output pyramid strides, relative to the network input tile.
STRIDES: tuple[int, ...] = (4, 8, 16, 32)


class ConvGNAct(nn.Sequential):
    """conv -> GroupNorm -> GELU.

    GroupNorm rather than BatchNorm throughout: segmentation runs use tiny
    batches, and at inference the app runs batch size 1, where BatchNorm's
    running statistics would make the result depend on how tiles were grouped.
    """

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 32) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
            nn.GroupNorm(min(groups, cout), cout),
            nn.GELU(),
        )


def resize_to(x: torch.Tensor, hw: Sequence[int]) -> torch.Tensor:
    """Bilinear resample to ``hw``, a no-op when already there.

    ``hw`` is a ``Sequence[int]`` because callers pass both literal tuples and
    ``torch.Size`` slices of another tensor's shape.
    """
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    return F.interpolate(x, size=tuple(hw), mode="bilinear", align_corners=False)


class SegModel(nn.Module):
    """Encoder + neck + decoder: the single module the engine calls.

    ``encoder`` is a :class:`quantem.inference.encoders.FrozenEncoder`. It is
    frozen unless the pack adapts it (``lora`` installs adapters inside it;
    ``last_n``/``full`` replace base block weights) -- but at inference nothing
    is trained either way, so ``features()`` always runs under ``no_grad``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        layers: list[int],
    ) -> None:
        super().__init__()
        # Annotated Any: these are duck-typed against the contracts in this
        # module's docstring, not a nominal base class. The values are nn.Modules,
        # so torch still registers them and `.to(device)` reaches them.
        self.encoder: Any = encoder
        self.neck: Any = neck
        self.decoder: Any = decoder
        self.layers = list(layers)

    def features(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder.features(x, self.layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.features(image)
        pyramid = self.neck(feats, image)
        return self.decoder(pyramid, out_hw=image.shape[-2:])

    @property
    def aux_logits(self) -> list[torch.Tensor]:
        """Instance-side outputs from the last forward, or ``[]``.

        For the ``affinity_mws`` packs this is ``[affinities]``, ``[B, 10, H, W]``
        in ``[0, 1]``. They are what a mutex watershed would need to split two
        touching organelles; the app's post-processing is connected components
        and does not consume them yet. See ``../README.md``.
        """
        return getattr(self.decoder, "aux_logits", []) or []
