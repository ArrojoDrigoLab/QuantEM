"""Model contracts + the assembled SegModel for the three-stage segmentation pipeline.

    ENCODER (frozen) -> NECK -> DECODER (mask head)

Contract (every neck/decoder in models/{necks,decoders}.py must honor this so arms are swappable):

  * Encoder taps: ``feats`` = list of ``[B, embed_dim, H/16, W/16]`` patch grids, one per selected
    block (ascending order), from ``FrozenEncoder.features(x, layers)`` (frozen -> no_grad; any
    non-frozen ``encoder.adapt`` arm runs it with grad so the trainable encoder params get gradients).
  * NECK: ``forward(feats: list[Tensor], image: Tensor|None) -> list[Tensor]`` returning a feature
    pyramid — one map per stride in ``STRIDES`` (4,8,16,32), each ``[B, neck_channels, H/s, W/s]``.
    A neck may inject a raw-image high-frequency branch (``image`` is the normalised input, 1-channel).
  * DECODER: ``forward(pyramid: list[Tensor], out_hw) -> Tensor`` returning dense logits
    ``[B, num_classes, H, W]``. Deep-supervision decoders may stash aux logits on ``self.aux_logits``
    (a list) for the training loop; eval always uses the primary output.

Instance decoders (the decoder experiment mito 3-way: query / bottom-up / affinity) reuse the same dense-logits contract
for their semantic foreground channel (so metrics + the fixed CC/MWS instance post-proc work
uniformly); their instance-specific outputs ride on ``self.aux_logits`` / decoder attributes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical output pyramid strides (relative to the network input, i.e. the 512 canonical tile).
STRIDES: tuple[int, ...] = (4, 8, 16, 32)


def require(pkg: str, *, arm: str) -> object:
    """Lazy-import an optional reference implementation; actionable error if absent.

    The mutex-watershed post-processing of the ``affinity_mws`` arm rests on the optional
    ``affogato`` package. Routing that import through this helper names both the package and the
    arm in the error, so the caller can fall back or fail closed on a clear signal instead of an
    opaque ImportError.
    """
    import importlib

    try:
        return importlib.import_module(pkg)
    except Exception as exc:  # pragma: no cover - exercised on the CPU path only
        raise ImportError(
            f"the segmentation arm '{arm}' needs '{pkg}', which is not installed in this environment. "
            f"Install it to enable this arm. "
            f"Original error: {exc!r}"
        ) from exc


class ConvGNAct(nn.Sequential):
    """conv3x3 -> GroupNorm -> GELU (the shared building block; GN is batch-size robust)."""

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


class SegModel(nn.Module):
    """Frozen encoder + trainable neck + decoder. The single module the train/eval loops drive."""

    def __init__(self, encoder, neck: nn.Module, decoder: nn.Module, layers: list[int],
                 encoder_trainable: bool = False, conditioner: nn.Module | None = None):
        super().__init__()
        self.encoder = encoder  # FrozenEncoder (params frozen; adapters, if any, live inside it)
        self.neck = neck
        self.decoder = decoder
        self.layers = list(layers)
        self.encoder_trainable = bool(encoder_trainable)  # True whenever encoder.adapt trains encoder params
        self.conditioner = conditioner  # image-style conditioning (None = no conditioning arms)

    def features(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder.features(x, self.layers, grad=self.encoder_trainable)

    def set_conditioning_context(self, **kwargs) -> None:
        """Set the conditioner's per-forward context (metadata / source_ids / alpha / preset code)."""
        if self.conditioner is not None:
            self.conditioner.set_context(**kwargs)

    def set_record_context(self, record: dict, device) -> None:
        """Eval helper: set per-record metadata/source context (batch size 1)."""
        if self.conditioner is not None:
            self.conditioner.set_record_context(record, device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Conditional encoder adapters: compute the per-image/source code + install it on every block's
        # CondLoRAAdapter before the backbone runs (the adapter hooks read it during self.features).
        cond_ctrl = getattr(self.encoder, "_cond_lora_ctrl", None)
        if cond_ctrl is not None:
            src = getattr(self, "_cond_lora_source_ids", None)
            cond_ctrl.before_forward(image, source_ids=src)
        feats = self.features(image)
        if self.conditioner is not None:
            # Compute + install the active style code (FiLM hooks read it during neck/decoder forward).
            self.conditioner.before_forward(image, feats)
        pyramid = self.neck(feats, image)
        return self.decoder(pyramid, out_hw=image.shape[-2:])

    @property
    def aux_logits(self):
        """Deep-supervision / instance aux outputs from the last decoder forward (or [])."""
        return getattr(self.decoder, "aux_logits", []) or []

    def trainable_parameters(self):
        """Params the optimizer should see: neck + decoder always; encoder adapters if trainable;
        the image-style conditioner (style encoder + FiLM heads + adversary) when present."""
        seen = set()
        mods = [self.neck, self.decoder]
        if self.conditioner is not None:
            mods.append(self.conditioner)
        for m in mods:
            for p in m.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    yield p
        if self.encoder_trainable:
            for p in self.encoder.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    yield p
