"""DoDNet-style organelle-conditioned dense head (dynamic filter generation).

Ports DoDNet's dynamic head (Zhang et al., "DoDNet: Learning to segment multi-organ and tumors from
multiple partially labeled datasets", CVPR 2021; arXiv:2011.10217; github.com/jianpengz/DoDNet) to the
segmentation training pyramid→dense-head contract. In DoDNet a single shared encoder/decoder is followed by a small
controller MLP that maps the task/organelle one-hot code to the weights (and biases) of the final few
conv layers, which are then applied as a dynamic filtering head. One shared network segments many organelles;
the task code selects the task. This is the mechanism that lets a single model be a specialist-by-code.

Adaptation to the feature contract used here:
  * The neck pyramid is fused to the finest stride with a shared trunk (``ConvGNAct`` blocks, GroupNorm — the
    segmentation training building block), exactly like the native instance decoders' ``_SharedDecoderTrunk``.
  * A controller MLP maps the K-dim organelle code -> the flattened weights+biases of a small dynamic head:
    two 1×1 conv layers ``mid_channels`` then a final 1×1 to ``num_classes`` (DoDNet uses 3 dynamic conv
    layers, as does the default here; the count is configurable). The dynamic convs are applied per-sample
    (grouped conv over the batch so each sample uses its own generated weights) — DoDNet's exact trick.
  * Semantic-logit contract preserved: ``forward(pyramid, out_hw) -> [B, num_classes, H, W]``; ``aux_logits=[]``.

Registered into ``models.decoders.DECODERS`` under key ``"dodnet"`` at import (additive and reversible;
``decoders.py`` is not modified). ``build_decoder(spec, ...)`` then dispatches it like any native head.

Organelle code: set via ``set_organelle_code(code)`` before the forward pass (a ``[B, K]`` or ``[K]`` tensor /
index). The mixed-dataset trainer sets it per step; eval sets it once per organelle. If unset, defaults to
organelle index 0 (so a plain ``model(x)`` still runs — needed by the sliding-window eval which calls
``model(image)``).

A FiLM-MoE variant (``mechanism="film_moe"``) reuses ``FiLMHead(n_experts=K)`` as an alternative to the
dynamic-conv path; the default mechanism is the dynamic conv (``mechanism="dynamic"``), which is what the
study reports.

Torch-only (CPU safe); GroupNorm throughout rather than BatchNorm, matching the rest of the segmentation
stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models.base import ConvGNAct, resize_to


class _SharedTrunk(nn.Module):
    """Fuse the pyramid (fine..coarse) to the finest resolution -> one [B, C, H/4, W/4] feature map.

    Mirrors ``models.decoders._SharedDecoderTrunk`` (kept local to avoid importing a private symbol)."""

    def __init__(self, in_channels: int, n_levels: int, channels: int = 256):
        super().__init__()
        self.proj = nn.ModuleList(ConvGNAct(in_channels, channels, k=1) for _ in range(n_levels))
        self.fuse = ConvGNAct(n_levels * channels, channels, k=3)
        self.refine = ConvGNAct(channels, channels, k=3)

    def forward(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        target_hw = pyramid[0].shape[-2:]
        feats = [resize_to(self.proj[i](pyramid[i]), target_hw) for i in range(len(pyramid))]
        return self.refine(self.fuse(torch.cat(feats, dim=1)))


class DoDNetHead(nn.Module):
    """DoDNet dynamic-head decoder: shared trunk + a controller that generates the final conv weights from
    the organelle code.

    Args:
        in_channels:  neck out-channels (pyramid map channels).
        strides:      pyramid strides (len = #levels).
        num_classes:  output classes (binary-per-organelle -> 2).
        channels:     trunk width.
        n_organelles: K, the size of the organelle code (one-hot dimension). The controller input dim.
        mid_channels: width of the dynamic hidden conv layers.
        n_dynamic:    number of dynamic conv layers (>=1; DoDNet uses 3, as does the default here). All 1×1.
        mechanism:    ``"dynamic"`` (DoDNet dynamic filter generation; the default) or
                      ``"film_moe"`` (FiLMHead(n_experts=K) modulation of the trunk + a static head — the
                      simpler alternative mechanism).
    """

    def __init__(self, in_channels: int, strides: tuple, num_classes: int, *, channels: int = 256,
                 n_organelles: int = 2, mid_channels: int = 8, n_dynamic: int = 3,
                 mechanism: str = "dynamic"):
        super().__init__()
        self.strides = tuple(strides)
        self.num_classes = int(num_classes)
        self.n_organelles = max(1, int(n_organelles))
        self.mid_channels = int(mid_channels)
        self.n_dynamic = max(1, int(n_dynamic))
        self.mechanism = str(mechanism)
        self.trunk = _SharedTrunk(in_channels, len(self.strides), channels)
        # Project the trunk feature to the dynamic head's input width (the "feature map" the dynamic convs
        # filter). DoDNet applies the generated filters to a fixed-width feature; here it is ``mid_channels``.
        self.pre = ConvGNAct(channels, self.mid_channels, k=3)
        self.aux_logits: list[torch.Tensor] = []

        # Dynamic conv layer shapes (all 1×1 convs; grouped-per-sample at apply time).
        # layer 0: mid->mid, ... , layer n-2: mid->mid, layer n-1: mid->num_classes.
        widths = [self.mid_channels] * self.n_dynamic + [self.num_classes]
        self._layer_shapes = [(widths[i + 1], widths[i]) for i in range(self.n_dynamic)]  # (cout, cin)
        n_weight = sum(cout * cin for cout, cin in self._layer_shapes)                    # 1×1 -> k=1
        n_bias = sum(cout for cout, _ in self._layer_shapes)
        self._n_weight, self._n_bias = n_weight, n_bias

        if self.mechanism == "dynamic":
            # Controller MLP: organelle code (K) -> flattened (weights ++ biases) of the dynamic head.
            self.controller = nn.Sequential(
                nn.Linear(self.n_organelles, 256), nn.GELU(),
                nn.Linear(256, n_weight + n_bias),
            )
        elif self.mechanism == "film_moe":
            # Alternative: FiLM(n_experts=K) modulates the trunk feature, then a static conv head. Reuses
            # the MoE-conditioned head from the image-style conditioning module.
            from ...models.conditioning.film import FiLMHead
            self.film = FiLMHead(self.n_organelles, self.mid_channels, n_experts=self.n_organelles)
            self.static_head = nn.Conv2d(self.mid_channels, self.num_classes, 1)
        else:
            raise ValueError(f"DoDNetHead mechanism must be 'dynamic'|'film_moe', got {mechanism!r}")

        # Active organelle code, set via set_organelle_code() before forward (default: index 0 one-hot).
        self.register_buffer("_default_code", self._one_hot(0), persistent=False)
        self._code: torch.Tensor | None = None

    # -- code handling ------------------------------------------------------
    def _one_hot(self, idx: int) -> torch.Tensor:
        v = torch.zeros(self.n_organelles)
        v[int(idx) % self.n_organelles] = 1.0
        return v

    def set_organelle_code(self, code) -> None:
        """Set the active organelle code before ``forward``. Accepts an int index, a 1-D ``[K]`` one-hot,
        or a batched ``[B, K]`` one-hot tensor. Stored as a float tensor; broadcast to the batch at forward."""
        if code is None:
            self._code = None
            return
        if isinstance(code, int):
            self._code = self._one_hot(code)
            return
        t = torch.as_tensor(code).float()
        self._code = t

    def _resolve_code(self, batch: int, device, dtype) -> torch.Tensor:
        c = self._code if self._code is not None else self._default_code
        c = c.to(device=device, dtype=dtype)
        if c.dim() == 1:
            c = c.unsqueeze(0).expand(batch, -1)
        elif c.shape[0] == 1 and batch > 1:
            c = c.expand(batch, -1)
        return c.contiguous()

    # -- dynamic filtering --------------------------------------------------
    def _split_params(self, params: torch.Tensor):
        """Split the controller output ``[B, n_weight + n_bias]`` into per-layer (weight, bias) lists."""
        weights, biases = params[:, :self._n_weight], params[:, self._n_weight:]
        w_layers, b_layers = [], []
        wo = bo = 0
        for cout, cin in self._layer_shapes:
            nw, nb = cout * cin, cout
            w_layers.append(weights[:, wo:wo + nw]); wo += nw
            b_layers.append(biases[:, bo:bo + nb]); bo += nb
        return w_layers, b_layers

    def _apply_dynamic(self, feat: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply the per-sample generated 1×1 conv stack to ``feat`` [B, mid, H, W] via grouped conv.

        DoDNet's trick: stack the batch along the channel dim and run a grouped conv with ``groups=B`` so
        each sample is filtered by its own generated weights in a single conv call.
        """
        b, c, h, w = feat.shape
        w_layers, b_layers = self._split_params(params)
        x = feat.reshape(1, b * c, h, w)                       # [1, B*mid, H, W]
        for li, (cout, cin) in enumerate(self._layer_shapes):
            wt = w_layers[li].reshape(b * cout, cin, 1, 1)     # grouped weight: (B*cout, cin, 1, 1)
            bs = b_layers[li].reshape(b * cout)
            x = F.conv2d(x, wt, bias=bs, stride=1, padding=0, groups=b)
            if li < len(self._layer_shapes) - 1:
                x = F.relu(x)
        out = x.reshape(b, self.num_classes, h, w)
        return out

    def forward(self, pyramid: list[torch.Tensor], out_hw) -> torch.Tensor:
        self.aux_logits = []
        feat = self.pre(self.trunk(pyramid))                   # [B, mid, H/4, W/4]
        b = feat.shape[0]
        if self.mechanism == "dynamic":
            code = self._resolve_code(b, feat.device, feat.dtype)
            params = self.controller(code)                     # [B, n_weight + n_bias]
            logits = self._apply_dynamic(feat, params)
        else:  # film_moe
            code = self._resolve_code(b, feat.device, feat.dtype)
            gamma, beta = self.film(code)                      # [B, mid]
            feat = gamma.view(b, -1, 1, 1) * feat + beta.view(b, -1, 1, 1)
            logits = self.static_head(feat)
        return resize_to(logits, out_hw)


# --------------------------------------------------------------------------- #
# Registry hook (additive; segmentation_training/models/decoders.py is not modified)
# --------------------------------------------------------------------------- #
def _build_dodnet(spec, in_channels, strides, num_classes):
    p = dict(getattr(spec, "params", {}) or {})
    return DoDNetHead(in_channels, strides, num_classes,
                      channels=int(p.get("channels", 256)),
                      n_organelles=int(p.get("n_organelles", 2)),
                      mid_channels=int(p.get("mid_channels", 8)),
                      n_dynamic=int(p.get("n_dynamic", 3)),
                      mechanism=str(p.get("mechanism", "dynamic")))


def register_dodnet() -> None:
    """Register the ``dodnet`` decoder into ``models.decoders.DECODERS`` (idempotent). Called at import."""
    from ...models.decoders import DECODERS
    DECODERS.setdefault("dodnet", _build_dodnet)


# Register at import so ``build_decoder({"type": "dodnet"}, ...)`` works as soon as this module is imported.
register_dodnet()
