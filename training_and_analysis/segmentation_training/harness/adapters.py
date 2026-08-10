"""Encoder-side LoRA adapters: static Conv-LoRA (``adapt="lora"``/``"lora_ln"``) and the conditional
variant (``adapt="cond_lora"``). These are the adaptation modes that add trainable encoder params;
``last_n`` and ``full`` instead unfreeze base weights and are handled in ``hooks.encoder_adaptation``.

Injects low-rank conv residuals inside the frozen ViT (base weights stay frozen) via forward hooks
on selected transformer blocks. Each hook takes the block's token output ``[B, n_prefix + P, D]``,
splits off the prefix tokens (CLS + storage/register), reshapes the P patch tokens to their square
grid, applies a low-rank down -> depthwise-conv -> up residual, and adds it back. Framework-agnostic
(works for both encoder frameworks the harness loads, ``dinov3`` and ``timm_vit``); only the adapter
params are trainable.

Reference: Zhong et al., "Convolution Meets LoRA" (Conv-LoRA), ICLR 2024. The bottleneck conv is what
makes it *Conv*-LoRA rather than plain LoRA; ``conv=False`` falls back to linear LoRA.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConvLoRAAdapter(nn.Module):
    """Low-rank residual on patch tokens: Linear(D->r) -> [depthwise Conv on the r-grid] -> Linear(r->D).

    Zero-initialised up-projection so the adapter starts as identity (training only departs from the
    frozen encoder as it learns), matching LoRA practice.
    """

    def __init__(self, dim: int, n_prefix: int, rank: int = 8, conv: bool = True, ksize: int = 3):
        super().__init__()
        self.n_prefix = int(n_prefix)
        self.rank = int(rank)
        self.conv_enabled = bool(conv)
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)  # identity at init
        if conv:
            self.dwconv = nn.Conv2d(rank, rank, ksize, padding=ksize // 2, groups=rank, bias=True)
            nn.init.zeros_(self.dwconv.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        prefix, patches = tokens[:, : self.n_prefix, :], tokens[:, self.n_prefix:, :]
        b, p, d = patches.shape
        z = self.act(self.down(patches))  # [B, P, r]
        if self.conv_enabled:
            g = int(round(math.sqrt(p)))
            if g * g == p:
                zc = z.transpose(1, 2).reshape(b, self.rank, g, g)
                zc = self.dwconv(zc)
                z = zc.reshape(b, self.rank, p).transpose(1, 2)
        delta = self.up(z)  # [B, P, D], zero at init
        patches = patches + delta
        return torch.cat([prefix, patches], dim=1)


class CondLoRAAdapter(nn.Module):
    """Conditional Conv-LoRA: the low-rank bottleneck ``z`` is modulated by a per-image (or per-source)
    code ``c`` — a different mechanism from static LoRA (fixed adapters) and from image-style conditioning
    FiLM, which conditions neck/decoder norms rather than the encoder. ``set_code(c)`` before forward.

    modes:
      * ``hyper`` — hypernetwork-lite: ``c -> (gamma, beta)`` FiLM the r-dim bottleneck ``z' = (1+gamma)z+beta``.
      * ``moe``   — K expert up-projections; a code-driven softmax gate mixes them per-input.
    The output path is zero-initialised so the adapter is the identity at init (``up``/experts zero;
    the conditioning heads ``mod``/``gate`` keep zero bias and small-normal weights) → cond-LoRA starts
    byte-identical to the frozen encoder + a static-LoRA path, and departs from it only as it learns;
    with ``c is None`` it degrades to a plain static LoRA residual (safe eval fallback).
    """

    def __init__(self, dim: int, n_prefix: int, rank: int = 8, conv: bool = True, ksize: int = 3,
                 cond_dim: int = 64, mode: str = "hyper", n_experts: int = 4):
        super().__init__()
        self.n_prefix = int(n_prefix)
        self.rank = int(rank)
        self.conv_enabled = bool(conv)
        self.mode = str(mode)
        self.n_experts = max(1, int(n_experts))
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        if conv:
            self.dwconv = nn.Conv2d(rank, rank, ksize, padding=ksize // 2, groups=rank, bias=True)
            nn.init.zeros_(self.dwconv.bias)
        if self.mode == "moe":
            self.up = nn.ModuleList([nn.Linear(rank, dim, bias=False) for _ in range(self.n_experts)])
            for u in self.up:
                nn.init.zeros_(u.weight)                       # experts identity at init
            self.gate = nn.Linear(cond_dim, self.n_experts)
            nn.init.normal_(self.gate.weight, std=0.1); nn.init.zeros_(self.gate.bias)   # mild-diverse routing (not zero:
            #   identity-at-init is carried by the zero experts; a non-zero gate lets routing be expressive the moment the
            #   experts depart, so a null result means "conditioning does not help", not "gate under-trained").
        else:  # hyper
            self.up = nn.Linear(rank, dim, bias=False)
            nn.init.zeros_(self.up.weight)                     # identity at init (carries identity alone)
            self.mod = nn.Linear(cond_dim, 2 * rank)
            nn.init.normal_(self.mod.weight, std=0.05); nn.init.zeros_(self.mod.bias)   # gamma small-nonzero at init: z is
            #   mildly modulated but delta=up(z')=0 (up zero) so the output is still the identity — this keeps the
            #   conditioning head expressive from the first step at which up departs, avoiding an "inert" confound.
        self._code: torch.Tensor | None = None

    def set_code(self, c: torch.Tensor | None) -> None:
        self._code = c

    def _bottleneck(self, patches: torch.Tensor) -> torch.Tensor:
        b, p, _ = patches.shape
        z = self.act(self.down(patches))                      # [B, P, r]
        if self.conv_enabled:
            g = int(round(math.sqrt(p)))
            if g * g == p:
                zc = z.transpose(1, 2).reshape(b, self.rank, g, g)
                zc = self.dwconv(zc)
                z = zc.reshape(b, self.rank, p).transpose(1, 2)
        return z

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        prefix, patches = tokens[:, : self.n_prefix, :], tokens[:, self.n_prefix:, :]
        b = patches.shape[0]
        z = self._bottleneck(patches)                         # [B, P, r]
        c = self._code
        if self.mode == "moe":
            if c is not None:
                w = torch.softmax(self.gate(c), dim=-1)        # [B, K]
                stacked = torch.stack([u(z) for u in self.up], dim=-1)   # [B, P, D, K]
                delta = (stacked * w.view(b, 1, 1, self.n_experts)).sum(-1)
            else:
                delta = self.up[0](z)                          # unconditioned fallback
        else:  # hyper
            if c is not None:
                gb = self.mod(c)                               # [B, 2r]
                gamma, beta = gb[:, : self.rank], gb[:, self.rank:]
                z = (1.0 + gamma).unsqueeze(1) * z + beta.unsqueeze(1)   # per-image modulation of z
            delta = self.up(z)                                 # [B, P, D], zero at init
        return torch.cat([prefix, patches + delta], dim=1)


class CondLoRAController(nn.Module):
    """Computes a per-image (``image`` -> StyleEncoder) or per-source (``source`` -> Embedding) code and
    broadcasts it to every block's CondLoRAAdapter before the encoder forward. Owns the adapters + the code
    net so all cond-LoRA params register with the encoder (train at ``adapter_lr``)."""

    def __init__(self, adapters: nn.ModuleList, *, cond_source: str = "image", cond_dim: int = 64,
                 source_vocab: int | None = None):
        super().__init__()
        self.adapters = adapters
        self.cond_source = str(cond_source)
        if self.cond_source == "source":
            self.embed = nn.Embedding(max(2, int(source_vocab or 64)), cond_dim)
        else:
            from ..models.conditioning.style_encoder import StyleEncoder
            self.style = StyleEncoder(style_dim=cond_dim, use_stats=True, feat_dim=0)

    def before_forward(self, image: torch.Tensor, source_ids: torch.Tensor | None = None) -> None:
        if self.cond_source == "source" and source_ids is not None:
            c = self.embed(source_ids.long().view(-1))
        else:
            c = self.style(image)                              # [B, cond_dim]
        # COND_ABLATE: remove or neutralise the per-image code at eval to test
        # whether conditioning does real work or the trained capacity does. 'none' -> static up-path (skip
        # modulation); 'zero' -> constant zero code; 'const' -> batch-mean code; 'shuffle' -> permute codes.
        import os
        ab = os.environ.get("COND_ABLATE", "")
        if ab and c is not None:
            if ab == "none":
                c = None
            elif ab == "zero":
                c = torch.zeros_like(c)
            elif ab == "const":
                c = c.mean(0, keepdim=True).expand_as(c)
            elif ab == "shuffle" and c.shape[0] > 1:
                c = c[torch.randperm(c.shape[0], device=c.device)]
        for ad in self.adapters:
            ad.set_code(c)


def install_cond_lora(encoder, *, rank: int = 8, conv: bool = True, blocks: list[int] | None = None,
                      mode: str = "hyper", cond_source: str = "image", cond_dim: int = 64,
                      n_experts: int = 4, source_vocab: int | None = None):
    """Attach CondLoRAAdapters to every block + a CondLoRAController (image/source code net). Idempotent.
    The controller is stored at ``encoder._cond_lora_ctrl``; ``SegModel.forward`` calls its ``before_forward``
    to set the per-input code before the backbone runs."""
    if getattr(encoder, "_cond_lora_ctrl", None) is not None:
        return encoder._cond_lora_ctrl
    bb = encoder.backbone
    depth = len(bb.blocks)
    if blocks is None:
        blocks = list(range(depth))
    n_prefix = _infer_n_prefix(encoder)
    dim = int(getattr(bb, "embed_dim", 0) or encoder.embedding_dim)  # block-output channel dim
    adapters = nn.ModuleList()
    handles = []
    for i in blocks:
        ad = CondLoRAAdapter(dim, n_prefix=n_prefix, rank=rank, conv=conv, cond_dim=cond_dim,
                             mode=mode, n_experts=n_experts)
        adapters.append(ad)

        def mk(adapter):
            def hook(_module, _inp, out):
                if isinstance(out, tuple):
                    return (adapter(out[0]), *out[1:])
                return adapter(out)
            return hook

        handles.append(bb.blocks[i].register_forward_hook(mk(ad)))
    ctrl = CondLoRAController(adapters, cond_source=cond_source, cond_dim=cond_dim, source_vocab=source_vocab)
    encoder._cond_lora_ctrl = ctrl
    encoder._cond_lora_handles = handles
    for p in ctrl.parameters():
        p.requires_grad_(True)
    return ctrl


def _infer_n_prefix(encoder) -> int:
    """CLS (+ storage/register) token count in front of the patch tokens."""
    if encoder.framework == "dinov3":
        bb = encoder.backbone
        ns = getattr(bb, "n_storage_tokens", None)
        if ns is None and hasattr(bb, "storage_tokens") and bb.storage_tokens is not None:
            ns = int(bb.storage_tokens.shape[1])
        return 1 + int(ns or 0)
    if encoder.framework == "timm_vit":
        # timm ViTs expose num_prefix_tokens (1 CLS + any register/storage tokens: OmniEM/dinov2 = 1,
        # Meta-DINOv3 = 1 + register tokens). Returning a flat 1 would mis-split registers as patches.
        npt = getattr(encoder.backbone, "num_prefix_tokens", None)
        return int(npt) if npt is not None else 1
    return 1  # CLS only


def install_conv_lora(encoder, *, rank: int = 8, conv: bool = True, blocks: list[int] | None = None):
    """Attach ConvLoRA adapters to ``encoder.backbone.blocks[i]`` and return the trainable adapter
    modules (as an nn.ModuleList so they register with an owning nn.Module / the optimizer).

    The base backbone stays frozen (``requires_grad_(False)`` from FrozenEncoder). Callers must run
    features with ``grad=True`` so adapter gradients flow. Idempotent per-encoder.
    """
    if getattr(encoder, "_conv_lora", None) is not None:
        return encoder._conv_lora
    bb = encoder.backbone
    depth = len(bb.blocks)
    if blocks is None:
        blocks = list(range(depth))  # every block by default
    n_prefix = _infer_n_prefix(encoder)
    # The hook fires on block outputs, whose channel dim is the backbone's internal embed dim rather
    # than encoder.embedding_dim. The two agree for the ViTs loaded here, so this is normally a no-op.
    dim = int(getattr(bb, "embed_dim", 0) or encoder.embedding_dim)
    adapters = nn.ModuleList()
    handles = []
    for i in blocks:
        ad = ConvLoRAAdapter(dim, n_prefix=n_prefix, rank=rank, conv=conv)
        adapters.append(ad)

        def mk(adapter):
            def hook(_module, _inp, out):
                # Blocks return the token tensor, optionally inside a tuple; handle both.
                if isinstance(out, tuple):
                    return (adapter(out[0]), *out[1:])
                return adapter(out)
            return hook

        handles.append(bb.blocks[i].register_forward_hook(mk(ad)))
    encoder._conv_lora = adapters
    encoder._conv_lora_handles = handles
    for p in adapters.parameters():
        p.requires_grad_(True)
    return adapters
