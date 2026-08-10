"""Encoder-side LoRA adapters, the same implementation the segmentation-training adaptation arms use,
so the adapted probe applies an identical mechanism while the rest of this harness — dataset, decoder,
evaluation — stays unchanged.

Injects low-rank residual adapters inside the (otherwise frozen) ViT via forward hooks on each block.
The hook takes the block's token output ``[B, n_prefix + P, D]``, splits the prefix tokens
(CLS + storage/register), applies a low-rank down -> (optional depthwise conv) -> up residual on the
patch tokens, and adds it back. Framework-agnostic: works for both encoder frameworks the
``FrozenEncoder`` wraps, ``dinov3`` and the external ``timm_vit`` baselines (references only
``encoder.backbone.blocks``, ``encoder.framework``, ``encoder.embedding_dim`` — all present on the
encoder comparison's FrozenEncoder).

The encoder-adaptation experiment's LoRA arms in segmentation training use ``rank=8, conv=False``
(plain low-rank LoRA). Base weights stay frozen; only the adapter params train (at ``adapter_lr``).
Callers must run features with grad so adapter grads flow.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

class ConvLoRAAdapter(nn.Module):
    """Low-rank residual on patch tokens: Linear(D->r) -> [optional depthwise Conv on the r-grid] ->
    Linear(r->D). Zero-initialised up-projection so the adapter starts as identity (training only departs
    from the frozen encoder as it learns), matching LoRA practice. ``conv=False`` = plain linear LoRA."""

    def __init__(self, dim: int, n_prefix: int, rank: int = 8, conv: bool = False, ksize: int = 3):
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
    return 1  # plain ViT: CLS only

def install_conv_lora(encoder, *, rank: int = 8, conv: bool = False, blocks: list[int] | None = None):
    """Attach LoRA adapters to ``encoder.backbone.blocks[i]`` and return them (an nn.ModuleList so they
    register with the owning FrozenEncoder / the optimizer). Base backbone stays frozen; callers must run
    features with grad so adapter gradients flow. Idempotent per-encoder."""
    if getattr(encoder, "_conv_lora", None) is not None:
        return encoder._conv_lora
    bb = encoder.backbone
    depth = len(bb.blocks)
    if blocks is None:
        blocks = list(range(depth))  # every block by default
    n_prefix = _infer_n_prefix(encoder)
    dim = int(getattr(bb, "embed_dim", 0) or encoder.embedding_dim)
    adapters = nn.ModuleList()
    handles = []
    for i in blocks:
        ad = ConvLoRAAdapter(dim, n_prefix=n_prefix, rank=rank, conv=conv)
        adapters.append(ad)

        def mk(adapter):
            def hook(_module, _inp, out):
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

def remove_conv_lora(encoder) -> None:
    """Remove installed LoRA adapters + their forward hooks (so a fresh install trains from scratch)."""
    for h in getattr(encoder, "_conv_lora_handles", []) or []:
        try:
            h.remove()
        except Exception:
            pass
    if hasattr(encoder, "_conv_lora"):
        del encoder._conv_lora
    if hasattr(encoder, "_conv_lora_handles"):
        del encoder._conv_lora_handles
