"""LoRA adapters, installed as forward hooks inside the frozen encoder.

Ported from ``segmentation_training/harness/adapters.py``. All four OmniEM models use
``adapt: lora`` with ``rank: 8, conv: false`` — i.e. plain low-rank LoRA, not Conv-LoRA. The
conv branch is kept because the stored ``adapt_params`` record it explicitly and a future head
could enable it.

The hook splits the block's token output into prefix tokens (CLS + register/storage) and patch
tokens; a wrong prefix count silently corrupts features, so it is asserted against the spec at
build time rather than inferred loosely. timm reports it as ``num_prefix_tokens``:
5 for QuantEM (1 CLS + 4 storage), 1 for OmniEM (DINOv2, CLS only).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConvLoRAAdapter(nn.Module):
    """Low-rank residual on patch tokens: ``Linear(D->r) -> [depthwise conv] -> Linear(r->D)``.

    The up-projection is zero-initialised, so the adapter is the identity at init.
    """

    def __init__(self, dim: int, n_prefix: int, rank: int = 8, conv: bool = True, ksize: int = 3):
        super().__init__()
        self.n_prefix = int(n_prefix)
        self.rank = int(rank)
        self.conv_enabled = bool(conv)
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)
        if conv:
            self.dwconv = nn.Conv2d(rank, rank, ksize, padding=ksize // 2, groups=rank, bias=True)
            nn.init.zeros_(self.dwconv.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        prefix, patches = tokens[:, : self.n_prefix, :], tokens[:, self.n_prefix :, :]
        b, p, _ = patches.shape
        z = self.act(self.down(patches))
        if self.conv_enabled:
            g = int(round(math.sqrt(p)))
            if g * g == p:  # square grid only; non-square skips the conv
                zc = z.transpose(1, 2).reshape(b, self.rank, g, g)
                z = self.dwconv(zc).reshape(b, self.rank, p).transpose(1, 2)
        return torch.cat([prefix, patches + self.up(z)], dim=1)


def install_conv_lora(encoder, *, rank: int = 8, conv: bool = False, blocks=None):
    """Attach adapters to ``encoder.backbone.blocks[i]`` and return them as an ``nn.ModuleList``.

    Idempotent per encoder. The base backbone stays frozen; only adapter params are trainable.
    """
    if getattr(encoder, "_conv_lora", None) is not None:
        return encoder._conv_lora

    bb = encoder.backbone
    depth = len(bb.blocks)
    if blocks is None:
        blocks = list(range(depth))

    n_prefix = int(getattr(bb, "num_prefix_tokens", 1))
    expected = encoder.spec.n_prefix_tokens
    if n_prefix != expected:
        raise RuntimeError(
            f"prefix-token mismatch installing LoRA: backbone reports {n_prefix}, spec expects "
            f"{expected}. Token splitting would be silently wrong."
        )

    dim = int(getattr(bb, "embed_dim", 0) or encoder.embedding_dim)
    adapters, handles = nn.ModuleList(), []
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


def unfreeze_last_n_blocks(encoder, n: int) -> None:
    """``adapt: last_n`` — mark the last ``n`` blocks trainable.

    Only flips ``requires_grad``; the *values* always come from the released head artifact, which
    stores those blocks under ``encoder_trainable``.
    """
    blocks = encoder.backbone.blocks
    depth = len(blocks)
    n = max(0, min(int(n), depth))
    for i in range(depth - n, depth):
        for p in blocks[i].parameters():
            p.requires_grad_(True)


def apply_adaptation(encoder, mode: str, params: dict | None = None) -> bool:
    """Configure encoder-side adaptation. Returns whether any encoder params are trainable.

    ``lora`` is the only mode that changes the module graph; ``last_n`` and ``full`` only flip
    ``requires_grad``, because their weights arrive by name-matched copy from the artifact.
    """
    params = params or {}
    mode = (mode or "frozen").lower()
    if mode == "frozen":
        return False
    if mode == "lora":
        install_conv_lora(
            encoder,
            rank=int(params.get("rank", 8)),
            conv=bool(params.get("conv", False)),
            blocks=params.get("blocks"),
        )
        return True
    if mode == "last_n":
        unfreeze_last_n_blocks(encoder, int(params.get("n", 4)))
        return True
    if mode == "full":
        for p in encoder.backbone.parameters():
            p.requires_grad_(True)
        return True
    raise ValueError(f"unknown adaptation mode {mode!r}")
