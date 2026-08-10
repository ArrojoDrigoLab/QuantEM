"""LoRA adapters injected inside a frozen ViT.

Not included: the conditional-adapter variants (``CondLoRAAdapter`` /
``CondLoRAController`` / ``install_cond_lora``), which belong to an ablation no
released pack uses, and the SAM3 branch of the prefix inference.

Reference: Zhong et al., "Convolution Meets LoRA" (Conv-LoRA), ICLR 2024. The
depthwise bottleneck conv is what makes it *Conv*-LoRA; ``conv=False`` is plain
low-rank LoRA.

**The four OmniEM packs use** ``conv=False`` (``adapt_params: {rank: 8, conv:
false}``), which their checkpoints confirm: each carries 48 adapter tensors =
24 blocks x {``down.weight``, ``up.weight``}, with no ``dwconv`` entries. So
``ConvLoRAAdapter.dwconv`` is never constructed for a released pack; it is kept
because the class must stay shape-compatible with any future pack that sets
``conv: true``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

#: Encoder adaptation modes the released packs use, plus the no-op.
ADAPT_MODES = ("frozen", "lora", "lora_ln", "last_n", "full")


class ConvLoRAAdapter(nn.Module):
    """Low-rank residual on patch tokens: ``Linear(D->r) -> [dwconv] -> Linear(r->D)``.

    The up-projection is zero-initialised so the adapter is the identity at
    init. That matters here for a reason beyond training practice: it means an
    adapter tensor that fails to load leaves the encoder *unchanged* rather than
    randomly perturbed, so a silent load failure degrades to the base encoder
    instead of to noise. The loader still checks names and shapes.
    """

    def __init__(self, dim: int, n_prefix: int, rank: int = 8, conv: bool = True, ksize: int = 3) -> None:
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
            assert self.dwconv.bias is not None  # bias=True above; torch types it Optional
            nn.init.zeros_(self.dwconv.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        prefix, patches = tokens[:, : self.n_prefix, :], tokens[:, self.n_prefix :, :]
        b, p, _ = patches.shape
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


def infer_n_prefix(encoder: Any) -> int:
    """Count of CLS + register/storage tokens ahead of the patch tokens.

    Getting this wrong does not raise -- it just treats register tokens as
    patches, so the adapter's token split is off by a few and the residual lands
    on the wrong rows. Both supported backbones report it explicitly:
    DINOv3 via ``n_storage_tokens`` (4 for the QuantEM ViT-B), timm via
    ``num_prefix_tokens`` (1 for the OmniEM DINOv2 ViT-L).
    """
    bb = encoder.backbone
    if encoder.framework == "dinov3":
        ns = getattr(bb, "n_storage_tokens", None)
        if ns is None and getattr(bb, "storage_tokens", None) is not None:
            ns = int(bb.storage_tokens.shape[1])
        return 1 + int(ns or 0)
    npt = getattr(bb, "num_prefix_tokens", None)
    return int(npt) if npt is not None else 1


def install_conv_lora(
    encoder: Any,
    *,
    rank: int = 8,
    conv: bool = True,
    blocks: list[int] | None = None,
) -> nn.ModuleList:
    """Attach adapters to ``encoder.backbone.blocks[i]`` via forward hooks.

    The adapters are stored at ``encoder._conv_lora`` (an ``nn.ModuleList``), so
    they register under the names ``_conv_lora.<i>.{down,up}.weight`` -- which is
    exactly how the released heads serialise them. Idempotent per encoder.
    """
    if getattr(encoder, "_conv_lora", None) is not None:
        return encoder._conv_lora
    bb = encoder.backbone
    depth = len(bb.blocks)
    if blocks is None:
        blocks = list(range(depth))
    n_prefix = infer_n_prefix(encoder)
    dim = int(getattr(bb, "embed_dim", 0) or encoder.embedding_dim)
    adapters = nn.ModuleList()
    handles = []
    for i in blocks:
        ad = ConvLoRAAdapter(dim, n_prefix=n_prefix, rank=rank, conv=conv)
        adapters.append(ad)

        def mk(adapter: nn.Module) -> Callable[..., Any]:
            def hook(_module: Any, _inp: Any, out: Any) -> Any:
                # DINOv3/timm blocks return the token tensor, sometimes in a tuple.
                if isinstance(out, tuple):
                    return (adapter(out[0]), *out[1:])
                return adapter(out)

            return hook

        handles.append(bb.blocks[i].register_forward_hook(mk(ad)))
    encoder._conv_lora = adapters
    encoder._conv_lora_handles = handles
    return adapters


def apply_adaptation(encoder: Any, mode: str = "frozen", params: dict | None = None) -> bool:
    """Create whatever encoder-side parameters ``mode`` implies. Returns whether any exist.

    Inference never trains, so ``requires_grad`` is irrelevant here; what
    matters is that the right *parameters exist to load into*. ``lora`` must
    install the adapter modules before the head's adapter tensors can be copied
    in, and ``last_n``/``full`` need no construction at all because the head
    simply overwrites base block weights that are already there.
    """
    params = params or {}
    mode = (mode or "frozen").lower()
    if mode == "frozen":
        return False
    if mode in ("lora", "conv_lora", "lora_ln"):
        install_conv_lora(
            encoder,
            rank=int(params.get("rank", 8)),
            conv=bool(params.get("conv", True)),
            blocks=params.get("blocks"),
        )
        return True
    if mode in ("last_n", "full"):
        # The head carries replacement weights for base backbone parameters that
        # already exist; nothing to construct.
        return True
    raise ValueError(f"unknown encoder.adapt mode {mode!r} (expected one of {ADAPT_MODES})")
