"""Encoder adaptation dispatcher: maps ``cfg.adapt`` to which encoder parameters train, on the
``FrozenEncoder`` this harness wraps. The mechanism is the same one the segmentation-training
adaptation experiments use, so the two are directly comparable.

Modes (``cfg.adapt``; ``cfg.adapt_params`` is the per-mode dict):
  * ``frozen``  — no trainable encoder params (the default).
  * ``lora``    — low-rank token adapters inside every block (base weights frozen). ``adapt_params``:
                  ``rank`` (default 8); ``conv`` (default False, plain LoRA, as the adaptation arms used);
                  ``blocks`` (default None, every block).
  * ``lora_ln`` — ``lora`` + trainable LayerNorm affines.
  * ``last_n``  — unfreeze the last N transformer blocks' base weights (``adapt_params.n``, default 4).
  * ``full``    — unfreeze all encoder base weights.

Returns ``encoder_trainable`` (bool): True when any encoder-side params train, so the caller runs the
backbone forward with grad and collects the newly-``requires_grad`` params into the optimizer at
``cfg.adapter_lr`` (the ``last_n`` arm sets that an order of magnitude below the ``lora`` arm, since it
unfreezes whole block weights). The base backbone stays in ``.eval()`` regardless; only
``requires_grad`` flips here.
"""

from __future__ import annotations

ADAPT_MODES = ("frozen", "lora", "lora_ln", "last_n", "full")

def apply_adaptation(encoder, mode: str = "frozen", params: dict | None = None) -> bool:
    """Configure which encoder params train for ``mode``; return whether any do (encoder_trainable)."""
    params = params or {}
    mode = (mode or "frozen").lower()
    if mode == "frozen":
        return False
    if mode in ("lora", "conv_lora", "lora_ln"):
        from .adapters import install_conv_lora

        install_conv_lora(
            encoder,
            rank=int(params.get("rank", 8)),
            conv=bool(params.get("conv", False)),
            blocks=params.get("blocks"),
        )
        if mode == "lora_ln":
            _unfreeze_layernorms(encoder)
        return True
    if mode == "last_n":
        _unfreeze_last_n_blocks(encoder, int(params.get("n", 4)))
        return True
    if mode == "full":
        for p in encoder.backbone.parameters():
            p.requires_grad_(True)
        return True
    raise ValueError(f"unknown adapt mode {mode!r} (expected one of {ADAPT_MODES})")

def _unfreeze_last_n_blocks(encoder, n: int) -> None:
    blocks = encoder.backbone.blocks
    depth = len(blocks)
    n = max(0, min(int(n), depth))
    for i in range(depth - n, depth):
        for p in blocks[i].parameters():
            p.requires_grad_(True)

def _unfreeze_layernorms(encoder) -> None:
    import torch.nn as nn

    for m in encoder.backbone.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
