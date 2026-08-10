"""Hook — encoder adaptation.

Neck, decoder and loss training keeps the encoder frozen; encoder adaptation instead selects an
adaptation mode via ``cfg.encoder.adapt``. This module centralises the mode -> (which encoder params
train) policy. Called from ``segmentation_training.harness.train.build_segmodel``.

Modes (``ADAPT_MODES``, read from ``cfg.encoder.adapt``; ``adapt_params`` is the per-mode dict):
  * ``frozen``  — no trainable encoder params (the default for neck, decoder and loss training).
  * ``lora``    — low-rank token adapters inside every block (base weights frozen). ``adapt_params``:
                  ``rank`` (default 8); ``conv`` (default True = Conv-LoRA;
                  ``False`` = plain low-rank LoRA, the non-convolutional control that isolates the
                  locality prior); ``blocks`` (default all).
  * ``lora_ln`` — ``lora`` + trainable LayerNorm affines.
  * ``cond_lora`` — LoRA whose bottleneck is modulated per input rather than static. ``adapt_params``
                  adds ``mode`` (``hyper`` | ``moe``), ``cond_source`` (``image`` | ``source``),
                  ``cond_dim`` (default 64), ``n_experts`` (default 4) and ``source_vocab``;
                  ``conv`` defaults to False here.
  * ``last_n``  — unfreeze the last N transformer blocks' base weights (``adapt_params.n``, default 4).
  * ``full``    — unfreeze all encoder base weights.

``conv_lora`` is accepted as an alias for ``lora``.

Returns ``encoder_trainable`` (bool): True when any encoder-side params train, so ``SegModel`` runs the
backbone forward with grad and ``train._param_groups`` collects the newly-``requires_grad`` params (at
``cfg.optim.adapter_lr``, which is kept small for ``last_n``/``full``). The base backbone stays in ``.eval()``
(set by the trainer) regardless — only ``requires_grad`` flips here. The adapter installers
(``install_conv_lora``, ``install_cond_lora``) are idempotent per encoder, so re-entry is safe.
"""

from __future__ import annotations

ADAPT_MODES = ("frozen", "lora", "lora_ln", "last_n", "full", "cond_lora")


def apply_adaptation(encoder, mode: str = "frozen", params: dict | None = None) -> bool:
    """Configure which encoder params train for ``mode``; return whether any do (encoder_trainable)."""
    params = params or {}
    mode = (mode or "frozen").lower()
    if mode == "frozen":
        return False
    if mode in ("lora", "conv_lora", "lora_ln"):
        from ..harness.adapters import install_conv_lora

        install_conv_lora(
            encoder,
            rank=int(params.get("rank", 8)),
            conv=bool(params.get("conv", True)),
            blocks=params.get("blocks"),
        )
        if mode == "lora_ln":
            _unfreeze_layernorms(encoder)
        return True
    if mode == "cond_lora":
        # Conditional encoder adapters: the LoRA bottleneck is modulated per-image (hypernet/MoE) or
        # per-source. This is a distinct mechanism from static LoRA and from image-style FiLM
        # conditioning, which acts on the neck and decoder norms.
        from ..harness.adapters import install_cond_lora

        install_cond_lora(
            encoder,
            rank=int(params.get("rank", 8)),
            conv=bool(params.get("conv", False)),
            blocks=params.get("blocks"),
            mode=str(params.get("mode", "hyper")),          # hyper | moe
            cond_source=str(params.get("cond_source", "image")),  # image | source
            cond_dim=int(params.get("cond_dim", 64)),
            n_experts=int(params.get("n_experts", 4)),
            source_vocab=params.get("source_vocab"),
        )
        return True
    if mode == "last_n":
        _unfreeze_last_n_blocks(encoder, int(params.get("n", 4)))
        return True
    if mode == "full":
        for p in encoder.backbone.parameters():
            p.requires_grad_(True)
        return True
    raise ValueError(
        f"unknown encoder.adapt mode {mode!r} (expected one of {ADAPT_MODES} or 'conv_lora')"
    )


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
