"""Load a trained segmentation head (an adaptation or conditioning arm) back onto a frozen encoder -> a ready-to-eval ``SegModel``.

A saved ``head.pt`` holds ``{neck, decoder, encoder_trainable, adapters, conditioner, meta_vocab}``;
``build_and_load_head`` reconstructs the full stack: installs the LoRA adapters (or unfreezes base blocks)
per ``cfg.encoder.adapt``, attaches the image-style conditioning conditioner if the head carries one, and
loads every state dict. This is the loader the test-time arms, the parallel evaluation workers and the
experiment packages' adapted base all go through.

Runs on CPU (torch + the harness only).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .meta import MetaVocab
from .train import build_segmodel


def _load_named_params(module, state: dict) -> tuple[int, int]:
    """Copy ``{name: tensor}`` into ``module``'s params by name (shape-checked). Returns (loaded, skipped).

    Works for both the LoRA arm (``_conv_lora.*`` keys) and last_n/full (``backbone.blocks.*`` base keys)
    since ``encoder_trainable`` was saved from ``encoder.named_parameters()``.
    """
    own = dict(module.named_parameters())
    loaded = skipped = 0
    for k, v in state.items():
        p = own.get(k)
        if p is not None and tuple(p.shape) == tuple(v.shape):
            with torch.no_grad():
                p.copy_(v.to(p.dtype))
            loaded += 1
        else:
            skipped += 1
    return loaded, skipped


def build_and_load_head(cfg, encoder, head_path, device: str = "cpu", strict: bool = True):
    """Rebuild a ``SegModel`` for ``cfg`` on ``encoder`` and load ``head_path``. Returns (model, vocab, info)."""
    ckpt = torch.load(str(head_path), map_location="cpu", weights_only=False)
    vocab = MetaVocab.from_dict(ckpt.get("meta_vocab"))
    field_sizes = vocab.sizes() if vocab is not None else None

    model = build_segmodel(cfg, encoder, field_sizes=field_sizes)
    info: dict = {}

    model.neck.load_state_dict(ckpt["neck"], strict=strict)
    model.decoder.load_state_dict(ckpt["decoder"], strict=strict)

    # Encoder-side trainable params (LoRA adapters and/or unfrozen base blocks).
    enc_state = ckpt.get("encoder_trainable")
    if enc_state:
        info["encoder_loaded"], info["encoder_skipped"] = _load_named_params(model.encoder, enc_state)
    elif ckpt.get("adapters") is not None and getattr(model.encoder, "_conv_lora", None) is not None:
        model.encoder._conv_lora.load_state_dict(ckpt["adapters"], strict=strict)
        info["encoder_loaded"] = sum(1 for _ in model.encoder._conv_lora.parameters())

    # image-style conditioning conditioner (style encoder + FiLM heads + adversary).
    if ckpt.get("conditioner") is not None:
        if model.conditioner is None:
            raise ValueError("head.pt carries a conditioner but cfg.cond.enabled is False — enable "
                             "conditioning in the config used to load an image-style conditioning head.")
        missing, unexpected = model.conditioner.load_state_dict(ckpt["conditioner"], strict=False)
        info["conditioner_missing"], info["conditioner_unexpected"] = list(missing), list(unexpected)
        model.conditioner.vocab = vocab
        model._meta_vocab = vocab

    return model.to(device).eval(), vocab, info


def inspect_head(head_path) -> dict:
    """Structural summary of a ``head.pt`` (no encoder needed)."""
    ckpt = torch.load(str(head_path), map_location="cpu", weights_only=False)
    out = {"path": str(head_path), "keys": sorted(ckpt.keys())}
    for k in ("neck", "decoder", "encoder_trainable", "adapters", "conditioner"):
        v = ckpt.get(k)
        out[f"n_{k}"] = (len(v) if isinstance(v, dict) else (0 if v is None else 1))
    mv = ckpt.get("meta_vocab")
    out["meta_vocab_fields"] = list(mv.get("fields", [])) if mv else None
    if isinstance(ckpt.get("adapters"), dict) and ckpt["adapters"]:
        k0 = sorted(ckpt["adapters"])[0]
        out["adapter_example"] = {k0: tuple(ckpt["adapters"][k0].shape)}
    return out


def main(argv=None) -> None:
    import argparse
    import json

    from ..config.schema import load_seg_config

    p = argparse.ArgumentParser(description="Load a trained segmentation head onto a frozen encoder.")
    p.add_argument("--config", required=True, help="Resolved config YAML the head was trained with.")
    p.add_argument("--head", required=True, help="head.pt path.")
    p.add_argument("--run-dir", default=None, help="Encoder run dir (checkpoint_index.json). Omit = "
                                                   "structural inspection only (no encoder needed).")
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)

    if not a.run_dir:
        print(json.dumps(inspect_head(a.head), indent=2, default=str))
        return
    from .run_seg import resolve_encoder, resolve_device

    cfg = load_seg_config(a.config)
    cfg.encoder.run_dir = a.run_dir
    device = resolve_device(a.device)
    enc, _ = resolve_encoder(cfg, device)
    enc.to(device)
    model, vocab, info = build_and_load_head(cfg, enc, a.head, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"loaded": True, "adapt": cfg.encoder.adapt, "cond_enabled": cfg.cond.enabled,
                      "conditioner": model.conditioner is not None, "vocab_fields":
                      (vocab.fields if vocab else None), "total_params": n_params, **info},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
