"""Assemble a ``SegModel`` for a config and load a released ``head.pt`` into it.

Not included: the style-conditioner and ``MetaVocab`` wiring (all eight released
heads carry ``conditioner: None`` and ``meta_vocab: None``) and the argparse
bring-up CLI.

One deliberate strictness choice: **a partial load is an error here, not a
warning.** A loader that merely counts skipped tensors will, on a name typo or a
shape drift, yield a model that runs, produces plausible-looking probabilities,
and is not the published model. Where metrics are being watched that is
recoverable, because they collapse visibly; here the output goes straight into a
figure, so :func:`build_and_load_head` raises instead and names the offending
tensors. Pass ``strict=False`` only to inspect a checkpoint you already know is
mismatched.

A released ``head.pt`` is a dict of state dicts::

    {"neck": ..., "decoder": ..., "encoder_trainable": ..., "adapters": ...,
     "conditioner": None, "meta_vocab": None}

``encoder_trainable`` is keyed against ``encoder.named_parameters()``, so it is
``_conv_lora.<i>.{down,up}.weight`` for the LoRA packs and
``backbone.blocks.<i>....`` for the ``last_n`` / ``full`` packs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .adapters import apply_adaptation
from .base import SegModel
from .decoders import build_decoder
from .necks import build_neck
from .schema import HeadConfig

logger = logging.getLogger(__name__)


class HeadLoadError(RuntimeError):
    """A released head did not load cleanly into the rebuilt architecture."""


def build_segmodel(cfg: HeadConfig, encoder: Any) -> SegModel:
    """Wire ``encoder`` + neck + decoder into a :class:`SegModel` for ``cfg``.

    An exported encoder (``embeds_pack_state``) skips adaptation: its adapters
    and replaced blocks are already inside the traced graph, and it has no
    ``.backbone.blocks`` to install hooks on anyway.
    """
    layers = cfg.encoder.resolved_layers(encoder.depth)
    neck: Any = build_neck(
        cfg.neck,
        encoder.embedding_dim,
        len(layers),
        encoder.patch_size,
        out_channels=cfg.neck_out_channels,
    )
    decoder = build_decoder(cfg.decoder, neck.out_channels, neck.strides, cfg.num_classes)
    if not getattr(encoder, "embeds_pack_state", False):
        apply_adaptation(encoder, cfg.encoder.adapt, cfg.encoder.adapt_params or {})
    return SegModel(encoder, neck, decoder, layers)


def _load_named_params(module: nn.Module, state: dict) -> list[str]:
    """Copy ``{name: tensor}`` into ``module``'s parameters and buffers by name.

    Returns the names that could **not** be placed (absent, or a shape
    mismatch). Buffers are consulted as well as parameters because the
    timm-built QuantEM encoder registers the attention k-bias as a *buffer*
    (timm's Eva design), while an HF-installed head carries trained values for
    it; parameters win when a name exists as both.
    """
    own = dict(module.named_buffers())
    own.update(dict(module.named_parameters()))
    unplaced: list[str] = []
    for k, v in state.items():
        p = own.get(k)
        if p is not None and tuple(p.shape) == tuple(v.shape):
            with torch.no_grad():
                p.copy_(v.to(p.dtype))
        else:
            unplaced.append(k)
    return unplaced


def build_and_load_head(
    cfg: HeadConfig,
    encoder: Any,
    head_path: str | Path,
    device: str = "cpu",
    strict: bool = True,
) -> tuple[SegModel, dict]:
    """Rebuild the architecture for ``cfg`` on ``encoder`` and load ``head_path``.

    Returns ``(model, info)`` with the model in ``eval()`` on ``device``.

    Raises:
        HeadLoadError: when ``strict`` and any tensor in the checkpoint could not
            be placed, or the checkpoint carries a conditioner this build cannot
            reproduce.
    """
    head_path = Path(head_path)
    # weights_only=False: the file is a plain dict of tensors, but it was written
    # by torch.save without the safetensors wrapper, and the caller has already
    # verified its sha256 through the registry cache before we get here.
    ckpt = torch.load(str(head_path), map_location="cpu", weights_only=False)

    if ckpt.get("conditioner") is not None:
        raise HeadLoadError(
            f"{head_path.name} carries an E1 style conditioner, which is not vendored "
            "(no released pack uses one). Refusing to load it without the conditioning stack."
        )

    model = build_segmodel(cfg, encoder)
    info: dict[str, object] = {
        "neck": cfg.neck.type,
        "decoder": cfg.decoder.type,
        "adapt": cfg.encoder.adapt,
        "layers": list(model.layers),
    }

    problems: list[str] = []
    for part in ("neck", "decoder"):
        missing, unexpected = getattr(model, part).load_state_dict(ckpt[part], strict=False)
        info[f"{part}_tensors"] = len(ckpt[part])
        if missing or unexpected:
            problems.append(
                f"{part}: missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}"
            )

    # Encoder-side parameters: LoRA adapters, or replaced base blocks. An
    # exported encoder already contains them -- see build_segmodel.
    enc_state = ckpt.get("encoder_trainable")
    # A QuantEM pack whose index is the research tree's carries these tensors in
    # DINOv3's naming, and its encoder may nonetheless have been built through
    # timm (see quantem.inference.encoders.build_encoder). The builder attaches
    # the translation; applying it here rather than inside _load_named_params
    # keeps the pending_overlay check below comparing like with like. Without
    # it every fine-tuned block would go unplaced -- which strict mode does
    # catch, loudly, but only because someone wrote this line.
    overlay_remap = getattr(model.encoder, "overlay_remap", None)
    if enc_state and overlay_remap is not None:
        enc_state = overlay_remap(enc_state)
    exported = bool(getattr(model.encoder, "embeds_pack_state", False))
    if exported:
        info["encoder_state"] = "baked into the exported encoder"
    elif enc_state:
        unplaced = _load_named_params(model.encoder, enc_state)
        info["encoder_tensors"] = len(enc_state)
        info["encoder_unplaced"] = len(unplaced)
        if unplaced:
            problems.append(f"encoder: {len(unplaced)} unplaced, e.g. {unplaced[:6]}")
    elif (
        ckpt.get("adapters") is not None and getattr(model.encoder, "_conv_lora", None) is not None
    ):
        missing, unexpected = model.encoder._conv_lora.load_state_dict(
            ckpt["adapters"], strict=False
        )
        info["encoder_tensors"] = len(ckpt["adapters"])
        if missing or unexpected:
            problems.append(
                f"adapters: missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}"
            )

    # A weight source that ships without the pack's fine-tuned blocks records
    # them as pending_overlay (see encoders.build_quantem_timm_encoder: the HF
    # trunk carries blocks 0-7, the head carries 8-11). Every one of them must
    # have been covered by the head just loaded: a block left at random init
    # produces a plausible-looking segmentation that is not the published
    # model, which is precisely the silent failure this loader exists to
    # refuse.
    pending = list(getattr(model.encoder, "pending_overlay", ()) or ())
    if pending and not exported:
        uncovered = sorted(set(pending) - set(enc_state or ()))
        if uncovered:
            problems.append(
                f"encoder: the weight source left {len(pending)} tensors for the head "
                f"to provide and {len(uncovered)} were not, e.g. {uncovered[:4]}"
            )

    if problems:
        message = (
            f"{head_path.name} did not load cleanly into a "
            f"{cfg.neck.type} neck + {cfg.decoder.type} decoder "
            f"(adapt={cfg.encoder.adapt}): " + "; ".join(problems)
        )
        if strict:
            raise HeadLoadError(
                message + ". The rebuilt architecture does not match this checkpoint, so "
                "its output would not be the published model."
            )
        logger.warning("%s -- continuing because strict=False", message)
        info["problems"] = problems

    return model.to(device).eval(), info


def inspect_head(head_path: str | Path) -> dict:
    """Structural summary of a ``head.pt`` without building anything.

    Useful for checking a downloaded pack against
    :data:`quantem.registry.manifest.ARCHITECTURE` before paying for an encoder.
    """
    ckpt = torch.load(str(head_path), map_location="cpu", weights_only=False)
    out: dict[str, object] = {"path": str(head_path), "keys": sorted(ckpt.keys())}
    for k in ("neck", "decoder", "encoder_trainable", "adapters", "conditioner"):
        v = ckpt.get(k)
        out[f"n_{k}"] = len(v) if isinstance(v, dict) else (0 if v is None else 1)
    neck = ckpt.get("neck")
    if isinstance(neck, dict) and neck:
        # The tap-fuse input width is embed_dim * n_taps, which identifies the encoder.
        for key in ("fuse.0.weight", "tap_fuse.0.weight"):
            if key in neck:
                out["tap_fuse_in_channels"] = int(neck[key].shape[1])
                break
    return out
