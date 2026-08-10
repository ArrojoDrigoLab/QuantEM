"""Assemble a released model: encoder + neck + decoder, then load its weights.

This mirrors the reference path
``resolve_encoder -> build_segmodel -> build_and_load_head -> predict_region``, with the 145-field
training config replaced by a frozen :class:`~quantem_em.spec.ModelSpec`.

Weight loading follows ``load_adapted.py`` semantics exactly:

* ``neck`` and ``decoder`` load strictly;
* ``encoder_trainable`` is copied **by name, shape-checked**, silently skipping anything that does
  not match — which is how ``last_n`` (blocks 8-11 only) and ``full`` (the entire encoder) both work
  against a base encoder that was already built and loaded;
* LoRA adapters load into the installed adapter modules.
"""

from __future__ import annotations

import torch

from ..spec import ModelSpec
from .adapters import apply_adaptation
from .base import STRIDES, Encoder, SegModel
from .decoders import build_decoder
from .encoders import build_encoder
from .necks import build_neck

#: Encoder-side tensors in a released head that have no home in the timm graph. ``mask_token`` is
#: used only by the masked-image-modelling pretraining objective.
_ENCODER_DROPPABLE = ("backbone.mask_token",)


def _remap_encoder_trainable(state: dict, spec: ModelSpec) -> dict:
    """Translate a head's ``encoder_trainable`` block into the assembled encoder's naming.

    The released heads were saved from the *reference* encoder, so their keys use DINOv3 naming
    relative to the ``Encoder`` module:

    * ``adapt="lora"``   -> ``_conv_lora.{i}.{down,up}.weight``  (already correct, passes through)
    * ``adapt="last_n"`` -> ``backbone.blocks.{8..11}.*``
    * ``adapt="full"``   -> the entire ``backbone.*``

    For the two ``backbone.*`` cases the DINOv3 names must be converted the same way the base
    checkpoint is (``ls1.gamma`` -> ``gamma_1``, ``storage_tokens`` -> ``reg_token``, ``qkv.bias``
    split into ``q_bias``/``k_bias``/``v_bias``). Skipping this step is not a loud failure — the
    non-matching tensors are simply never copied, leaving fine-tuned blocks at base-encoder values.
    """
    if spec.family != "quantem":
        return dict(state)

    from .encoders.quantem_vit import remap_reference_state_dict

    passthrough = {k: v for k, v in state.items() if not k.startswith("backbone.")}
    backbone_src = {
        k[len("backbone.") :]: v
        for k, v in state.items()
        if k.startswith("backbone.") and k not in _ENCODER_DROPPABLE
    }
    remapped = remap_reference_state_dict(backbone_src)
    remapped.pop("__rope_periods__", None)  # a buffer, never in encoder_trainable
    out = {f"backbone.{k}": v for k, v in remapped.items()}
    out.update(passthrough)
    return out


def _load_named_tensors(module: torch.nn.Module, state: dict) -> tuple[int, list[str]]:
    """Copy ``{name: tensor}`` into ``module`` by name, shape-checked.

    Returns ``(loaded, unmatched_keys)``. Extends ``load_adapted._load_named_params`` to search
    **buffers as well as parameters**, because ``k_bias`` is a buffer in the timm graph.
    """
    own = dict(module.named_parameters())
    own.update(dict(module.named_buffers()))
    loaded, unmatched = 0, []
    for k, v in state.items():
        p = own.get(k)
        if p is not None and tuple(p.shape) == tuple(v.shape):
            with torch.no_grad():
                p.copy_(v.to(p.dtype))
            loaded += 1
        else:
            unmatched.append(k)
    return loaded, unmatched


def build_model(
    spec: ModelSpec,
    *,
    encoder_state: dict | None = None,
    head_state: dict | None = None,
    device: str = "cpu",
    strict_encoder: bool = True,
) -> SegModel:
    """Build and load one released model.

    Parameters
    ----------
    encoder_state
        Base encoder tensors in timm naming. May be ``None`` for ``adapt="full"``, whose head
        artifact carries the whole encoder.
    head_state
        ``{"neck": ..., "decoder": ..., "encoder_trainable": ..., "adapters": ...}``.
    """
    img_size = spec.effective_tile()

    backbone = build_encoder(
        spec.encoder,
        encoder_state,
        img_size=img_size,
        strict=strict_encoder and encoder_state is not None,
    )
    encoder = Encoder(backbone, spec.encoder, apply_encoder_norm=spec.apply_encoder_norm)

    layers = encoder.resolved_layers(spec.feature_layers)
    n_taps = len(layers)

    neck = build_neck(spec.neck, spec.encoder.embed_dim, n_taps, spec.neck_out_channels)
    decoder = build_decoder(spec.decoder, neck.out_channels, STRIDES, spec.num_classes)

    # Adaptation must be applied BEFORE loading, so LoRA modules exist to receive their weights.
    encoder_trainable = apply_adaptation(encoder, spec.adapt, spec.adapt_params)

    model = SegModel(encoder, neck, decoder, layers, encoder_trainable=encoder_trainable)

    if head_state is not None:
        model.neck.load_state_dict(head_state["neck"], strict=True)
        model.decoder.load_state_dict(head_state["decoder"], strict=True)

        enc_state = head_state.get("encoder_trainable")
        if enc_state:
            n_src = len(enc_state)
            remapped = _remap_encoder_trainable(enc_state, spec)
            loaded, unmatched = _load_named_tensors(model.encoder, remapped)
            # Every encoder-side tensor in a released head must land. Anything left over means the
            # encoder was assembled differently than it was trained, and the model would run at
            # partially-base weights while looking perfectly healthy.
            if unmatched:
                raise RuntimeError(
                    f"{spec.model_id}: {len(unmatched)} of {len(remapped)} encoder tensors did not "
                    f"match the assembled encoder (silent-corruption guard). First: {unmatched[:6]}"
                )
            model._encoder_load_info = {"source_tensors": n_src, "loaded": loaded}
        elif (
            head_state.get("adapters") is not None
            and getattr(model.encoder, "_conv_lora", None) is not None
        ):
            model.encoder._conv_lora.load_state_dict(head_state["adapters"], strict=True)

    return model.to(device).eval()


def build_from_artifacts(spec: ModelSpec, tensors: dict, *, device: str = "cpu") -> SegModel:
    """Build a model from published artifacts (flat, prefixed keys).

    ``tensors`` is the merge of every artifact the model needs — trunk first, then the model
    artifact, so per-organelle fine-tuned blocks overwrite their base counterparts. Keys are
    ``encoder.*`` / ``neck.*`` / ``decoder.*`` / ``adapters.*``.
    """
    enc, neck, dec, adapters = {}, {}, {}, {}
    for k, v in tensors.items():
        if k.startswith("encoder."):
            enc[k[len("encoder.") :]] = v
        elif k.startswith("neck."):
            neck[k[len("neck.") :]] = v
        elif k.startswith("decoder."):
            dec[k[len("decoder.") :]] = v
        elif k.startswith("adapters."):
            adapters[k[len("adapters.") :]] = v
        else:
            raise ValueError(f"unexpected artifact key {k!r}")

    periods = enc.pop("rope.periods", None)
    if periods is not None:
        enc["__rope_periods__"] = periods

    head_state = {
        "neck": neck,
        "decoder": dec,
        # Encoder tensors already carry the assembled naming, so no remap is needed here.
        "encoder_trainable": None,
        "adapters": adapters or None,
    }
    model = build_model(
        spec, encoder_state=enc, head_state=head_state, device=device, strict_encoder=True
    )
    return model


def load_reference_head(path) -> dict:
    """Read a staged ``head.pt``. Packaging-time only — the runtime path uses safetensors.

    ``head.pt`` is a pickle loaded with ``weights_only=False``; that is exactly why the released
    artifacts are converted to safetensors and never downloaded in this form.
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return {
        "neck": ck["neck"],
        "decoder": ck["decoder"],
        "encoder_trainable": ck.get("encoder_trainable"),
        "adapters": ck.get("adapters"),
    }
