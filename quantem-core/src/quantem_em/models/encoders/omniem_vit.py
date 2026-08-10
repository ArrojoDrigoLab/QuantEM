"""The OmniEM (EM-DINO) ViT-L/14 encoder.

This one was never a problem: the published OmniEM checkpoint is a DINOv2 ViT-L/14 and the research
harness already loaded it through timm's own Apache-2.0 DINOv2 implementation
(``foundation_baselines/external_vit.py``: ``vit_large_patch14_dinov2.lvd142m``, ``strip_prefix="vit."``).
DINOv2 was relicensed from CC-BY-NC to Apache-2.0 on 2023-08-31, so no custom-licence code is
involved anywhere in this path.

Input contract, matching ``external_vit.preprocess``: the dataset hands the encoder a raw ``[0, 1]``
single-channel tile (``dataset_mean=0``, ``dataset_std=1``); the 1 -> 3 channel replication and the
EM-specific per-channel normalisation (0.595446 / 0.211906) happen in ``Encoder.preprocess``.

All four OmniEM heads adapt with rank-8 LoRA, which leaves the base weights untouched — so this
encoder is genuinely shared across them and is downloaded once.

Licence note: the OmniEM *code* is MIT, but the published checkpoints carry no licence instrument
at all. That affects whether we may mirror the file, not whether we may use it, so the checkpoint is
fetched from the authors' own distribution point.
"""

from __future__ import annotations

import torch

from ...spec import EncoderSpec

_DROP_PREFIXES = ("head.",)


def remap_reference_state_dict(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop the pretraining head; the remaining keys already match timm's ViT naming."""
    return {k: v for k, v in src.items() if not k.startswith(_DROP_PREFIXES)}


def build_omniem_encoder(
    spec: EncoderSpec,
    state_dict: dict[str, torch.Tensor] | None = None,
    *,
    img_size: int = 518,
    strict: bool = True,
):
    """Build the OmniEM ViT-L and, if given, load ``state_dict``.

    ``img_size`` should be the effective tile (518 for patch 14 at a 512 nominal tile). With
    ``dynamic_img_size=True`` the position embedding is interpolated per input, so a different
    working size still runs.
    """
    import timm

    model = timm.create_model(
        spec.timm_model,
        pretrained=False,
        in_chans=spec.in_chans,
        img_size=img_size,
        num_classes=0,
        dynamic_img_size=True,
    )

    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # A pretraining checkpoint legitimately lacks nothing here, but tolerate the mask token.
        missing = [m for m in missing if m not in ("mask_token",)]
        unexpected = [u for u in unexpected if u not in ("mask_token",)]
        if strict and (missing or unexpected):
            raise RuntimeError(
                "unclean OmniEM encoder load (silent-corruption guard): "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )

    n = int(getattr(model, "num_prefix_tokens", 1))
    if n != spec.n_prefix_tokens:
        raise RuntimeError(
            f"prefix-token mismatch: timm reports {n}, spec expects {spec.n_prefix_tokens}."
        )
    return model.eval()


def load_reference_checkpoint(path, spec: EncoderSpec) -> dict[str, torch.Tensor]:
    """Read the published ``backbone_emdino_v1.pt`` and return timm-named tensors."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck.get(spec.checkpoint_key, ck) if spec.checkpoint_key else ck
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    prefix = spec.strip_prefix or ""
    src = (
        {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)} if prefix else dict(sd)
    )
    if not src:
        raise ValueError(f"no {prefix!r} keys in {path}")
    return remap_reference_state_dict(src)
