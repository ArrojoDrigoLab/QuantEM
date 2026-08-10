"""Frozen-encoder loading + dense feature extraction for the checkpoints the index records.

Loads strictly from the project's ``checkpoint_index.json`` (via ``em_ssl.utils.CheckpointIndex``);
weights come from the path the index records, never from a model hub. The encoder is always frozen
(eval + no grad). Feature extraction is symmetric across the two frameworks the loader accepts,
``dinov3`` and ``timm_vit``: both return the patch-token grid of the selected blocks as
``[B, C, H/p, W/p]`` for the encoder's patch size ``p``, with the encoder's final LayerNorm applied
per selected layer (``dinov3`` via ``get_intermediate_layers``, ``timm_vit`` via
``foundation_baselines.external_vit.extract_intermediates``).
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import torch
import torch.nn as nn


def select_checkpoints(index, n: int = 4, steps: list[int] | None = None) -> list:
    """Pick ``n`` evenly-spaced loadable encoder checkpoints (kind teacher/encoder) by step.

    ``steps`` overrides with explicit step values. Spanning training also exposes the
    SSL-progress-vs-downstream-quality trend.
    """
    recs = sorted(index.teacher_checkpoints(), key=lambda r: r.step)
    if not recs:
        return []
    if steps:
        by_step = {r.step: r for r in recs}
        return [by_step[s] for s in steps if s in by_step]
    if n >= len(recs):
        return recs
    # evenly spaced indices across the available checkpoints, inclusive of first & last
    idx = [round(i * (len(recs) - 1) / (n - 1)) for i in range(n)] if n > 1 else [len(recs) - 1]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(recs[i])
    return out


class FrozenEncoder(nn.Module):
    """Wraps a frozen backbone, exposing a uniform ``extract(x, layers)``."""

    def __init__(self, backbone: nn.Module, framework: str, arch: str, depth: int,
                 embedding_dim: int, patch_size: int, image_mean, image_std,
                 apply_encoder_norm: bool = True):
        super().__init__()
        self.backbone = backbone
        self.framework = framework
        self.arch = arch
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.image_mean = float(image_mean[0] if isinstance(image_mean, (list, tuple)) else image_mean)
        self.image_std = float(image_std[0] if isinstance(image_std, (list, tuple)) else image_std)
        self.apply_encoder_norm = apply_encoder_norm
        # External (timm) baseline encoders carry their native input contract here: the dataset hands
        # them a raw [0,1] single-channel tile (image_mean=0/std=1) and channel-replication + per-channel
        # normalization happen in _extract_impl. None for the project's single-channel EM encoders,
        # which load through the dinov3 framework and consume the dataset's normalized tile directly.
        self._ext_in_chans: int | None = None
        self._ext_norm_mean: list | None = None
        self._ext_norm_std: list | None = None
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_manifest(cls, ckpt_path: str | Path, manifest, tile_size: int,
                      apply_encoder_norm: bool = True) -> "FrozenEncoder":
        fw = manifest.framework
        if fw == "timm_vit":
            return _build_timm_vit_encoder(cls, ckpt_path, manifest, apply_encoder_norm)
        if fw == "dinov3":
            backbone = _build_dinov3(ckpt_path, manifest)
        else:
            raise ValueError(f"Unknown framework {fw!r} (expected dinov3|timm_vit)")
        return cls(
            backbone=backbone, framework=fw, arch=manifest.arch, depth=manifest.depth,
            embedding_dim=manifest.embedding_dim, patch_size=manifest.patch_size,
            image_mean=manifest.image_mean, image_std=manifest.image_std,
            apply_encoder_norm=apply_encoder_norm,
        )

    # -- feature extraction -------------------------------------------------
    def features(self, x: torch.Tensor, layers: list[int], grad: bool = False) -> list[torch.Tensor]:
        """Tapped feature grids, optionally with gradients (for the conv_lora adapter arm).

        ``grad=False`` (default) runs frozen under ``no_grad``, the default for neck, decoder and loss
        training, where the encoder is fixed. ``grad=True`` keeps the graph so trainable encoder-side adapters (see
        ``segmentation_training.harness.adapters``) receive gradients; the base backbone weights stay frozen regardless.
        """
        if grad:
            return self._extract_impl(x, layers)
        with torch.no_grad():
            return self._extract_impl(x, layers)

    @torch.no_grad()
    def extract(self, x: torch.Tensor, layers: list[int]) -> list[torch.Tensor]:
        """Frozen feature extraction (alias for ``features(x, layers, grad=False)``)."""
        return self._extract_impl(x, layers)

    def _extract_impl(self, x: torch.Tensor, layers: list[int]) -> list[torch.Tensor]:
        """Return one ``[B, C, H/p, W/p]`` feature map per requested block index (``p`` = patch size).

        Layers are sorted ascending so the returned order is identical across frameworks: DINOv3's
        ``get_intermediate_layers`` always emits ascending block order, so the timm path must too
        (else the decoder's channel-concat order would differ between encoders for an unsorted layer
        list).
        """
        layers = sorted(layers)
        if self.framework == "dinov3":
            feats = self.backbone.get_intermediate_layers(
                x, n=layers, reshape=True, norm=self.apply_encoder_norm, return_class_token=False,
            )
            return list(feats)
        if self.framework == "timm_vit":
            from foundation_baselines.external_vit import extract_intermediates, preprocess
            x = preprocess(x, self._ext_in_chans, self._ext_norm_mean, self._ext_norm_std)
            return extract_intermediates(self.backbone, x, layers, norm=self.apply_encoder_norm)
        raise ValueError(f"Unknown framework {self.framework!r} (expected dinov3|timm_vit)")


def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _build_dinov3(ckpt_path, manifest) -> nn.Module:
    module = importlib.import_module("dinov3.models.vision_transformer")
    factory = getattr(module, manifest.arch)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("teacher", ckpt) if isinstance(ckpt, dict) else ckpt
    backbone_sd = _strip_prefix(sd, "backbone.")
    if not backbone_sd:
        raise ValueError(f"No 'backbone.*' keys in {ckpt_path}")
    # The bare arch factory uses ctor defaults that differ from the SSL training config: in particular
    # LayerScale is off by default (layerscale_init=None) but on in training, so a naive build drops
    # the checkpoint's ls1/ls2.gamma and yields features that do not match the trained model. The block
    # config is therefore reconstructed from the checkpoint's own keys.
    from em_ssl.utils.checkpoint_index import infer_dinov3_build_kwargs
    kwargs = infer_dinov3_build_kwargs(backbone_sd, {"patch_size": int(manifest.patch_size), "in_chans": 1})
    model = factory(**kwargs)
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    _warn_load("dinov3", missing, unexpected)
    if unexpected:
        warnings.warn(f"[dinov3] {len(unexpected)} checkpoint keys did not load into the backbone "
                      f"({list(unexpected)[:6]}). The rebuild config does not match this run, so the "
                      f"features will not match the trained model and the resulting metrics are not valid.")
    return model.eval()


def _build_timm_vit_encoder(cls, ckpt_path, manifest, apply_encoder_norm):
    """Build a frozen external timm ViT baseline from its manifest.

    Covers every published baseline registered in ``foundation_baselines.external_vit``; the manifest,
    not this module, decides which timm architecture and weight file are used.

    Dataset-facing ``image_mean``/``image_std`` are set to 0/1 so the dataset hands the encoder a raw
    [0,1] single-channel tile; the encoder's native per-channel normalization + channel replication are
    applied in ``_extract_impl`` via ``foundation_baselines.external_vit.preprocess``, so the EM
    neck/decoder arms stay byte-identical.
    """
    from foundation_baselines.external_vit import build_external_backbone, spec_from_manifest

    spec = spec_from_manifest(manifest)
    backbone = build_external_backbone(spec, ckpt_path)
    enc = cls(
        backbone=backbone, framework="timm_vit", arch=manifest.arch, depth=manifest.depth,
        embedding_dim=manifest.embedding_dim, patch_size=manifest.patch_size,
        image_mean=[0.0], image_std=[1.0], apply_encoder_norm=apply_encoder_norm,
    )
    enc._ext_in_chans = int(spec.in_chans)
    enc._ext_norm_mean = list(spec.norm_mean)
    enc._ext_norm_std = list(spec.norm_std)
    return enc



def _warn_load(fw: str, missing, unexpected) -> None:
    if missing:
        warnings.warn(f"[{fw}] {len(missing)} missing keys on load (first: {list(missing)[:4]})")
    if unexpected:
        warnings.warn(f"[{fw}] {len(unexpected)} unexpected keys on load (first: {list(unexpected)[:4]})")
