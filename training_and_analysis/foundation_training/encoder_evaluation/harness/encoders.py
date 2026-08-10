"""Frozen-encoder loading and dense feature extraction.

Loads strictly from ``checkpoint_index.json`` (via ``em_ssl.utils.CheckpointIndex``); every weight
comes from a local file and nothing is fetched at load time. The encoder is always frozen (eval + no
grad). Feature extraction is symmetric across the two supported frameworks, ``dinov3`` and
``timm_vit``: both return the patch-token grid of the selected blocks as ``[B, C, H/p, W/p]`` for the
manifest's patch size ``p``, with the encoder's final LayerNorm applied per selected layer (``dinov3``
via ``get_intermediate_layers``, ``timm_vit`` via timm's ``forward_intermediates``).
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
        # External (timm) encoders carry their native input contract here: the dataset hands them a
        # raw [0,1] single-channel tile and preprocessing — channel replication and per-channel
        # normalization — happens in _extract_impl. None for the single-channel EM encoders.
        self._ext_in_chans: int | None = None
        self._ext_norm_mean: list | None = None
        self._ext_norm_std: list | None = None
        # Common comparison region in px, set by run_probe. None returns the full tap grid; when set,
        # extract() crops each tap to the central compare_tile-worth of tokens.
        self.compare_tile: int | None = None
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
    def _extract_impl(self, x: torch.Tensor, layers: list[int]) -> list[torch.Tensor]:
        """Return one ``[B, C, H/p, W/p]`` feature map per requested block index.

        Layers are sorted ascending so the returned order is identical across frameworks: DINOv3's
        ``get_intermediate_layers`` always emits ascending block order, so the external timm path must too
        (else the decoder's channel-concat order would differ between encoders for an unsorted layer list).
        """
        layers = sorted(layers)
        if self.framework == "dinov3":
            feats = list(self.backbone.get_intermediate_layers(
                x, n=layers, reshape=True, norm=self.apply_encoder_norm, return_class_token=False,
            ))
        elif self.framework == "timm_vit":
            from foundation_baselines.external_vit import extract_intermediates, preprocess
            x = preprocess(x, self._ext_in_chans, self._ext_norm_mean, self._ext_norm_std)
            feats = extract_intermediates(self.backbone, x, layers, norm=self.apply_encoder_norm)
        else:
            raise ValueError(f"Unknown framework {self.framework!r} (expected dinov3|timm_vit)")
        return self._crop_central_tokens(feats)

    @torch.no_grad()
    def extract(self, x: torch.Tensor, layers: list[int]) -> list[torch.Tensor]:
        """Frozen (no-grad) feature extraction — the standard probe path."""
        return self._extract_impl(x, layers)

    def extract_train(self, x: torch.Tensor, layers: list[int]) -> list[torch.Tensor]:
        """Grad-enabled extraction for encoder adaptation (LoRA / last_n): the backbone forward runs
        with grad so the trainable adapter / unfrozen-block params receive gradients. Base weights stay
        frozen (requires_grad=False). Used by train_head when the encoder is adapted."""
        return self._extract_impl(x, layers)

    def _crop_central_tokens(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Token crop: keep only the central ``compare_tile`` region of each tap grid.

        The encoder read the full ``tile_size`` context window, so these central tokens have already
        attended (global self-attention) to the surrounding context. Cropping to the common region
        before the decoder gives every encoder's decoder the same common-size input, while a
        wide-context encoder's central tokens still carry the extra context. No-op when
        ``compare_tile`` is None or already the grid size.
        """
        if self.compare_tile is None:
            return feats
        n = max(1, round(self.compare_tile / self.patch_size))
        out = []
        for f in feats:
            H, W = f.shape[-2], f.shape[-1]
            if H <= n and W <= n:
                out.append(f)
                continue
            oh, ow = max(0, (H - n) // 2), max(0, (W - n) // 2)
            out.append(f[:, :, oh:oh + n, ow:ow + n])
        return out

# --------------------------------------------------------------------------- #
# Framework-specific backbone builders
# --------------------------------------------------------------------------- #
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
    # The bare arch factory uses ctor defaults that differ from the SSL training config: LayerScale is
    # disabled by default (layerscale_init=None) but enabled in training, so a build from the defaults
    # would drop the checkpoint's ls1/ls2.gamma and corrupt the features. The block config is instead
    # reconstructed from the checkpoint's own keys (the shared helper fino_diagnostics also uses).
    from em_ssl.utils.checkpoint_index import infer_dinov3_build_kwargs
    kwargs = infer_dinov3_build_kwargs(backbone_sd, {"patch_size": int(manifest.patch_size), "in_chans": 1})
    model = factory(**kwargs)
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    _warn_load("dinov3", missing, unexpected)
    if unexpected:
        warnings.warn(f"[dinov3] {len(unexpected)} checkpoint keys did not load into the backbone "
                      f"({list(unexpected)[:6]}). The rebuild config does not match this run, so the "
                      f"extracted features do not correspond to the trained model and any metrics "
                      f"computed from them are invalid.")
    return model.eval()

def _build_timm_vit_encoder(cls, ckpt_path, manifest, apply_encoder_norm):
    """Build a frozen external timm ViT baseline from its manifest.

    Covers every encoder in ``configs/encoder_comparison/``: EMCF-MAE, the natural-image DINOv2 and
    DINOv3 at ViT-L, and OmniEM. The wrapper's dataset-facing ``image_mean``/``image_std`` are set to
    0/1 so the dataset hands the encoder a raw [0,1] single-channel tile; the encoder's native
    per-channel normalization + channel replication (ImageNet for EMCF and the natural-image
    DINOv2/DINOv3, EM stats for OmniEM) are applied in ``_extract_impl`` via
    ``foundation_baselines.external_vit.preprocess``, leaving the ``dinov3`` path untouched.
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
