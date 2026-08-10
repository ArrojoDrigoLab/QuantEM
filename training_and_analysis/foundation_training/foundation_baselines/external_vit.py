"""External foundation-model ViTs loaded via ``timm`` as frozen feature extractors.

Every encoder here is publicly released and used for comparison only, never trained or fine-tuned:
EMCellFound, natural-image DINOv2 and DINOv3 at ViT-L, and OmniEM/EM-DINO. Each is a ViT loadable
through ``timm`` from a local weight file, and each exposes the tap interface both harnesses use — a
list of ``[B, C, H/p, W/p]`` grids for the requested block indices — via ``timm``'s uniform
``forward_intermediates``, which covers both the plain ``VisionTransformer`` and the ``Eva`` class timm
uses for DINOv3.

The load recipe for each is a configuration in ``configs/encoder_comparison/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import torch

# Standard ImageNet stats (0-1 image space) — the default here, and what the EMCF, Meta-DINOv2 and
# Meta-DINOv3 configurations set.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# OmniEM/EM-DINO's EM stats (single grayscale channel, broadcast to 3).
OMNIEM_MEAN = [0.595446]
OMNIEM_STD = [0.211906]

@dataclass
class ExternalEncoderSpec:
    """How to build + load + preprocess one external ViT baseline as a frozen extractor."""

    name: str
    timm_model: str            # timm.create_model name (arch; weights loaded separately from a file)
    weight_file: str           # filename under foundation_weights/<name>/ (.pth / .pt / .safetensors)
    objective: str             # provenance label: the pretraining objective of the released encoder
    arch: str                  # "vit_base" | "vit_large" (provenance / manifest)
    patch_size: int
    embed_dim: int
    depth: int
    in_chans: int = 3
    base_tile: int = 512       # target crop; the actual tile is round_to_patch(base_tile, patch_size)
    img_size_build: int = 224  # img_size passed to timm.create_model (sizes the learned pos-embed grid;
                               # ignored by RoPE models). Runtime uses dynamic_img_size to reach base_tile.
    dynamic_img_size: bool = True
    strip_prefix: str | None = None      # e.g. "vit." for OmniEM's bare backbone checkpoint
    drop_key_prefixes: tuple = ("head.",)
    allow_unexpected: tuple = ("mask_token",)  # unused pretraining tokens that need not load
    norm_mean: list = field(default_factory=lambda: list(IMAGENET_MEAN))
    norm_std: list = field(default_factory=lambda: list(IMAGENET_STD))
    # RoPE encoders generate position per forward from normalized coordinates, so they can read a larger
    # context window at inference and the context sweep applies to them. Encoders with a learned absolute
    # position embedding must interpolate it and go out of distribution at much larger sizes, so they stay
    # at their native tile.
    context_sweepable: bool = False

    def tile_size(self) -> int:
        return round_to_patch(self.base_tile, self.patch_size)

# The published encoders compared against, one configuration each in
# configs/encoder_comparison/. Each loads through timm from a local weight file.
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "encoder_comparison"

def load_registry(config_dir: str | Path | None = None) -> dict[str, ExternalEncoderSpec]:
    """Build the encoder registry from configs/encoder_comparison/*.yaml."""
    import yaml

    d = Path(config_dir or CONFIG_DIR)
    out: dict[str, ExternalEncoderSpec] = {}
    known = {f.name for f in fields(ExternalEncoderSpec)}
    for p in sorted(d.glob("*.yaml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        kw = {k: v for k, v in raw.items() if k in known}
        for tup in ("drop_key_prefixes", "allow_unexpected"):
            if tup in kw and kw[tup] is not None:
                kw[tup] = tuple(kw[tup])
        spec = ExternalEncoderSpec(**kw)
        out[spec.name] = spec
    return out

REGISTRY: dict[str, ExternalEncoderSpec] = load_registry()

def spec_from_manifest(manifest) -> "ExternalEncoderSpec":
    """Rebuild an ``ExternalEncoderSpec`` from a persisted ``EncoderManifest`` (framework ``timm_vit``),
    so the harness loader is fully driven by the checkpoint_index.json (not this module's REGISTRY)."""
    fe = dict(getattr(manifest, "feature_entry_point", None) or {})
    return ExternalEncoderSpec(
        name=manifest.run_id,
        timm_model=fe["timm_model"],
        weight_file="",  # the concrete path is the CheckpointRecord's, not needed here
        objective=manifest.objective,
        arch=manifest.arch,
        patch_size=int(manifest.patch_size),
        embed_dim=int(manifest.embedding_dim),
        depth=int(manifest.depth),
        in_chans=int(fe.get("in_chans", getattr(manifest, "input_channels", 3))),
        base_tile=int(fe.get("base_tile", 512)),
        img_size_build=int(fe.get("img_size_build", 224)),
        dynamic_img_size=bool(fe.get("dynamic_img_size", True)),
        strip_prefix=fe.get("strip_prefix"),
        drop_key_prefixes=tuple(fe.get("drop_key_prefixes", ("head.",))),
        allow_unexpected=tuple(fe.get("allow_unexpected", ("mask_token",))),
        norm_mean=list(manifest.image_mean),
        norm_std=list(manifest.image_std),
        context_sweepable=bool(fe.get("context_sweepable", False)),
    )

def round_to_patch(base: int, patch: int) -> int:
    """Nearest positive multiple of ``patch`` to ``base`` (patch-16 -> 512; patch-14 -> 518)."""
    k = max(1, round(base / patch))
    return int(k * patch)

def load_state_dict_any(path: str | Path) -> dict:
    """Load a state dict from ``.safetensors`` or a torch ``.pth``/``.pt`` (unwrapping a
    ``model`` / ``state_dict`` / ``teacher`` wrapper key when present)."""
    p = str(path)
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(p)
    ck = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(ck, dict):
        for k in ("model", "state_dict", "teacher"):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
    return ck

def build_external_backbone(spec: ExternalEncoderSpec, weight_path: str | Path):
    """Build the timm ViT, load ``weight_path``, assert a clean load, return the frozen eval model."""
    import timm

    model = timm.create_model(spec.timm_model, pretrained=False, num_classes=0,
                              in_chans=spec.in_chans, img_size=spec.img_size_build,
                              dynamic_img_size=spec.dynamic_img_size)
    sd = load_state_dict_any(weight_path)
    if spec.strip_prefix:
        stripped = {k[len(spec.strip_prefix):]: v for k, v in sd.items() if k.startswith(spec.strip_prefix)}
        sd = stripped or sd
    sd = {k: v for k, v in sd.items() if not any(k.startswith(p) for p in spec.drop_key_prefixes)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not any(k.startswith(p) for p in spec.drop_key_prefixes)]
    unexpected = [k for k in unexpected if k not in spec.allow_unexpected]
    if missing or unexpected:
        raise RuntimeError(
            f"[{spec.name}] unclean state_dict load into '{spec.timm_model}': "
            f"missing={missing[:8]} unexpected={unexpected[:8]}. Refusing to return a "
            f"partially-initialised encoder (silent-corruption guard)."
        )
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()

def _broadcast(vals: list[float], n: int) -> list[float]:
    if len(vals) == n:
        return list(vals)
    if len(vals) == 1:
        return list(vals) * n
    raise ValueError(f"norm stat length {len(vals)} incompatible with in_chans {n}")

def preprocess(x: torch.Tensor, in_chans: int, mean: list[float], std: list[float]) -> torch.Tensor:
    """Map the harness's single-channel [0,1] tile to the encoder's native input: replicate to
    ``in_chans`` and normalize per-channel. (The dataset is configured to hand these encoders an
    un-normalized [0,1] tile — image_mean=0/std=1 — so all normalization happens here.)"""
    if x.shape[1] == 1 and in_chans > 1:
        x = x.repeat(1, in_chans, 1, 1)
    m = torch.tensor(_broadcast(mean, in_chans), device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(_broadcast(std, in_chans), device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - m) / s

def extract_intermediates(model, x: torch.Tensor, layers: list[int], norm: bool = True) -> list[torch.Tensor]:
    """Uniform tap extraction: ``[B, C, H/p, W/p]`` per requested block index (ascending), via timm's
    ``forward_intermediates`` (supported by both VisionTransformer and the Eva DINOv3 class)."""
    idxs = sorted(int(i) for i in layers)
    out = model.forward_intermediates(
        x, indices=idxs, norm=norm, output_fmt="NCHW", intermediates_only=True,
    )
    return list(out)
