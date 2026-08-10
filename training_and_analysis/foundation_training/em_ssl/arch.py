"""Shared architecture registry: embed dims, depth and FFN ratio per ViT size.

DINOv3 ViT factory names and their dimensions (from the upstream
dinov3/models/vision_transformer.py factory functions). They supply the checkpoint_index metadata
(embedding_dim, depth), so the decoder probe knows the feature width without instantiating the model.

The DINOv3 "huge" arch is named ``vit_huge2``; upstream has no ``vit_huge``, and the alias table
below is what accepts that spelling in a config.
RoPE constraint satisfied by all: embed_dim % (4*num_heads) == 0.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ArchSpec:
    name: str
    embed_dim: int
    depth: int
    num_heads: int
    ffn_ratio: float = 4.0

ARCH_SPECS: dict[str, ArchSpec] = {
    "vit_small": ArchSpec("vit_small", 384, 12, 6, 4.0),
    "vit_base": ArchSpec("vit_base", 768, 12, 12, 4.0),
    "vit_large": ArchSpec("vit_large", 1024, 24, 16, 4.0),
    "vit_so400m": ArchSpec("vit_so400m", 1152, 27, 18, 3.777777778),
    "vit_huge2": ArchSpec("vit_huge2", 1280, 32, 20, 4.0),
    "vit_giant2": ArchSpec("vit_giant2", 1536, 40, 24, 4.0),
    "vit_7b": ArchSpec("vit_7b", 4096, 40, 32, 3.0),
}

# Friendly aliases accepted in configs.
_ALIASES = {
    "vit_huge": "vit_huge2",
    "vitb": "vit_base",
    "vitl": "vit_large",
    "vith": "vit_huge2",
    "vits": "vit_small",
}

def resolve_arch(name: str) -> ArchSpec:
    key = _ALIASES.get(name, name)
    if key not in ARCH_SPECS:
        raise KeyError(f"Unknown arch '{name}'. Known: {sorted(ARCH_SPECS)} (+ aliases {sorted(_ALIASES)}).")
    return ARCH_SPECS[key]
