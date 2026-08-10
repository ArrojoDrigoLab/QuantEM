"""Eval-time style estimation for the ``dataset`` style scope of image-style conditioning, plus the retrieval helper.

The dataset-scope arm estimates a per-source style code from that source's unlabelled tiles (the
deployment scenario: at test time only the held-out source's images are available, not its labels, so the
estimate is unsupervised and uses no held-out annotation). This module pools the style encoder over a
sample of each source's tiles to produce one code per source, which the evaluator injects (FiLM preset)
for every window of that source. It also exposes the 'retrieve nearest training-source style' operation.

No GPU is needed: torch + the dataset loaders only.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .dataset import load_sample, normalize_em


def _center_tile(em: np.ndarray, tile: int) -> np.ndarray:
    """Center-crop / 0-pad a region to a single ``tile`` x ``tile`` window."""
    H, W = em.shape
    if H < tile or W < tile:
        ph, pw = max(tile - H, 0), max(tile - W, 0)
        mode = "constant"  # 0-padding rather than reflection, so no tissue is fabricated at the border
        em = np.pad(em, ((0, ph), (0, pw)), mode=mode)
        H, W = em.shape
    y0, x0 = (H - tile) // 2, (W - tile) // 2
    return em[y0:y0 + tile, x0:x0 + tile]


@torch.no_grad()
def precompute_source_codes(model, records, cfg, data_root, mean: float, std: float, device,
                            max_tiles_per_source: int = 16) -> dict[str, torch.Tensor]:
    """Estimate one style code per source from up to ``max_tiles_per_source`` of its unlabelled tiles.

    Returns ``{dataset: code[style_dim]}``. Uses only the images (never the masks) — the unsupervised
    deployment scenario. Deterministic (stable record order + fixed sample stride)."""
    conditioner = model.conditioner
    patch = int(getattr(model.encoder, "patch_size", 16))
    tile = ((int(cfg.encoder.tile_size) + patch - 1) // patch) * patch
    by_src: dict[str, list] = defaultdict(list)
    for r in records:
        by_src[str(r.get("dataset"))].append(r)
    codes: dict[str, torch.Tensor] = {}
    for src, recs in sorted(by_src.items()):
        recs = sorted(recs, key=lambda r: str(r.get("sample_id")))
        step = max(1, len(recs) // max_tiles_per_source)
        chosen = recs[::step][:max_tiles_per_source]
        tiles = []
        for r in chosen:
            em, _, _ = load_sample(r, data_root, with_inst=False)
            t = _center_tile(em, tile)
            tiles.append(normalize_em(t, mean, std))
        x = torch.from_numpy(np.stack(tiles)).unsqueeze(1).float().to(device)  # [n,1,H,W]
        code = conditioner.style_encoder(x)  # [n, style_dim]
        codes[src] = code.mean(dim=0).detach()  # pooled per-source deployment code
    return codes
