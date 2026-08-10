"""Style-scope pooling — estimate the style code over different scopes + multi-prototype.

Scope is a config axis (``cfg.cond.style_scope``):
  * ``tile``    — per-crop code (no pooling); the default.
  * ``source``  — pool per-tile codes over crops of the same source within the batch (the multi-tile /
                  volume-slice-group proxy at train time).
  * ``dataset`` — pool over many unlabelled tiles of a source (the deployment scope, estimable at test
                  time from a held-out source's unlabelled tiles). At train time it reduces to ``source``
                  pooling within the batch; at test time one code per source is precomputed up front by
                  ``harness/conditioning_eval.py``.

Multi-prototype (tile scope, ``n_prototypes>1``): encode a spatial grid of the tile into K region codes and
combine them with a learned soft-assignment gate — captures within-image appearance variability.

``torch_kmeans`` is a torch-only Lloyd's k-means (no sklearn dependency) reused by the test-time support
arms: prototype clustering in ``harness/tta.py`` and the self-support prototypes in ``positional_debias.py``.

Torch-only (no GPU needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pool_by_source(codes: torch.Tensor, source_ids: torch.Tensor | None) -> torch.Tensor:
    """Replace each sample's code with the mean code over same-source samples in the batch.

    ``codes`` [B, d], ``source_ids`` [B] (long). Returns [B, d]. ``None`` source_ids -> identity (tile
    scope). Deterministic + differentiable (a scatter-mean), so it trains end-to-end.
    """
    if source_ids is None:
        return codes
    b, d = codes.shape
    # Pool only when there is one source id per code (e.g. a training batch). When the code batch is a set
    # of tiles that doesn't line up with the source ids (a single-record eval / a TTA tile batch), fall
    # back to tile scope rather than crash.
    if source_ids.numel() != b:
        return codes
    sid = source_ids.to(codes.device).long().view(b)
    uniq, inv = torch.unique(sid, return_inverse=True)
    sums = torch.zeros(uniq.numel(), d, device=codes.device, dtype=codes.dtype)
    sums.index_add_(0, inv, codes)
    counts = torch.zeros(uniq.numel(), device=codes.device, dtype=codes.dtype)
    counts.index_add_(0, inv, torch.ones(b, device=codes.device, dtype=codes.dtype))
    means = sums / counts.clamp_min(1.0).unsqueeze(1)
    return means[inv]


def torch_kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0):
    """Lloyd's k-means on ``x`` [N, d] -> (labels [N], centroids [k, d]). Torch-only, no sklearn.

    Deterministic k-means++-lite seeding via a fixed generator; empty clusters keep their prior centroid.
    """
    n, d = x.shape
    k = max(1, min(int(k), n))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    # k-means++-style seeding: first centroid random, rest by squared-distance weighting.
    idx0 = torch.randint(0, n, (1,), generator=g).item()
    centroids = [x[idx0]]
    for _ in range(1, k):
        d2 = torch.stack([((x - c) ** 2).sum(1) for c in centroids], dim=1).min(dim=1).values
        probs = (d2 / d2.sum().clamp_min(1e-12)).cpu()
        nxt = torch.multinomial(probs, 1, generator=g).item()
        centroids.append(x[nxt])
    C = torch.stack(centroids, dim=0)
    labels = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(int(iters)):
        dist = torch.cdist(x, C)              # [N, k]
        new = dist.argmin(dim=1)
        if torch.equal(new, labels):
            labels = new
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = x[m].mean(dim=0)
    return labels, C


class PrototypeGate(nn.Module):
    """Combine K region prototype codes [B, K, d] into one effective code [B, d] via a soft gate."""

    def __init__(self, style_dim: int):
        super().__init__()
        self.score = nn.Linear(style_dim, 1)

    def forward(self, protos: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(protos).squeeze(-1), dim=1).unsqueeze(-1)  # [B, K, 1]
        return (protos * w).sum(dim=1)


def grid_encode(style_encoder, image: torch.Tensor, grid: int) -> torch.Tensor:
    """Encode a ``grid``x``grid`` spatial partition of the tile into region codes [B, grid*grid, d]."""
    b, c, h, w = image.shape
    gh, gw = h // grid, w // grid
    codes = []
    for i in range(grid):
        for j in range(grid):
            sub = image[:, :, i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            codes.append(style_encoder(sub))
    return torch.stack(codes, dim=1)  # [B, grid*grid, d]
