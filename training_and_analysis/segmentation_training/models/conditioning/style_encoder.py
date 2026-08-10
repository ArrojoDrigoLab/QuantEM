"""Style encoder — maps an EM tile (+ cheap low-level statistics) to a continuous style code.

Style lives in low-level image statistics (overall + organelle-specific contrast, texture scale, noise),
so the encoder is small: a few strided convs on the raw tile fused with hand-computed statistics
(intensity percentiles, local contrast, gradient energy, a noise estimate, and a radial Fourier-amplitude
summary — the "what scale/texture is this image" descriptor). Trained end-to-end with the segmentation
loss. Two non-spurious-style options: a gradient-reversed source adversary (``DomainAdversary``) so the
code does not just memorise source identity, and code dropout (``cond.metadata_dropout``, applied by
``hooks.film_conditioning``) for graceful degradation when the code is absent/noisy.

Torch-only (no GPU needed): torch.fft / torch.quantile / conv only; no sklearn and no numpy-BLAS.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ConvGNAct
from .grl import grad_reverse

# Names of the scalar low-level statistics; the order matches ``low_level_stats``.
_PCTS = (0.05, 0.25, 0.5, 0.75, 0.95)
_SPEC_BINS = 8
STAT_NAMES = (
    tuple(f"pct_{int(p * 100)}" for p in _PCTS)
    + ("mean", "std", "local_contrast", "grad_energy", "noise_sigma")
    + tuple(f"radfft_{i}" for i in range(_SPEC_BINS))
)
STATS_DIM = len(STAT_NAMES)


def _radial_bin_index(h: int, w_half: int, bins: int) -> torch.Tensor:
    """Assign each rFFT grid cell to one of ``bins`` power-spaced radial frequency bins ([H, w_half] long)."""
    fy = torch.fft.fftfreq(h).abs().view(h, 1)          # [-0.5,0.5) folded to [0,0.5]
    fx = torch.fft.rfftfreq(2 * (w_half - 1) if w_half > 1 else 1)  # [0,0.5], length w_half
    fx = fx.view(1, -1)
    r = torch.sqrt(fy * fy + fx * fx)                    # [h, w_half], in [0, ~0.707]
    r = r / (r.max() + 1e-8)
    # edges spaced by a 1.5 power law, narrowing them towards 0 so low frequencies (large-scale
    # contrast) get their own bins.
    edges = torch.linspace(0, 1, bins + 1) ** 1.5
    idx = torch.bucketize(r, edges[1:-1].contiguous())  # 0..bins-1
    return idx.clamp_(0, bins - 1).long()


@torch.no_grad()
def low_level_stats(image: torch.Tensor, spec_size: int = 128, bins: int = _SPEC_BINS) -> torch.Tensor:
    """Cheap per-image low-level statistics, ``[B, STATS_DIM]`` (detached; a fixed appearance descriptor).

    ``image`` is the normalised 1-channel EM tile ``[B, 1, H, W]``. Percentiles/mean/std/contrast/
    gradient/noise are computed at native resolution; the radial Fourier-amplitude summary is computed on a
    fixed ``spec_size`` downsample so the radial binning is resolution-independent and cheap.
    """
    if image.dim() != 4:
        raise ValueError(f"low_level_stats expects [B,1,H,W]; got {tuple(image.shape)}")
    b = image.shape[0]
    flat = image.reshape(b, -1).float()
    q = torch.quantile(flat, torch.tensor(_PCTS, device=flat.device, dtype=flat.dtype), dim=1).t()  # [B,5]
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True)
    # local contrast: mean of windowed std (avg-pool of x^2 minus (avg-pool x)^2).
    k = 9
    mu = F.avg_pool2d(image, k, stride=k)
    mu2 = F.avg_pool2d(image * image, k, stride=k)
    local_contrast = (mu2 - mu * mu).clamp_min(0).sqrt().mean(dim=(1, 2, 3), keepdim=False).view(b, 1)
    # gradient energy: mean abs finite-difference.
    gx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=(1, 2, 3)).view(b, 1)
    gy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=(1, 2, 3)).view(b, 1)
    grad_energy = 0.5 * (gx + gy)
    # noise estimate (Immerkaer): sigma ~ mean|conv(x, L)| * sqrt(pi/2)/6 with L the 3x3 Laplacian-of-Laplacian.
    lap = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]],
                       device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
    noise = F.conv2d(image, lap).abs().mean(dim=(1, 2, 3)).view(b, 1) * 0.20888
    # radial Fourier-amplitude summary on a fixed-size downsample.
    small = F.interpolate(image, size=(spec_size, spec_size), mode="bilinear", align_corners=False)
    small = small - small.mean(dim=(2, 3), keepdim=True)
    amp = torch.fft.rfft2(small[:, 0]).abs()             # [B, spec_size, spec_size//2+1]
    idx = _radial_bin_index(amp.shape[1], amp.shape[2], bins).to(amp.device).reshape(-1)
    amp_flat = amp.reshape(b, -1)
    rad = torch.zeros(b, bins, device=amp.device, dtype=amp.dtype)
    rad.scatter_add_(1, idx.unsqueeze(0).expand(b, -1), amp_flat)
    counts = torch.zeros(bins, device=amp.device, dtype=amp.dtype).scatter_add_(
        0, idx, torch.ones_like(idx, dtype=amp.dtype))
    rad = rad / counts.clamp_min(1.0)
    rad = torch.log1p(rad)
    rad = rad / (rad.norm(dim=1, keepdim=True) + 1e-6)   # scale-free spectral shape
    return torch.cat([q, mean, std, local_contrast, grad_energy, noise, rad], dim=1)


class StyleEncoder(nn.Module):
    """Inferred image-style code: small conv trunk on the raw tile + fused low-level statistics.

    ``forward(image, feats=None) -> [B, style_dim]``. When ``feat_dim>0`` the (detached) globally-pooled
    coarsest encoder tap is also fused (``style_from_features``). The trunk downsamples aggressively so the
    code depends on appearance statistics, not spatial layout.
    """

    def __init__(self, style_dim: int = 64, hidden: int = 64, use_stats: bool = True,
                 feat_dim: int = 0, in_ch: int = 1):
        super().__init__()
        self.use_stats = bool(use_stats)
        self.feat_dim = int(feat_dim)
        self.trunk = nn.Sequential(
            ConvGNAct(in_ch, hidden, k=3, stride=2),      # /2
            ConvGNAct(hidden, hidden, k=3, stride=2),     # /4
            ConvGNAct(hidden, hidden, k=3, stride=2),     # /8
            nn.AdaptiveAvgPool2d(1),
        )
        fused = hidden + (STATS_DIM if use_stats else 0) + (self.feat_dim if self.feat_dim else 0)
        self.head = nn.Sequential(
            nn.Linear(fused, 2 * style_dim), nn.GELU(), nn.Linear(2 * style_dim, style_dim),
        )
        self.style_dim = int(style_dim)

    def forward(self, image: torch.Tensor, feats: list[torch.Tensor] | None = None) -> torch.Tensor:
        parts = [self.trunk(image).flatten(1)]
        if self.use_stats:
            parts.append(low_level_stats(image).to(image.dtype))
        if self.feat_dim and feats:
            parts.append(feats[-1].detach().mean(dim=(2, 3)))  # global-pooled coarsest tap
        return self.head(torch.cat(parts, dim=1))


class ConfidentFeatureStyle(nn.Module):
    """Pooled-global code producer — a per-image, per-organelle appearance code pooled from the encoder features of
    confident organelle regions (GT foreground at train; the model's own confident prediction at test).

    This is the explicit annotator move: "recalibrate what a mitochondrion looks like in this image from
    the confidently identified ones." Distinct from the global style encoder (global image statistics, organelle-agnostic). Multi-
    prototype (``k_proto>1``): K learned query slots attend over the confident pixels -> K appearance modes
    -> gate (within-image variability). Empty confident set -> falls back to the global mean, so a source
    with almost no confident seeds (such as the FAST-EM under-caller) supplies little signal.

    Features are detached (an appearance descriptor, like the style encoder — not an encoder grad path).
    """

    def __init__(self, embed_dim: int, style_dim: int = 64, k_proto: int = 1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(embed_dim, 2 * style_dim), nn.GELU(),
                                  nn.Linear(2 * style_dim, style_dim))
        self.k = max(1, int(k_proto))
        self.style_dim = int(style_dim)
        if self.k > 1:
            self.queries = nn.Parameter(torch.randn(self.k, style_dim) * 0.02)
            from .pooling import PrototypeGate
            self.gate = PrototypeGate(style_dim)

    def forward(self, feats: list[torch.Tensor], fg_mask: torch.Tensor | None) -> torch.Tensor:
        x = feats[-1].detach()                                   # [B, C, h, w] coarsest tap
        b, c, h, w = x.shape
        p = self.proj(x.flatten(2).transpose(1, 2))             # [B, hw, style_dim]
        if fg_mask is None:
            w_m = torch.ones(b, h * w, device=x.device, dtype=p.dtype)
        else:
            m = F.interpolate(fg_mask.float(), size=(h, w), mode="bilinear", align_corners=False)
            w_m = (m.flatten(1) > 0.5).to(p.dtype)               # [B, hw] confident-pixel mask
        empty = w_m.sum(1) < 1                                   # no confident pixels -> global fallback
        if empty.any():
            w_m = w_m.clone()
            w_m[empty] = 1.0
        if self.k == 1:
            return (p * w_m[..., None]).sum(1) / w_m.sum(1, keepdim=True).clamp_min(1.0)
        # multi-prototype: K queries softmax-attend over the confident pixels only.
        q = F.normalize(self.queries, dim=1)
        att = F.normalize(p, dim=2) @ q.t()                     # [B, hw, K]
        att = att.masked_fill(w_m[..., None] < 0.5, float("-inf"))
        att = torch.softmax(att, dim=1)
        protos = torch.nan_to_num(torch.einsum("bnk,bnd->bkd", att, p))  # [B, K, d]
        return self.gate(protos)


class DomainAdversary(nn.Module):
    """Gradient-reversed adversary: predict ``targets`` (source/modality) from the style code.

    ``forward(code, alpha) -> {target: logits}``. The GRL negates the adversary's gradient into the style
    encoder (scaled by ``alpha`` = the DANN lambda), so the code is pushed to be un-predictive of source.
    The per-target CE loss is computed by the trainer against the batch metadata."""

    def __init__(self, style_dim: int, targets: list[str], n_classes: dict[str, int], hidden: int = 128):
        super().__init__()
        self.targets = list(targets)
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(style_dim, hidden), nn.ReLU(inplace=True),
                             nn.Linear(hidden, max(2, int(n_classes.get(t, 2)))))
            for t in self.targets})

    def forward(self, code: torch.Tensor, alpha: float) -> dict[str, torch.Tensor]:
        rev = grad_reverse(code, alpha)
        return {t: head(rev) for t, head in self.heads.items()}
