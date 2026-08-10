"""MixStyle / DSU training-time feature-statistic mixing — the cheapest, lowest-risk DG arm.

Both perturb instance-level feature statistics (channel mean/std over space) at early neck/decoder layers,
synthesising unseen "styles" so the head cannot lean on source-specific appearance shortcuts. No test-time
style estimation (plug-and-play), works with limited labels, train-only (identity at eval).

* MixStyle (Zhou et al., ICLR 2021; arXiv:2104.02008) — interpolates two samples' statistics with a
  Beta-sampled weight (stays inside the batch's convex hull of styles). Verbatim from the official
  KaiyangZhou/Dassl.pytorch ``dassl/modeling/ops/mixstyle.py``.
* DSU (Li et al., ICLR 2022; lixiaotong97/DSU) — samples fresh statistics from a Gaussian whose spread is
  the batch-variance of the statistics (can synthesise styles outside the observed set). Verbatim from the
  official ``dsu.py``. Needs batch>1 (batch-variance), which the default batch size of 8 satisfies.

Multi-source extension: ``mix='crossdomain'`` builds a permutation that pairs each sample with one from
a different source when possible (mixing statistics across sources within a batch), falling back to
random. The per-sample source ids are supplied via ``set_source_ids`` before the forward, which the
conditioner does.

Torch-only (no GPU needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _crossdomain_perm(source_ids: torch.Tensor) -> torch.Tensor:
    """Permutation pairing each sample with a random different-source partner (random fallback)."""
    b = source_ids.shape[0]
    perm = torch.randperm(b, device=source_ids.device)
    # For any sample whose partner shares its source, try to swap toward a different-source partner.
    for i in range(b):
        if source_ids[perm[i]] == source_ids[i]:
            cand = (source_ids[perm] != source_ids[i]).nonzero(as_tuple=True)[0]
            if cand.numel():
                j = cand[torch.randint(cand.numel(), (1,), device=source_ids.device)].item()
                perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    return perm


class MixStyle(nn.Module):
    """MixStyle (interpolate instance statistics across the batch). Verbatim logic + source-aware option."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, mix: str = "random"):
        super().__init__()
        self.p = float(p)
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.mix = str(mix)
        self._activated = True
        self._source_ids: torch.Tensor | None = None

    def set_source_ids(self, ids: torch.Tensor | None) -> None:
        self._source_ids = ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._activated:
            return x
        if torch.rand(()).item() > self.p:
            return x
        b = x.size(0)
        if b < 2:
            return x
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        lmda = self.beta.sample((b, 1, 1, 1)).to(x.device)
        if self.mix == "crossdomain" and self._source_ids is not None:
            perm = _crossdomain_perm(self._source_ids.to(x.device))
        else:
            perm = torch.randperm(b, device=x.device)
        mu_mix = mu * lmda + mu[perm] * (1 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1 - lmda)
        return x_normed * sig_mix + mu_mix


class DSU(nn.Module):
    """Distribution Uncertainty (DSU): sample perturbed statistics from a batch-estimated Gaussian."""

    def __init__(self, p: float = 0.5, eps: float = 1e-6, factor: float = 1.0):
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)
        self.factor = float(factor)
        self._activated = True

    def set_source_ids(self, ids: torch.Tensor | None) -> None:  # API parity with MixStyle
        pass

    def _sqrtvar(self, x: torch.Tensor) -> torch.Tensor:
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        return t.repeat(x.shape[0], 1)

    def _reparam(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(std) * self.factor * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._activated:
            return x
        if torch.rand(()).item() > self.p:
            return x
        b, c = x.shape[0], x.shape[1]
        if b < 2:
            return x
        mean = x.mean(dim=[2, 3])                     # [B,C]
        std = (x.var(dim=[2, 3]) + self.eps).sqrt()   # [B,C]
        beta = self._reparam(mean, self._sqrtvar(mean))
        gamma = self._reparam(std, self._sqrtvar(std))
        x = (x - mean.view(b, c, 1, 1)) / std.view(b, c, 1, 1)
        return x * gamma.view(b, c, 1, 1) + beta.view(b, c, 1, 1)


def build_mixer(kind: str, p: float, alpha: float, mix: str) -> nn.Module:
    """``mixstyle`` -> MixStyle, ``dsu`` -> DSU."""
    kind = (kind or "off").lower()
    if kind == "mixstyle":
        return MixStyle(p=p, alpha=alpha, mix=mix)
    if kind == "dsu":
        return DSU(p=p)
    raise ValueError(f"unknown mixstyle kind {kind!r} (expected 'mixstyle' or 'dsu')")


class MixerHooks(nn.Module):
    """Owns a single shared mixer instance and hooks it onto the output of the given modules
    (early neck/decoder layers). Train-only via the mixer's own gating."""

    def __init__(self, mixer: nn.Module, modules: list[nn.Module]):
        super().__init__()
        self.mixer = mixer
        self._handles = [m.register_forward_hook(self._hook) for m in modules]

    def _hook(self, _module, _inp, out):
        if isinstance(out, tuple):
            return (self.mixer(out[0]), *out[1:])
        return self.mixer(out)

    def set_source_ids(self, ids: torch.Tensor | None) -> None:
        self.mixer.set_source_ids(ids)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
