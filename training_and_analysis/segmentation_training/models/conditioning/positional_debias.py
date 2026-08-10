"""Positional-feature debiasing + matchability diagnostic + self-support prototypes (the feature-matching arm).

For the feature-matching test-time arm, dense features carry a strong positional bias that swamps
appearance similarity. INSID3 (CVPR 2026, arXiv:2603.28480; github.com/visinf/insid3) removes it without
retraining: pass a low-semantic (constant/noise) tile through the frozen encoder, L2-normalise its patch
features, SVD the channel x patch matrix, and project real features onto the orthogonal complement of the
top-s positional singular directions. Matching is then done in the debiased, re-normalised space.

Because LoRA adaptation can reshape the feature manifold, ``matchability`` measures how matchable the
adapted features are — positional energy plus nearest-neighbour cosine coherence. The matching arms
compute it once per run and carry it in their result summary, so a weak second pass can be read against
the feature quality it had to work with.

``SelfSupportHead`` is the Self-Support FSS propagation (Fan et al., ECCV 2022; github.com/fanq15/SSP):
confident query-foreground pixels become query-specific prototypes (global + local/adaptive) that are
propagated to low-confidence regions by cosine similarity — the multi-prototype second pass.

Torch-only, no GPU needed: torch.linalg.svd + torch cosine. The reference implementation relies on
torchvision and einops; this one depends on torch alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PositionalDebias:
    """Positional-feature debiasing on a frozen encoder's dense features."""

    def __init__(self, encoder, layer: int | None = None, svd_components: int = 64):
        self.encoder = encoder
        self.layer = layer if layer is not None else (encoder.depth - 1)
        self.svd_components = int(svd_components)
        self.basis: torch.Tensor | None = None  # [C, s], the positional subspace

    @torch.no_grad()
    def build_basis(self, tile: int, device, n_lowsem: int = 4) -> torch.Tensor:
        """Estimate the positional subspace from ``n_lowsem`` low-semantic (constant + noise) tiles.

        Uses the left singular vectors of the L2-normalised, patch-mean-centred channel x patch matrix
        (robust to the exact low-semantic image). Built once per encoder+resolution.
        """
        imgs = [torch.zeros(1, 1, tile, tile, device=device)]
        for i in range(1, n_lowsem):
            g = torch.Generator(device="cpu").manual_seed(i)
            imgs.append(torch.randn(1, 1, tile, tile, generator=g).to(device) * 0.1)
        cols = []
        for im in imgs:
            fmap = self.encoder.features(im, [self.layer], grad=False)[0]  # [1, C, H, W]
            f = F.normalize(fmap, p=2, dim=1)
            c = f.shape[1]
            cols.append(f.reshape(c, -1))
        E = torch.cat(cols, dim=1).float()             # [C, P]
        E = E - E.mean(dim=1, keepdim=True)            # mean-centre over patches
        U, _, _ = torch.linalg.svd(E, full_matrices=False)  # U: [C, C]
        s = min(self.svd_components, U.shape[1])
        self.basis = U[:, :s].contiguous().to(fmap.dtype)   # [C, s]
        return self.basis

    def debias(self, fmap: torch.Tensor) -> torch.Tensor:
        """Project ``[B, C, H, W]`` features onto the positional subspace's orthogonal complement.

        L2-normalise -> ``(I - B Bᵀ) x`` -> L2-normalise (the post-projection renorm is load-bearing).
        """
        if self.basis is None:
            raise RuntimeError("call build_basis() before debias().")
        b, c, h, w = fmap.shape
        x = F.normalize(fmap, p=2, dim=1).reshape(b, c, h * w)
        basis = self.basis.to(device=x.device, dtype=x.dtype)      # [C, s]
        proj = basis @ (basis.t() @ x)                             # [B, C, HW] positional component
        x = (x - proj).reshape(b, c, h, w)
        return F.normalize(x, p=2, dim=1)


@torch.no_grad()
def matchability(fmap: torch.Tensor, debias: "PositionalDebias | None" = None,
                 sample: int = 512) -> dict:
    """Matching-quality diagnostic reported with the matching arms (higher ``nn_coherence`` = more matchable).

    * ``positional_energy`` — fraction of feature variance in the positional subspace (0..1; high = position
      dominates appearance => matching is unreliable without debiasing).
    * ``nn_coherence`` — mean nearest-neighbour cosine among sampled patches after (optional) debiasing;
      structured/matchable features score higher than a near-uniform manifold.
    """
    b, c, h, w = fmap.shape
    x = F.normalize(fmap, p=2, dim=1).reshape(b, c, h * w)[0].t()   # [P, C]
    out: dict = {}
    if debias is not None and debias.basis is not None:
        basis = debias.basis.to(x.device, x.dtype)
        proj = (x @ basis)                                          # [P, s]
        total = (x * x).sum().clamp_min(1e-8)
        out["positional_energy"] = float((proj * proj).sum() / total)
        x = debias.debias(fmap).reshape(b, c, h * w)[0].t()
    p = x.shape[0]
    idx = torch.randperm(p, device=x.device)[:min(sample, p)]
    xs = F.normalize(x[idx], p=2, dim=1)
    sim = xs @ xs.t()
    sim.fill_diagonal_(-2.0)
    out["nn_coherence"] = float(sim.max(dim=1).values.mean())
    return out


class SelfSupportHead:
    """Self-Support FSS propagation (Fan et al., ECCV 2022) — query-specific prototype refinement.

    ``predict(feature_q, coarse_prob)`` turns confident query pixels into global + local self-support
    prototypes and re-scores by cosine similarity. Operates in the (optionally debiased) feature space.
    """

    def __init__(self, fg_thres: float = 0.7, bg_thres: float = 0.6, topk: int = 12,
                 temp: float = 10.0, attn_temp: float = 2.0):
        self.fg_thres, self.bg_thres, self.topk = fg_thres, bg_thres, topk
        self.temp, self.attn_temp = temp, attn_temp

    def _sim(self, feat_q, fg_proto, bg_proto):
        sf = F.cosine_similarity(feat_q, fg_proto, dim=1)
        sb = F.cosine_similarity(feat_q, bg_proto, dim=1)
        return torch.stack([sb, sf], dim=1) * self.temp  # [B, 2, H, W]

    def _self_protos(self, feat_q, prob_fg, prob_bg):
        b, c, h, w = feat_q.shape
        fg_g, bg_g, fg_l, bg_l = [], [], [], []
        for i in range(b):
            cur = feat_q[i].view(c, -1)                     # [C, HW]
            pf, pb = prob_fg[i].view(-1), prob_bg[i].view(-1)
            fg = cur[:, pf > self.fg_thres] if (pf > self.fg_thres).any() else cur[:, torch.topk(pf, self.topk).indices]
            bg = cur[:, pb > self.bg_thres] if (pb > self.bg_thres).any() else cur[:, torch.topk(pb, self.topk).indices]
            fg_g.append(fg.mean(-1)[None])
            bg_g.append(bg.mean(-1)[None])
            fn = F.normalize(fg, dim=0); bn = F.normalize(bg, dim=0)
            cn = F.normalize(cur, dim=0).t()                 # [HW, C]
            fg_l.append(((cn @ fn * self.attn_temp).softmax(-1) @ fg.t()).t().view(c, h, w)[None])
            bg_l.append(((cn @ bn * self.attn_temp).softmax(-1) @ bg.t()).t().view(c, h, w)[None])
        return (torch.cat(fg_g)[..., None, None], torch.cat(bg_g)[..., None, None],
                torch.cat(fg_l), torch.cat(bg_l))

    @torch.no_grad()
    def predict(self, feature_q: torch.Tensor, coarse_prob: torch.Tensor) -> torch.Tensor:
        """``feature_q`` [B,C,H,W], ``coarse_prob`` [B,H,W] first-pass FG prob -> refined logits [B,2,H,W]."""
        pf = coarse_prob
        pb = 1.0 - coarse_prob
        ssfp, ssbp, asfp, asbp = self._self_protos(feature_q, pf, pb)
        # local prototypes are pixel-wise maps; global are single vectors. Mixing weights follow the
        # reference implementation.
        fp = 0.5 * ssfp + 0.5 * asfp                          # [B,C,H,W]
        bp = 0.3 * ssbp + 0.7 * asbp
        return self._sim(feature_q, fp, bp)
