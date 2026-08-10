"""Gradient-reversal layer (DANN) — the "non-spurious-style" option.

A domain adversary predicts source identity / modality from the style code; the gradient-reversal
layer (GRL) negates the adversary's gradient into the style encoder, so the code is pushed to be
un-predictive of source — discouraging it from encoding spurious source identity while keeping the
appearance signal FiLM needs.

Reference (verbatim GRL): Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation",
ICML 2015 (arXiv:1409.7495); Ganin et al., JMLR 2016. Canonical PyTorch port:
https://github.com/fungtion/DANN_py3/blob/master/functions.py

Torch-only (no GPU needed).
"""

from __future__ import annotations

import math

import torch
from torch.autograd import Function


class _ReverseGrad(Function):
    """Identity in forward; negate-and-scale-by-alpha in backward (the canonical DANN GRL)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)  # fresh view so autograd reliably inserts the custom node

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None  # None: no grad w.r.t. the (float) alpha


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply the gradient-reversal layer with coefficient ``alpha`` (== DANN lambda)."""
    return _ReverseGrad.apply(x, alpha)


def dann_lambda(progress: float, gamma: float = 10.0, lambda_max: float = 1.0) -> float:
    """DANN GRL coefficient schedule: ``lambda = lambda_max * (2 / (1 + exp(-gamma * p)) - 1)``.

    ``progress`` is global training progress in [0, 1] (0 at the first optimizer step, ~1 at the last),
    not per-epoch — resetting it re-shocks the feature extractor. ``gamma=10`` is the paper default;
    the ramp avoids the untrained adversary's noisy gradients destabilising the code early. ``lambda_max``
    scales the whole ramp (``cfg.cond.grad_reversal`` sets it; 0 disables the adversary entirely).
    """
    p = min(max(float(progress), 0.0), 1.0)
    return float(lambda_max) * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
