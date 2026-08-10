"""Mask-aware FINO guide losses + the GRL M+/M- gradient-direction contract (CPU, torch)."""

from __future__ import annotations

import torch
from torch.autograd import Function

from em_ssl.fino.guide_losses import compute_classification_loss, compute_regression_loss

class _GRL(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.lam, None

class _Head(torch.nn.Module):
    """Minimal upstream-equivalent guide head: GRL wrapper + a linear layer."""

    def __init__(self, d, out):
        super().__init__()
        self.lin = torch.nn.Linear(d, out)

    def forward(self, x, lam=0.0):
        return self.lin(_GRL.apply(x, lam))

def _cls_input(n_crops=2, B=4, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_crops, B, D, generator=g)

def test_classification_masking_excludes_invalid_samples():
    torch.manual_seed(0)
    head = _Head(8, 3)
    cls = _cls_input(D=8)
    labels = torch.tensor([0, 2, -1, 1])  # 3rd is missing
    valid = torch.tensor([True, True, False, True])
    loss, stats = compute_classification_loss(head, torch.nn.CrossEntropyLoss(), cls, labels, valid, lambda_value=1.0)
    assert stats["n_valid"] == 3 and stats["frac_valid"] == 0.75
    assert torch.isfinite(loss) and loss.requires_grad
    # Changing the masked-out sample's features must not change the loss.
    cls2 = cls.clone()
    cls2[:, 2, :] += 100.0
    loss2, _ = compute_classification_loss(head, torch.nn.CrossEntropyLoss(), cls2, labels, valid, lambda_value=1.0)
    assert torch.allclose(loss, loss2, atol=1e-5)

def test_classification_zero_valid_keeps_graph():
    head = _Head(8, 3)
    cls = _cls_input(D=8)
    labels = torch.full((4,), -1)
    valid = torch.zeros(4, dtype=torch.bool)
    loss, stats = compute_classification_loss(head, torch.nn.CrossEntropyLoss(), cls, labels, valid, lambda_value=1.0)
    assert stats["n_valid"] == 0 and float(loss.detach()) == 0.0 and loss.requires_grad
    loss.backward()  # must not error; head grads are zero
    assert head.lin.weight.grad is not None and torch.count_nonzero(head.lin.weight.grad) == 0

def test_regression_masking_and_mse():
    head = _Head(8, 1)
    cls = _cls_input(D=8)
    labels = torch.tensor([0.5, -0.3, 9.9, 0.1])
    valid = torch.tensor([True, True, False, True])
    loss, stats = compute_regression_loss(head, torch.nn.MSELoss(), cls, labels, valid, lambda_value=1.0)
    assert stats["n_valid"] == 3 and torch.isfinite(loss) and loss.requires_grad

def test_grl_sign_flips_backbone_gradient():
    """M+ (lambda>0) vs M- (lambda<0): identical loss value, reversed gradient to the features."""
    head = _Head(8, 3)
    labels = torch.tensor([0, 1, 2, 0])
    valid = torch.ones(4, dtype=torch.bool)

    cls_p = _cls_input(D=8).requires_grad_(True)
    loss_p, _ = compute_classification_loss(head, torch.nn.CrossEntropyLoss(), cls_p, labels, valid, lambda_value=1.0)
    loss_p.backward()
    grad_pos = cls_p.grad.clone()

    cls_m = cls_p.detach().clone().requires_grad_(True)
    loss_m, _ = compute_classification_loss(head, torch.nn.CrossEntropyLoss(), cls_m, labels, valid, lambda_value=-1.0)
    loss_m.backward()
    grad_neg = cls_m.grad.clone()

    assert torch.allclose(loss_p, loss_m, atol=1e-6)        # GRL forward is identity
    assert torch.allclose(grad_neg, -grad_pos, atol=1e-6)   # backbone gradient reversed
