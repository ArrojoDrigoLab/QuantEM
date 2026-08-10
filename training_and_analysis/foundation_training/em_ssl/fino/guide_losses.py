"""Mask-aware FINO guide losses (EM extension over the upstream reference).

The upstream ``dinov3.train.metadata_utils.guide_losses`` assumes *every* sample carries a
valid metadata label (FMoW/HPA). The EM corpus has partial metadata, so these variants
take a per-sample ``valid_mask`` and compute the loss/metric over valid samples only, while
keeping every crop's head output in the autograd graph (so the head still receives a — zero —
gradient when a batch has no valid samples, which matters under FSDP).

Head-agnostic: ``head`` is any callable ``head(x, lambda_) -> logits`` (the upstream
``Classifier`` / ``Regressor``). The sign of ``lambda_value`` selects M+ (auxiliary, >=0) vs
M- (adversarial GRL, <0); the masking here is orthogonal to that.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

def _as_mask(valid_mask: Tensor | None, n: int, device) -> Tensor:
    if valid_mask is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    return valid_mask.bool()

def compute_classification_loss(
    head: nn.Module,
    loss_fn: nn.Module,
    cls_input: Tensor,
    labels: Tensor,
    valid_mask: Tensor | None,
    lambda_value: float,
    use_bce: bool = False,
) -> tuple[Tensor, dict]:
    """Masked classification loss averaged over all crops; accuracy read off the first crop.

    Args:
        head: guide head, ``head(x, lambda_) -> [B, C]``.
        loss_fn: ``CrossEntropyLoss`` (or ``BCEWithLogitsLoss`` when ``use_bce``).
        cls_input: CLS pre-head embeddings ``[n_crops, B, D]``.
        labels: ``[B]`` long class indices, or ``[B, C]`` multi-hot when ``use_bce``; entries with
            valid=False are ignored either way.
        valid_mask: ``[B]`` bool (None -> all valid).
        lambda_value: gradient-scaling lambda (sign selects M+/M-).
    Returns ``(loss, {"acc", "n_valid", "frac_valid"})``.
    """
    B = labels.shape[0]
    mask = _as_mask(valid_mask, B, labels.device)
    n_valid = int(mask.sum().item())
    outputs = [head(crop_cls, lambda_value) for crop_cls in cls_input]

    if n_valid == 0:
        loss = sum(o.float().sum() * 0.0 for o in outputs) / len(outputs)
        return loss, {"acc": 0.0, "n_valid": 0, "frac_valid": 0.0}

    if use_bce:
        yv = labels[mask].float()
        loss = sum(loss_fn(o.float()[mask], yv) for o in outputs) / len(outputs)
        with torch.no_grad():
            pred = (torch.sigmoid(outputs[0].float()[mask]) > 0.5).float()
            acc = ((pred == yv).sum(dim=1) == labels.shape[1]).float().mean().item()
    else:
        yv = labels[mask]
        loss = sum(loss_fn(o.float()[mask], yv) for o in outputs) / len(outputs)
        with torch.no_grad():
            pred = outputs[0].float()[mask].argmax(dim=1)
            acc = (pred == yv).float().mean().item()

    return loss, {"acc": float(acc), "n_valid": n_valid, "frac_valid": n_valid / max(1, B)}

def compute_regression_loss(
    head: nn.Module,
    loss_fn: nn.Module,
    cls_input: Tensor,
    labels: Tensor,
    valid_mask: Tensor | None,
    lambda_value: float,
) -> tuple[Tensor, dict]:
    """Masked regression loss averaged over all crops; MSE read off the first crop.

    ``labels`` are ``[B]`` or ``[B, n_outputs]`` (already standardised upstream in
    :meth:`em_ssl.fino.factors.FinoFactorSpec.encode_value`); a head exposing
    ``normalize_targets`` re-normalises them here before the loss. Returns
    ``(loss, {"mse", "n_valid", "frac_valid"})``.
    """
    B = labels.shape[0]
    mask = _as_mask(valid_mask, B, labels.device)
    n_valid = int(mask.sum().item())

    labels_float = labels.float()
    if labels_float.dim() == 1:
        labels_float = labels_float.unsqueeze(-1)
    reg = head.module if hasattr(head, "module") else head
    if hasattr(reg, "normalize_targets"):
        labels_float = reg.normalize_targets(labels_float)

    outputs = [head(crop_cls, lambda_value) for crop_cls in cls_input]

    if n_valid == 0:
        loss = sum(o.float().sum() * 0.0 for o in outputs) / len(outputs)
        return loss, {"mse": 0.0, "n_valid": 0, "frac_valid": 0.0}

    yv = labels_float[mask]
    loss = sum(loss_fn(o.float()[mask], yv) for o in outputs) / len(outputs)
    with torch.no_grad():
        mse = loss_fn(outputs[0].float()[mask], yv).item()
    return loss, {"mse": float(mse), "n_valid": n_valid, "frac_valid": n_valid / max(1, B)}
