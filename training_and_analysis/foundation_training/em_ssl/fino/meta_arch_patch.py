"""Graft EM partial-metadata masking onto the upstream ``GuidedSSLMetaArch``.

The official FINO meta-arch (``dinov3.train.guided_ssl_meta_arch.GuidedSSLMetaArch``) and the
``dinov3.train.metadata_utils`` heads/GRL/schedule are used directly rather than re-implemented.
The single behavioural change the EM corpus requires is per-sample masking of guide
losses for tiles whose metadata is missing (upstream assumes complete metadata).

``apply_fino_grafts()`` replaces ``GuidedSSLMetaArch._compute_guide_losses`` with a mask-aware
version that:

  * reads a parallel ``<name>_valid`` mask off the collated metadata dataclass and computes each
    guide loss / metric over valid samples only (via :mod:`em_ssl.fino.guide_losses`);
  * still emits the upstream ``guide_<name>_weighted`` keys, so the unchanged
    ``GuidedSSLMetaArch.forward_backward`` (and its optional grad-norm normalization for >=2
    guides) keeps working;
  * adds compact ``fino/lambda``, ``fino/<name>_*`` and ``fino/total_loss`` scalar metrics (the
    metadata-metric vocabulary) for the metric logger.

Nothing here imports dinov3 at module load (consistent with ``em_ssl.integration.dinov3_patch``);
the graft is idempotent and a no-op on a base without the FINO branch.
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger("em_ssl.fino")

def _em_compute_guide_losses(self, *, student_global, teacher_global, metadata, iteration):
    """Mask-aware replacement for GuidedSSLMetaArch._compute_guide_losses."""
    from dinov3.train.metadata_utils import compute_lambda, compute_prototypical_loss

    from .guide_losses import compute_classification_loss as em_cls
    from .guide_losses import compute_regression_loss as em_reg

    loss_dict: dict = {}
    lambda_value = compute_lambda(
        iteration,
        self._total_iterations,
        self._lambda_schedule_type,
        warmup_steps=self._lambda_warmup_steps,
    )
    loss_dict["guide_lambda"] = lambda_value
    loss_dict["fino/lambda"] = float(lambda_value)

    cls_pre_head = student_global["cls_pre_head"]
    total = 0.0

    for guide_cfg in self.guide_configs:
        name = guide_cfg.name
        head = self.guide_heads[name]
        loss_fn = self.guide_loss_fns[name]

        labels = getattr(metadata, name).cuda(non_blocking=True)
        valid = getattr(metadata, f"{name}_valid", None)
        valid = valid.cuda(non_blocking=True) if valid is not None else None

        effective_lambda = -lambda_value if guide_cfg.grl else lambda_value
        loss_dict[f"guide_{name}_lambda"] = effective_lambda

        if guide_cfg.grl and guide_cfg.grl_space == "prototype":
            cls_input = student_global["cls_after_head"]
        else:
            cls_input = cls_pre_head

        if guide_cfg.method == "regression":
            guide_loss, stats = em_reg(head, loss_fn, cls_input, labels, valid, effective_lambda)
            loss_dict[f"guide_{name}_mse"] = stats["mse"]
            loss_dict[f"fino/{name}_mse"] = stats["mse"]
        elif guide_cfg.method == "prototypical":
            # Unused by the reported runs; falls back to the upstream (unmasked) path so a
            # prototypical guide still runs when the configuration enables one.
            teacher_cls = teacher_global["cls_pre_head"]
            guide_loss, accuracy = compute_prototypical_loss(
                head, cls_input, labels, effective_lambda, teacher_cls_input=teacher_cls
            )
            stats = {"n_valid": int(labels.shape[0]), "frac_valid": 1.0}
            loss_dict[f"guide_{name}_accuracy"] = accuracy
            loss_dict[f"fino/{name}_acc"] = accuracy
        else:
            guide_loss, stats = em_cls(
                head, loss_fn, cls_input, labels, valid, effective_lambda, use_bce=guide_cfg.use_bce
            )
            loss_dict[f"guide_{name}_accuracy"] = stats["acc"]
            loss_dict[f"fino/{name}_acc"] = stats["acc"]

        effective_weight = guide_cfg.loss_weight / 10.0 if guide_cfg.grl else guide_cfg.loss_weight
        loss_dict[f"guide_{name}_loss"] = guide_loss
        loss_dict[f"guide_{name}_weighted"] = effective_weight * guide_loss

        # Compact EM/task metric vocabulary (all Python scalars; the upstream DINOv3 training loop
        # all-reduces every metric). guide_<name>_weighted/loss above stay tensors — forward_backward sums the
        # weighted ones for backprop — but these fino/* aliases are logging-only, so they are made
        # plain floats for consistency with the rest of the dict.
        loss_dict[f"fino/{name}_loss"] = float(guide_loss.detach()) if hasattr(guide_loss, "detach") else float(guide_loss)
        loss_dict[f"fino/{name}_weight"] = float(effective_weight)
        loss_dict[f"fino/{name}_lambda"] = float(effective_lambda)
        loss_dict[f"fino/{name}_grl"] = 1.0 if guide_cfg.grl else 0.0
        loss_dict[f"fino/{name}_n_valid"] = int(stats.get("n_valid", 0))
        loss_dict[f"fino/{name}_frac_valid"] = float(stats.get("frac_valid", 0.0))

        weighted = effective_weight * guide_loss
        total = weighted if isinstance(total, float) else total + weighted

    loss_dict["fino/total_loss"] = float(total.detach()) if hasattr(total, "detach") else float(total)
    return loss_dict

def apply_fino_grafts() -> bool:
    """Install the mask-aware guide-loss graft on GuidedSSLMetaArch (idempotent).

    Returns True if grafted, False if the FINO meta-arch is unavailable (non-FINO base) — in
    which case ordinary (non-guided) EM training is unaffected and only FINO runs would fail
    later, with a clear error.
    """
    try:
        from dinov3.train.guided_ssl_meta_arch import GuidedSSLMetaArch
    except Exception as exc:  # pragma: no cover - only on a non-FINO base
        warnings.warn(
            f"FINO guided meta-arch unavailable ({exc!r}); metadata-guided training disabled. "
            "Pin the DINOv3 FINO branch (third_party/dinov3.pin) and run third_party/fetch_dinov3.sh."
        )
        return False
    if getattr(GuidedSSLMetaArch._compute_guide_losses, "_em_masked", False):
        return True
    _em_compute_guide_losses._em_masked = True  # type: ignore[attr-defined]
    GuidedSSLMetaArch._compute_guide_losses = _em_compute_guide_losses
    logger.info("Installed EM mask-aware GuidedSSLMetaArch._compute_guide_losses graft.")
    return True
