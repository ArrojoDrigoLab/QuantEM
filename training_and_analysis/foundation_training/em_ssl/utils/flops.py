"""Analytic FLOP model for the DINOv3 multi-crop SSL step.

Why analytic (not a live profiler): ``torch.utils.flop_counter.FlopCounterMode`` is blind inside
``torch.compile`` graphs (the aten ops are fused into Triton kernels and never dispatched through
the mode), and full compile is the default. FLOPs are compile-independent, so a closed-form from
the config is the robust runtime workhorse; the counter is only useful as an offline eager
cross-check.

Conventions:
  * FLOPs counts multiply-adds as 2 FLOPs.
  * "per step" = one optimizer step, aggregated over all GPUs (global batch). It is the total
    compute spent, and is the quantity equalized across arms.
  * ``model_flops`` = the math the model does (fwd + bwd ≈ 3× fwd for grad paths, 1× for the
    no-grad teacher). ``hw_flops`` is the GPU-cost proxy and the standard axis: when
    ``activation_checkpointing`` is on it adds the forward that is recomputed on the backbone in the
    backward (≈ 4× fwd there); with it off the two coincide.
  * Captured: backbone (student 2 global + N local, teacher 2 global), DINO head (CLS tokens),
    iBOT head (expected masked patches). Omitted (<~1% total): patch-embed conv, LayerNorms,
    Sinkhorn-Knopp, KoLeo, the loss elementwise ops.
"""

from __future__ import annotations

from dataclasses import dataclass

_GRAD = 3.0  # fwd + bwd ≈ 3× fwd (no activation-ckpt recompute)
_GRAD_CKPT = 4.0  # + one recomputed forward in the backward


@dataclass(frozen=True)
class FlopBreakdown:
    model_flops_per_step: float  # aggregate, fwd+bwd, no act-ckpt recompute
    hw_flops_per_step: float  # + backbone act-ckpt recompute where enabled (standard axis)
    backbone_model: float
    dino_head: float
    ibot_head: float
    teacher: float  # teacher backbone fwd (no-grad), already inside backbone_model
    n_masked_per_global_crop: float

    @property
    def model_pflops_per_step(self) -> float:
        return self.model_flops_per_step / 1e15

    @property
    def hw_pflops_per_step(self) -> float:
        return self.hw_flops_per_step / 1e15


def _vit_fwd_flops(n_tokens: float, dim: int, depth: int, ffn_ratio: float) -> float:
    """Forward FLOPs for one crop's token sequence through a ViT backbone.

    Per layer: projections QKVO + MLP = (8 + 4·r)·N·d² ; attention scores+context = 4·N²·d.
    """
    proj = (8.0 + 4.0 * ffn_ratio) * n_tokens * dim * dim
    attn = 4.0 * n_tokens * n_tokens * dim
    return depth * (proj + attn)


def _head_fwd_flops_per_vector(in_dim: int, hidden: int, bottleneck: int, prototypes: int, nlayers: int) -> float:
    """Forward FLOPs for one token through a DINOHead (MLP -> L2norm -> prototype linear)."""
    if nlayers <= 1:
        macs = in_dim * bottleneck
    else:
        macs = in_dim * hidden + hidden * hidden * max(0, nlayers - 2) + hidden * bottleneck
    macs += bottleneck * prototypes  # last_layer (the big one: prototypes can be 65k+)
    return 2.0 * macs


def dinov3_flops_per_step(
    *,
    embed_dim: int,
    depth: int,
    ffn_ratio: float,
    patch_size: int,
    global_crops_size: int,
    local_crops_size: int,
    n_local_crops: int,
    batch_size_per_gpu: int,
    n_gpus: int,
    dino_hidden: int,
    dino_bottleneck: int,
    dino_prototypes: int,
    dino_nlayers: int,
    ibot_hidden: int,
    ibot_bottleneck: int,
    ibot_prototypes: int,
    ibot_nlayers: int,
    mask_ratio_min: float,
    mask_ratio_max: float,
    mask_sample_probability: float,
    n_storage_tokens: int = 0,
    n_global_crops: int = 2,
    activation_checkpointing: bool = False,
) -> FlopBreakdown:
    """Closed-form FLOPs for one DINOv3 multi-crop optimizer step (aggregate over GPUs)."""
    p = patch_size
    n_patches_g = (global_crops_size // p) ** 2
    n_tok_g = n_patches_g + 1 + n_storage_tokens  # + cls + registers
    n_tok_l = (local_crops_size // p) ** 2 + 1 + n_storage_tokens

    f_g = _vit_fwd_flops(n_tok_g, embed_dim, depth, ffn_ratio)
    f_l = _vit_fwd_flops(n_tok_l, embed_dim, depth, ffn_ratio)
    student_bb_fwd = n_global_crops * f_g + n_local_crops * f_l
    teacher_bb_fwd = n_global_crops * f_g  # no-grad, forward only

    backbone_model = _GRAD * student_bb_fwd + teacher_bb_fwd
    grad_bb = _GRAD_CKPT if activation_checkpointing else _GRAD
    backbone_hw = grad_bb * student_bb_fwd + teacher_bb_fwd

    dino_vec = _head_fwd_flops_per_vector(embed_dim, dino_hidden, dino_bottleneck, dino_prototypes, dino_nlayers)
    # DINO head runs on CLS tokens: student (global + local) and teacher (global).
    dino_head = _GRAD * (n_global_crops + n_local_crops) * dino_vec + n_global_crops * dino_vec

    ibot_vec = _head_fwd_flops_per_vector(embed_dim, ibot_hidden, ibot_bottleneck, ibot_prototypes, ibot_nlayers)
    mean_ratio = 0.5 * (mask_ratio_min + mask_ratio_max)
    n_masked = n_global_crops * mask_sample_probability * mean_ratio * n_patches_g  # expected, per sample
    ibot_head = _GRAD * n_masked * ibot_vec + n_masked * ibot_vec  # student (grad) + teacher (no-grad)

    gb = batch_size_per_gpu * max(1, n_gpus)
    per_sample_model = backbone_model + dino_head + ibot_head
    per_sample_hw = backbone_hw + dino_head + ibot_head  # heads are not activation-checkpointed
    return FlopBreakdown(
        model_flops_per_step=per_sample_model * gb,
        hw_flops_per_step=per_sample_hw * gb,
        backbone_model=backbone_model * gb,
        dino_head=dino_head * gb,
        ibot_head=ibot_head * gb,
        teacher=teacher_bb_fwd * gb,
        n_masked_per_global_crop=mask_sample_probability * mean_ratio * n_patches_g,
    )


