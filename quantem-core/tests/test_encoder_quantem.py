"""Guard tests for the QuantEM encoder load — the two silent-corruption failure modes.

Both need the original pretraining checkpoint, so they skip unless ``QUANTEM_REF_CKPT`` points at
one (on this machine: ``V:\\Chris\\m1_checkpoints\\m1_teacher_674999.pth``). The architecture-only
tests below run anywhere timm is installed.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from quantem_em.models.encoders.quantem_vit import (  # noqa: E402
    build_quantem_encoder,
    load_reference_checkpoint,
    remap_reference_state_dict,
)
from quantem_em.registry import QUANTEM_VITB  # noqa: E402

REF = os.environ.get("QUANTEM_REF_CKPT")
needs_ckpt = pytest.mark.skipif(not REF, reason="set QUANTEM_REF_CKPT to the m1_teacher checkpoint")


# --- architecture-only -------------------------------------------------------------------


def test_prefix_tokens_are_five():
    """1 CLS + 4 storage tokens. A wrong value silently corrupts LoRA token splitting."""
    m = build_quantem_encoder(QUANTEM_VITB, None, img_size=512)
    assert m.num_prefix_tokens == 5


def test_k_bias_is_a_persistent_buffer_after_build():
    """timm registers k_bias non-persistently and omits it from state_dict; we must not."""
    m = build_quantem_encoder(QUANTEM_VITB, None, img_size=512)
    sd = m.state_dict()
    for i in range(len(m.blocks)):
        assert f"blocks.{i}.attn.k_bias" in sd, "k_bias would not round-trip through save/load"


def test_rope_periods_buffer_is_true_bfloat16():
    """T11. The *dtype* matters, not just the values.

    timm derives the working precision of the whole rotary embedding from this buffer
    (``dtype = self.periods.dtype`` in ``_get_pos_embed_from_coords``). Holding bf16-rounded values
    in a float32 buffer makes timm compute the angles and sin/cos in float32, which differs from
    the reference by ~1e-2 and shifts the output probability by ~1e-1.
    """
    m = build_quantem_encoder(QUANTEM_VITB, None, img_size=512)
    assert m.rope.periods.dtype is torch.bfloat16


def test_layernorm_epsilon_matches_the_reference():
    """DINOv3 offers `layernorm` (1e-6) and `layernormbf16` (1e-5). Ours trained with 1e-6.

    timm's ``vit_base_patch16_dinov3`` hard-codes 1e-5 -- correct for Meta's released weights,
    wrong for ours. Getting this wrong perturbs all 25 LayerNorms and raises nothing.
    """
    m = build_quantem_encoder(QUANTEM_VITB, None, img_size=512)
    assert m.norm.eps == 1e-6
    for blk in m.blocks:
        assert blk.norm1.eps == 1e-6
        assert blk.norm2.eps == 1e-6


def test_attention_uses_reference_rope_dtype_semantics():
    """The reference casts q and k -- prefix tokens included -- down to the rope dtype."""
    m = build_quantem_encoder(QUANTEM_VITB, None, img_size=512)
    for blk in m.blocks:
        assert type(blk.attn).__name__ == "_ReferenceRopeAttention"

    # And it must actually bite: with a bf16 rope, the rotated queries are bf16-representable.
    seen = {}
    a = m.blocks[0].attn
    orig = a.forward

    def spy(x, rope=None, **kw):
        seen["rope_dtype"] = None if rope is None else rope.dtype
        return orig(x, rope=rope, **kw)

    a.forward = spy
    with torch.no_grad():
        m.forward_intermediates(
            torch.randn(1, 1, 512, 512),
            indices=[0],
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
    assert seen["rope_dtype"] is torch.bfloat16


def test_forward_intermediates_shapes():
    m = build_quantem_encoder(QUANTEM_VITB, None, img_size=512)
    with torch.no_grad():
        feats = m.forward_intermediates(
            torch.randn(1, 1, 512, 512),
            indices=[8, 9, 10, 11],
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
    assert len(feats) == 4
    assert all(tuple(f.shape) == (1, 768, 32, 32) for f in feats)


def test_dynamic_grid_supports_non_square():
    """RoPE recomputes per input, so tile size is not baked in at build time."""
    m = build_quantem_encoder(QUANTEM_VITB, None, img_size=512)
    with torch.no_grad():
        f = m.forward_intermediates(
            torch.randn(1, 1, 512, 768),
            indices=[11],
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
    assert tuple(f[0].shape) == (1, 768, 32, 48)


# --- against the real checkpoint ---------------------------------------------------------


@needs_ckpt
def test_every_trained_tensor_lands():
    sd = load_reference_checkpoint(REF, QUANTEM_VITB)
    # strict=True raises on any missing/unexpected key.
    build_quantem_encoder(QUANTEM_VITB, sd, img_size=512, strict=True)


@needs_ckpt
def test_k_bias_matches_checkpoint_and_is_large():
    """T10: timm's own converter discards this. Ours is live and it is the LARGEST bias term."""
    raw = torch.load(REF, map_location="cpu", weights_only=False)["teacher"]
    src = {k[len("backbone.") :]: v for k, v in raw.items() if k.startswith("backbone.")}
    m = build_quantem_encoder(QUANTEM_VITB, remap_reference_state_dict(src), img_size=512)

    assert not any("bias_mask" in k for k in src), (
        "checkpoint has bias_mask keys -> trained with mask_k_bias=True, k-bias would be inert"
    )
    for i in range(len(m.blocks)):
        want = src[f"blocks.{i}.attn.qkv.bias"].chunk(3, dim=-1)[1]
        got = m.blocks[i].attn.k_bias.detach().cpu()
        assert torch.allclose(got.float(), want.float()), f"block {i} k_bias not loaded"
        assert got.abs().max() > 1.0, f"block {i} k_bias is ~zero; expected O(1-7)"


@needs_ckpt
def test_rope_periods_match_the_reference_exactly():
    sd = load_reference_checkpoint(REF, QUANTEM_VITB)
    ref_periods = sd["__rope_periods__"].float()
    m = build_quantem_encoder(QUANTEM_VITB, sd, img_size=512)
    assert torch.equal(m.rope.periods.detach().cpu().float(), ref_periods)

    # And the naive path (regenerate fp32, don't truncate) would NOT match -- proving the fix is
    # load-bearing rather than cosmetic.
    fresh = build_quantem_encoder(QUANTEM_VITB, None, img_size=512)
    fresh.rope._init_weights() if hasattr(fresh.rope, "_init_weights") else None
