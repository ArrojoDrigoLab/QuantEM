"""The QuantEM ViT-B/16 encoder, built on timm's Apache-2.0 DINOv3 implementation.

Why timm rather than ``facebookresearch/dinov3``
------------------------------------------------
The checkpoint is only weights; the architecture class has to come from somewhere. The research
harness gets it via ``importlib.import_module("dinov3.models.vision_transformer")``, but that
package is not on PyPI, is fetched by a shell script, and carries a custom Meta licence — and PyPI
rejects any distribution declaring a URL dependency (``Can't have direct dependency``), so it
cannot even be expressed as a requirement.

timm implements the same architecture in ``timm/models/eva.py``, whose header states that the
DINOv3 code is a modification of the EVA model and is therefore Apache-2.0 like the rest of timm,
with only *Meta's weights* remaining under the DINOv3 licence. We ship our own weights, so nothing
of Meta's is redistributed. This is also not a new integration for this project: the Fig. 2
Meta-DINOv3 baselines already ran through timm (``foundation_baselines/external_vit.py``:
``dinov3_meta_vitb`` -> ``vit_base_patch16_dinov3.lvd1689m``).

Two adaptations are mandatory; both are verified by tests in ``tests/test_encoder_quantem.py``.

T10 — the attention k-bias
    ``timm.models.eva.EvaAttention`` makes ``q_bias``/``v_bias`` parameters but registers
    ``k_bias`` as a **non-persistent buffer zeroed at init**; it is absent from ``state_dict()``
    entirely, and timm's own ``checkpoint_filter_fn`` discards the k third of ``qkv.bias``. That is
    correct for Meta's distilled releases, which were trained with ``mask_k_bias=True`` and whose
    k-biases are all zero. **It is wrong here.** Our checkpoint has no ``bias_mask`` keys, so it was
    trained with ``mask_k_bias=False`` and the k-bias was live for all 675,000 steps — measured
    max |k| = 5.12 (block 0), 7.00 (block 5), 4.67 (block 11), consistently the largest of the
    three. Loading through timm's filter would silently zero 9,216 trained values.

T11 — the RoPE period buffer
    The reference stores ``rope_embed.periods`` as **bfloat16** (``pos_embed_rope_dtype`` defaults
    to ``"bf16"``); timm regenerates them in float32 from ``rope_temperature=100`` and skips the
    checkpoint's buffer. The difference is not negligible (max 8.0e-2). Truncating timm's fp32
    values through bfloat16 reproduces the reference exactly (measured delta 0.0).

Not a problem, despite appearances: ``M1_dinov3_vitb_512.yaml`` sets ``rope_rescale_coords: 2.0``,
but Meta gates ``shift``/``jitter``/``rescale_coords`` on ``self.training``, so all three are
no-ops under ``.eval()``. timm's author left the knob unexposed for the same reason.
"""

from __future__ import annotations

import torch

from ...spec import EncoderSpec

#: Keys present in the pretraining checkpoint that no inference path uses.
_PRETRAIN_ONLY = ("mask_token",)


def remap_reference_state_dict(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a DINOv3-format backbone state dict to timm ``Eva`` naming.

    ``src`` keys are already stripped of the ``teacher.``/``backbone.`` prefixes.

    Deliberately *not* ``timm.models.eva.checkpoint_filter_fn``: that helper drops the k-bias
    (see T10) and discards ``rope_embed.periods``. This keeps every trained tensor.

    The period buffer is returned under its timm name so callers can apply it explicitly; it is
    not part of ``load_state_dict`` because timm registers it non-persistently.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in src.items():
        if k in _PRETRAIN_ONLY:
            continue
        if k == "rope_embed.periods":
            out["__rope_periods__"] = v
            continue
        if k == "storage_tokens":
            out["reg_token"] = v
            continue
        if k.endswith("attn.qkv.bias"):
            q, kb, vb = v.chunk(3, dim=-1)
            base = k[: -len("qkv.bias")]
            out[base + "q_bias"] = q
            out[base + "k_bias"] = kb  # T10: kept, not discarded
            out[base + "v_bias"] = vb
            continue
        out[k.replace("ls1.gamma", "gamma_1").replace("ls2.gamma", "gamma_2")] = v
    return out


def build_quantem_encoder(
    spec: EncoderSpec,
    state_dict: dict[str, torch.Tensor] | None = None,
    *,
    img_size: int = 512,
    strict: bool = True,
):
    """Build the QuantEM ViT-B and, if given, load ``state_dict`` (already in timm naming).

    Raises on any missing or unexpected key when ``strict`` — silent partial loads are the
    failure mode this whole module exists to prevent.
    """
    from functools import partial

    import timm
    from torch import nn

    kwargs = {}
    if spec.norm_eps is not None:
        # timm's dinov3 entries hard-code eps=1e-5 (DINOv3's `layernormbf16`); our encoder trained
        # with the `layernorm` default of 1e-6. Leaving timm's value in place puts the wrong
        # epsilon in all 25 LayerNorms -- two per block plus the final one -- which perturbs every
        # block output and never raises anything.
        kwargs["norm_layer"] = partial(nn.LayerNorm, eps=spec.norm_eps)

    model = timm.create_model(
        spec.timm_model,
        pretrained=False,
        in_chans=spec.in_chans,
        img_size=img_size,
        num_classes=0,
        **kwargs,
    )

    # T10: promote k_bias to a persistent buffer so our trained values load here and round-trip
    # through any subsequent state_dict()/safetensors save.
    for blk in model.blocks:
        attn = blk.attn
        attn.register_buffer("k_bias", torch.zeros_like(attn.q_bias), persistent=True)

    if state_dict is not None:
        sd = dict(state_dict)
        periods = sd.pop("__rope_periods__", None)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "unclean QuantEM encoder load (silent-corruption guard): "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )

        if periods is not None:
            _set_rope_periods(model, periods)
        elif spec.rope_periods_bf16:
            _set_rope_periods(model, None)
    elif spec.rope_periods_bf16:
        _set_rope_periods(model, None)

    _assert_prefix_tokens(model, spec)
    return model.eval()


def _install_reference_attention(model) -> None:
    """Give every block the reference's rotary-embedding dtype semantics.

    ``dinov3/layers/attention.py::SelfAttention.apply_rope`` does this::

        q_dtype, k_dtype = q.dtype, k.dtype
        rope_dtype = sin.dtype                     # bfloat16
        q = q.to(dtype=rope_dtype)                 # the WHOLE tensor, prefix tokens included
        k = k.to(dtype=rope_dtype)
        q_prefix = q[:, :, :prefix, :]             # prefix sliced from the already-bf16 tensor
        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        q = q.to(dtype=q_dtype)                    # back to float32

    timm rotates only the non-prefix slice and leaves the prefix at float32. Two differences follow,
    and both matter: every query and key is quantised to bfloat16 in the reference, and the CLS and
    storage tokens are quantised too even though they are never rotated. Those five tokens are
    attended to by every patch token, so the discrepancy spreads across the whole map.

    Measured effect of ignoring this: output probabilities differ from the reference by up to 1e-1,
    which flips pixels near the 0.5 threshold. Everything else about the attention is already
    identical -- both sides call ``F.scaled_dot_product_attention`` on the same q, k, v.

    Only the rotary section is reimplemented; the projections and the attention call are timm's.
    """
    import torch.nn.functional as F
    from timm.layers import apply_rot_embed_cat
    from timm.models.eva import EvaAttention

    class _ReferenceRopeAttention(EvaAttention):
        def forward(self, x, rope=None, attn_mask=None, is_causal=False):
            b, n, c = x.shape
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            if rope is not None:
                npt = self.num_prefix_tokens
                q_dtype, k_dtype, rd = q.dtype, k.dtype, rope.dtype
                q, k = q.to(rd), k.to(rd)  # whole tensor, prefix included
                q = torch.cat(
                    [q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=True)],
                    dim=2,
                ).to(q_dtype)
                k = torch.cat(
                    [k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=True)],
                    dim=2,
                ).to(k_dtype)

            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
            return self.proj(x.transpose(1, 2).reshape(b, n, c))

    for blk in model.blocks:
        a = blk.attn
        # These are Identity for the released configuration; the override skips them, so refuse
        # rather than silently dropping a real module if timm's defaults ever change.
        for attr in ("q_norm", "k_norm", "norm"):
            m = getattr(a, attr, None)
            if m is not None and not isinstance(m, torch.nn.Identity):
                raise RuntimeError(
                    f"unexpected active {attr} in EvaAttention; refusing to override"
                )
        if getattr(a, "gate", None) is not None:
            raise RuntimeError("unexpected gated EvaAttention; refusing to override")
        if a.qkv is None or a.q_bias is None:
            raise RuntimeError("expected a fused qkv with separate q/k/v biases")
        a.__class__ = _ReferenceRopeAttention


def _set_rope_periods(model, periods: torch.Tensor | None) -> None:
    """T11: install the period buffer **as bfloat16**, not merely with bf16-rounded values.

    This is subtler than it looks and getting it wrong is silent. timm derives the working
    precision of the whole rotary embedding from this buffer::

        # timm.layers.pos_embed_sincos.RotaryEmbeddingDinoV3._get_pos_embed_from_coords
        dtype = self.periods.dtype
        coords = coords[:, :, None].to(device=device, dtype=dtype)
        angles = 2 * math.pi * coords / self.periods[None, None, :]
        sin, cos = torch.sin(angles), torch.cos(angles)

    The reference does the same, with ``dtype = bfloat16`` (``pos_embed_rope_dtype`` defaults to
    ``"bf16"``), so its coordinates, its 2*pi, its angles and its sin/cos are **all** bf16 — 2*pi
    rounds to 6.28125, and the accumulated rounding moves ``sin`` by up to 1.5e-2.

    Copying the checkpoint's bf16 values into a float32 buffer preserves the *values* and loses the
    *dtype*, so timm then computes in float32 and the embeddings differ by ~1e-2. That propagates
    through every block and shifts the output probability by ~1e-1 — a real, invisible divergence
    from the published numbers.

    With the buffer genuinely bf16, our sin/cos match the reference **exactly** (0.0e+00).
    """
    p = model.rope.periods
    src = p if periods is None else periods.to(device=p.device)
    model.rope.periods = src.to(device=p.device, dtype=torch.bfloat16)
    model.rope.pos_embed_cached = None
    # A bf16 rope buffer only reproduces the reference if q/k are cast down to it as well.
    _install_reference_attention(model)


def _assert_prefix_tokens(model, spec: EncoderSpec) -> None:
    n = int(getattr(model, "num_prefix_tokens", 1))
    if n != spec.n_prefix_tokens:
        raise RuntimeError(
            f"prefix-token mismatch: timm reports {n}, spec expects {spec.n_prefix_tokens}. "
            "Feature/LoRA token splitting would be silently wrong."
        )


def load_reference_checkpoint(path, spec: EncoderSpec) -> dict[str, torch.Tensor]:
    """Read an original ``m1_teacher_*.pth`` and return timm-named tensors.

    Packaging-time helper. The runtime path loads safetensors that were produced by this once.
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck.get(spec.checkpoint_key, ck) if spec.checkpoint_key else ck
    prefix = spec.strip_prefix or ""
    src = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    if not src:
        raise ValueError(f"no {prefix!r} keys in {path}")
    return remap_reference_state_dict(src)
