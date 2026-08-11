"""Building the foundation encoder a released pack sits on, in three tiers.

The eight packs are a neck + decoder + a handful of adapter tensors bolted onto
one of two frozen ViTs. The head is ours and always loads (see
:mod:`._fig3.load_head`); the *encoder* is where the packaging problem lives, so
it gets its own module and a strict preference order:

===== ================= ==============================================
tier   source            when it applies
===== ================= ==============================================
 (a)   exported artifact a TorchScript encoder sits beside the weights
 (b)   ``timm``          both families, always
 (c)   ``dinov3``        neither, unless timm cannot; development only
===== ================= ==============================================

**(a) is the shipping path.** :mod:`quantem.inference.export` traces a built
encoder -- adapters and fine-tuned blocks already applied -- into a
self-contained ``encoder_ts.pt`` next to the pack's ``head.pt``. Once that file
exists the app needs neither ``timm`` nor ``dinov3`` to run that pack, which is
the whole point: no research-tree architecture code, no third-party licence
surface, and the artifact's digest covers something that cannot silently drift.

**(b) covers all eight packs with no Meta code at all.** The four OmniEM packs
declare ``loader: timm_external`` and ``timm_model:
vit_large_patch14_dinov2.lvd142m``; the released weights are that architecture
with a ``vit.`` prefix. The four QuantEM packs are a DINOv3 ViT-B, which timm
also implements (``timm/models/eva.py``, Apache-2.0) -- an index that says
``framework: timm_vit`` reaches it directly, and one that says ``framework:
dinov3`` reaches it through :func:`build_quantem_timm_encoder_from_dinov3`,
which renames the tensors. ``timm`` is already a dependency.

**(c) is for development only, and is now genuinely last.** QuantEM does **not**
redistribute Meta's DINOv3 package -- it is not vendored here and is not a
dependency. This tier is reached only when timm cannot build a pack at all. It
matters because it is the reference implementation the timm path is verified
against; a developer who has the package can also build a pack and *export* it
to tier (a). If it is missing, :class:`EncoderUnavailable` says so and names the
export route rather than just failing.

**Why (b) is preferred over (a) on an accelerator, and where the order is
decided.** A traced graph is not portable between devices for this family (see
:func:`exported_encoder_name`), so on CUDA the shipped artifact fails and the
eager rebuild is what runs. :func:`quantem.inference.engine._build_and_prepare`
owns that ladder; this module only has to be able to build eagerly wherever it
is asked to.

Input normalisation
-------------------
The two families hand the network *differently scaled tensors*, and the
difference is invisible until the ER packs are wrong. See
:class:`EncoderContract`.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn

# ``DINOV3_PATH_ENV_VAR`` is re-exported: it has always been named here, and it
# now lives in a standard-library-only module so that the Models screen's
# runnability probe can agree with this one about what the variable means
# without importing torch. See quantem.inference.dinov3_hint.
from .dinov3_hint import DINOV3_PATH_ENV_VAR as DINOV3_PATH_ENV_VAR
from .dinov3_hint import dinov3_hint, hint_provides_dinov3

logger = logging.getLogger(__name__)

#: Filename an exported encoder takes inside a pack directory (tier a).
#: **This name means "traced on the CPU"** -- it is what every release bundle
#: ships and what every CPU install builds. See :func:`exported_encoder_name`.
EXPORTED_ENCODER_NAME = "encoder_ts.pt"

#: Name of the metadata blob embedded inside that TorchScript archive.
EXPORT_META_FILE = "quantem_encoder.json"

#: Metadata key recording the device an artifact was traced on. Absent in every
#: artifact written before device tagging existed, and ``"cpu"`` is the right
#: reading of that absence: the exporter's default was CPU and the release
#: builder never overrode it.
EXPORT_DEVICE_META_KEY = "traced_device"


class EncoderUnavailable(RuntimeError):
    """No tier could build this pack's encoder. Carries a user-facing explanation."""


# --- The input contract -----------------------------------------------------


@dataclass(frozen=True)
class EncoderContract:
    """How the tensor handed to ``SegModel.forward`` must be scaled.

    This is *not* the same as the encoder's EM corpus statistics
    (:data:`quantem.registry.manifest.ENCODER_NORM`), and conflating the two is
    the easiest way to get a wrong answer that still looks like a segmentation:

    * **QuantEM / dinov3** -- the training pipeline standardised the tile with
      the corpus statistics and fed the result straight to a 1-channel ViT. So
      ``input_mean = 0.583175``, ``input_std = 0.244468``, and the two notions
      coincide.
    * **OmniEM / timm** -- the training pipeline handed the encoder a *raw*
      ``[0, 1]`` tile (``image_mean=0``, ``image_std=1``) and the corpus
      statistics were applied **inside** the encoder, after replicating the
      single channel to three. So ``input_mean = 0.0``, ``input_std = 1.0``
      here, and ``0.595446 / 0.211906`` live in ``_ext_norm_*`` below.

    Normalising the OmniEM input up front would double-normalise the encoder and
    also feed the wrong distribution to the ``resnet34_detail`` neck's raw-image
    branch, which reads this same tensor. ``omniem:er`` uses that neck.
    :func:`quantem.inference.engine.load_model` asserts the built encoder's
    contract against :class:`~quantem.inference.specs.ModelSpec` so a future
    mismatch fails loudly.
    """

    input_mean: float
    input_std: float
    patch_size: int
    embedding_dim: int
    depth: int
    framework: str
    #: Which tier produced the encoder: ``"exported"``, ``"timm"`` or ``"dinov3"``.
    tier: str
    #: Set only for tier (a): which pack the artifact was exported for. The
    #: caller checks it, because an artifact from a sibling pack has the right
    #: shape and the wrong adapters, and would run without complaint.
    pack_id: str = ""


# --- The checkpoint index ---------------------------------------------------


@dataclass(frozen=True)
class EncoderManifest:
    """The ``encoder`` block of a family's ``checkpoint_index.json``.

    Read as published. The architecture facts are not re-derived here because
    the index is the artifact the research tree writes and the one whose digest
    the registry records; hard-coding them in the app would create a second
    source of truth that can drift.
    """

    arch: str
    depth: int
    embedding_dim: int
    patch_size: int
    framework: str
    image_mean: float
    image_std: float
    input_channels: int
    entry_point: dict[str, Any]
    run_id: str = ""

    @classmethod
    def from_index(cls, path: str | Path) -> EncoderManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        enc = raw["encoder"]

        def _scalar(v: Any) -> float:
            return float(v[0] if isinstance(v, (list, tuple)) else v)

        return cls(
            arch=str(enc["arch"]),
            depth=int(enc["depth"]),
            embedding_dim=int(enc["embedding_dim"]),
            patch_size=int(enc["patch_size"]),
            framework=str(enc["framework"]),
            image_mean=_scalar(enc["image_mean"]),
            image_std=_scalar(enc["image_std"]),
            input_channels=int(enc.get("input_channels", 1)),
            entry_point=dict(enc.get("feature_entry_point") or {}),
            run_id=str(enc.get("run_id", "")),
        )

    def checkpoint_paths(self, index_path: str | Path, step: int | None = None) -> list[str]:
        """Recorded checkpoint paths, newest first, optionally filtered to ``step``.

        The recorded paths are the research machine's (UNC shares, WSL mounts)
        and will not exist on a user's machine; installation resolves them by
        basename. This is only used to pick *which* file a pack wants.
        """
        raw = json.loads(Path(index_path).read_text(encoding="utf-8"))
        recs = sorted(raw.get("checkpoints") or [], key=lambda r: r.get("step") or 0, reverse=True)
        if step is not None:
            recs = [r for r in recs if int(r.get("step") or 0) == int(step)]
        return [str(r["path"]) for r in recs]


# --- Eager encoders ---------------------------------------------------------


class FrozenEncoder(nn.Module):
    """A frozen ViT exposing a uniform ``features(x, layers)``.

    Ported from ``fig3/harness/encoders.py`` (research tree, read-only), reduced
    to the two frameworks the released packs use. Feature extraction is
    symmetric across them: both return the patch-token grid of the selected
    blocks as ``[B, C, H/p, W/p]``, in ascending block order, with the encoder's
    final LayerNorm applied per tap.

    Ascending order is not cosmetic: the neck concatenates the taps on the
    channel axis into a trained 1x1 convolution, so any other order permutes its
    input.
    """

    def __init__(
        self,
        backbone: nn.Module,
        framework: str,
        depth: int,
        embedding_dim: int,
        patch_size: int,
        input_mean: float,
        input_std: float,
        apply_encoder_norm: bool = True,
    ) -> None:
        super().__init__()
        # Annotated Any because it is one of two unrelated third-party ViTs; the
        # value is still an nn.Module, so torch registers it as a submodule and
        # `.to(device)` reaches it.
        self.backbone: Any = backbone
        self.framework = framework
        self.depth = int(depth)
        self.embedding_dim = int(embedding_dim)
        self.patch_size = int(patch_size)
        self.input_mean = float(input_mean)
        self.input_std = float(input_std)
        self.apply_encoder_norm = bool(apply_encoder_norm)
        # Set for timm encoders: their native input contract (channel replication
        # + per-channel normalisation) is applied inside features(), not by the
        # caller. None for the 1-channel dinov3 encoder.
        self.ext_in_chans: int | None = None
        self.ext_norm_mean: float | None = None
        self.ext_norm_std: float | None = None
        # Parameter names (relative to this module) the weight source left
        # uninitialised on purpose, for the pack's head to overlay -- the
        # QuantEM trunk ships without the fine-tuned blocks 8-11. The head
        # loader refuses to return a model while any of these is uncovered.
        self.pending_overlay: list[str] = []
        # Set when the head's ``encoder_trainable`` block is written in a
        # different dialect from this module's parameter names, which happens
        # for a QuantEM pack whose index is DINOv3's and whose encoder was
        # nonetheless built through timm. The head loader puts the block through
        # this before placing it; without it the fine-tuned blocks simply never
        # land and the pack runs at trunk weights, silently. None means the
        # names already match.
        self.overlay_remap: Callable[[dict], dict] | None = None
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def features(self, x: torch.Tensor, layers: list[int]) -> list[torch.Tensor]:
        layers = sorted(int(i) for i in layers)
        if self.framework == "dinov3":
            feats = self.backbone.get_intermediate_layers(
                x,
                n=layers,
                reshape=True,
                norm=self.apply_encoder_norm,
                return_class_token=False,
            )
            return list(feats)
        if self.framework == "timm_vit":
            x = self._preprocess(x)
            out = self.backbone.forward_intermediates(
                x,
                indices=layers,
                norm=self.apply_encoder_norm,
                output_fmt="NCHW",
                intermediates_only=True,
            )
            return list(out)
        raise ValueError(f"Unknown encoder framework {self.framework!r}")

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """timm path: replicate the grey channel, then normalise per channel."""
        in_chans = int(self.ext_in_chans or 1)
        if x.shape[1] == 1 and in_chans > 1:
            x = x.repeat(1, in_chans, 1, 1)
        mean = float(self.ext_norm_mean or 0.0)
        std = float(self.ext_norm_std or 1.0)
        return (x - mean) / std


class ExportedEncoder(nn.Module):
    """Tier (a): a TorchScript encoder with its tap layers already baked in.

    The traced graph *is* the tap selection, so ``features()`` refuses a
    different ``layers`` list rather than silently returning the exported ones.

    It is also the pack's encoder, not the family's base encoder: the export ran
    after the head's LoRA adapters or replaced blocks were applied (see
    :mod:`quantem.inference.export` for why it must). Hence
    :attr:`embeds_pack_state` -- the head loader must *not* try to load those
    tensors a second time, and could not if it wanted to, because freezing
    dissolved their parameter names into the graph. :attr:`pack_id` is what lets
    the caller check it is the right pack's artifact.
    """

    #: The encoder-side tensors from ``head.pt`` are already inside the graph.
    embeds_pack_state = True

    def __init__(self, module: torch.jit.ScriptModule, meta: dict[str, Any]) -> None:
        super().__init__()
        self.module: Any = module
        self.meta = dict(meta)
        self.framework = "exported"
        self.pack_id = str(meta.get("pack_id", ""))
        self.adapt = str(meta.get("adapt", ""))
        self.depth = int(meta["depth"])
        self.embedding_dim = int(meta["embedding_dim"])
        self.patch_size = int(meta["patch_size"])
        self.input_mean = float(meta["input_mean"])
        self.input_std = float(meta["input_std"])
        self.layers = [int(i) for i in meta["layers"]]
        self.traced_tile = int(meta["traced_tile"])
        self.dynamic_spatial = bool(meta.get("dynamic_spatial", False))
        #: The device this graph was traced on. Absent means CPU -- see
        #: :data:`EXPORT_DEVICE_META_KEY`.
        self.traced_device = str(meta.get(EXPORT_DEVICE_META_KEY) or "cpu")

    def features(self, x: torch.Tensor, layers: list[int]) -> list[torch.Tensor]:
        want = sorted(int(i) for i in layers)
        if want != sorted(self.layers):
            raise EncoderUnavailable(
                f"exported encoder was traced with taps {sorted(self.layers)} but this "
                f"pack asks for {want}. Re-export it with the pack's own config."
            )
        size = int(x.shape[-1])
        if size != self.traced_tile and not self.dynamic_spatial:
            raise EncoderUnavailable(
                f"exported encoder was traced at tile {self.traced_tile} and its graph did "
                f"not verify at other sizes, but a {size}px window was requested. "
                "Re-export at this tile, or run the eager encoder."
            )
        return list(self.module(x))


# --- Tier (a): the exported artifact ----------------------------------------


def _device_kind(device: str | None) -> str:
    """``"cuda:1"`` -> ``"cuda"``. Duplicated from :mod:`.device` on purpose:
    that module imports torch lazily and this one imports it at module scope,
    so the dependency would run the wrong way."""
    return (str(device or "cpu").strip().lower().split(":", 1)[0]) or "cpu"


def exported_encoder_name(device: str = "cpu") -> str:
    """The artifact filename for a trace made on ``device``.

    A TorchScript artifact is **not** device-independent, which the exporter
    believed for a year and four of the eight shipped packs disproved. The
    DINOv3 rotary position encoding materialises tensors while it is traced and
    ``torch.jit.freeze`` folds them into the archive's constant table, where
    ``torch.jit.load(map_location=...)`` cannot reach them: a CPU trace of the
    QuantEM family dies on CUDA with *"Expected all tensors to be on the same
    device"*, and a CUDA trace of the same encoder dies the mirror death on the
    CPU. One name for both was therefore a name for a file that can only be
    right for whoever wrote it last -- and the self-healing re-export wrote it
    on whatever device that run happened to use, so one GPU run permanently
    broke the CPU path of a shared pack directory.

    Tagging the name removes the class rather than guarding it: a CUDA trace has
    nowhere to be written except a CUDA name. ``cpu`` keeps the bare
    :data:`EXPORTED_ENCODER_NAME` so every bundle, installer and checksum that
    already names it stays correct.
    """
    kind = _device_kind(device)
    if kind == "cpu":
        return EXPORTED_ENCODER_NAME
    stem, _, suffix = EXPORTED_ENCODER_NAME.rpartition(".")
    return f"{stem}.{kind}.{suffix}"


def exported_encoder_path(pack_root: str | Path, device: str = "cpu") -> Path:
    return Path(pack_root) / exported_encoder_name(device)


def load_exported_encoder(path: str | Path, device: str = "cpu") -> ExportedEncoder:
    """Load a TorchScript encoder and the metadata embedded in its archive.

    Refuses an artifact that **states** it was traced somewhere else. Only a
    stated one: whether a given trace survives a device change is a property of
    the encoder, not a rule -- MEASURED, the OmniEM ViT-L's CPU trace runs
    perfectly on CUDA, while the QuantEM ViT-B's does not, because only the
    latter's rotary position encoding recomputes itself from CPU-created
    coordinates inside ``forward``. So an artifact with no stamp (every release
    bundle to date) is *tried*, and
    :func:`quantem.inference.engine.prepare_for_device` decides by running it.
    A stamp, once present, is believed: it is written only by an exporter that
    knew the answer, and the alternative to refusing here is a raw
    device-mismatch ``RuntimeError`` from inside a traced graph.
    """
    extra: dict[str, bytes] = {EXPORT_META_FILE: b""}
    module = torch.jit.load(str(path), map_location=device, _extra_files=extra)
    blob = extra.get(EXPORT_META_FILE) or b""
    if not blob:
        raise EncoderUnavailable(
            f"{Path(path).name} is a TorchScript module but carries no {EXPORT_META_FILE}; "
            "it was not produced by quantem.inference.export and its tap layers, patch size "
            "and input normalisation are unknown."
        )
    meta = json.loads(blob.decode("utf-8"))
    stamped = meta.get(EXPORT_DEVICE_META_KEY)
    wanted = _device_kind(device)
    if stamped and _device_kind(stamped) != wanted:
        raise EncoderUnavailable(
            f"{Path(path).name} was traced on {_device_kind(stamped)!r} and is not the "
            f"artifact for {wanted!r}: a traced graph can carry device-locked constants "
            "that loading it elsewhere does not move. The encoder will be rebuilt instead."
        )
    module.eval()
    return ExportedEncoder(module, meta)


# --- Tier (b): timm ---------------------------------------------------------


def _load_state_dict_any(path: str | Path) -> dict:
    p = str(path)
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(p)
    ck = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(ck, dict):
        for k in ("model", "state_dict", "teacher"):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
    return ck


def build_timm_encoder(
    weight_path: str | Path,
    manifest: EncoderManifest,
    apply_encoder_norm: bool = True,
) -> FrozenEncoder:
    """Tier (b): build the encoder through ``timm`` from its own manifest.

    Everything needed is declared in ``checkpoint_index.json``'s
    ``feature_entry_point`` -- the timm architecture name, the key prefix to
    strip, the build image size, the input channel count. Nothing about the
    OmniEM family needs code from Meta or from the research tree.
    """
    import timm

    fe = manifest.entry_point
    model_name = fe.get("timm_model")
    if not model_name:
        raise EncoderUnavailable(
            f"encoder {manifest.run_id!r} declares framework 'timm_vit' but no "
            "'timm_model' in its feature_entry_point; cannot build it."
        )
    in_chans = int(fe.get("in_chans", manifest.input_channels or 3))
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        in_chans=in_chans,
        img_size=int(fe.get("img_size_build", 518)),
        dynamic_img_size=bool(fe.get("dynamic_img_size", True)),
    )

    sd = _load_state_dict_any(weight_path)
    strip = fe.get("strip_prefix")
    if strip:
        stripped = {k[len(strip) :]: v for k, v in sd.items() if k.startswith(strip)}
        sd = stripped or sd
    drop = tuple(fe.get("drop_key_prefixes", ("head.",)))
    sd = {k: v for k, v in sd.items() if not any(k.startswith(p) for p in drop)}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    allow = set(fe.get("allow_unexpected", ("mask_token",)))
    missing = [k for k in missing if not any(k.startswith(p) for p in drop)]
    unexpected = [k for k in unexpected if k not in allow]
    if missing or unexpected:
        # A partially initialised encoder still produces a probability map. It is
        # just not the published model, and nothing downstream would notice.
        raise EncoderUnavailable(
            f"unclean load of {Path(weight_path).name} into timm {model_name!r}: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}. Refusing a partially "
            "initialised encoder."
        )

    enc = FrozenEncoder(
        backbone=model,
        framework="timm_vit",
        depth=manifest.depth,
        embedding_dim=manifest.embedding_dim,
        patch_size=manifest.patch_size,
        # The dataset handed this encoder a raw [0, 1] tile; see EncoderContract.
        input_mean=0.0,
        input_std=1.0,
        apply_encoder_norm=apply_encoder_norm,
    )
    enc.ext_in_chans = in_chans
    enc.ext_norm_mean = manifest.image_mean
    enc.ext_norm_std = manifest.image_std
    return enc


# --- Tier (b), QuantEM variant: the DINOv3 ViT-B through timm ---------------
#
# The QuantEM Hugging Face release publishes the family's encoder as
# timm-named safetensors ("vit_base_patch16_dinov3_qkvb", timm >= 1.0.20),
# which is what lets an HF-installed pack build its encoder with **no Meta
# code at all** -- timm's implementation lives in timm/models/eva.py and is
# Apache-2.0. Ported from quantem-core's models/encoders/quantem_vit.py, which
# established (and pinned in its own tests) the four adaptations timm needs to
# reproduce the reference DINOv3 forward exactly:
#
# T10  the attention k-bias. timm registers ``k_bias`` as a non-persistent
#      zeroed buffer because Meta's distilled checkpoints trained with
#      ``mask_k_bias=True``. Ours did not: the k-bias was live for all 675k
#      steps (measured max |k| = 7.0 at block 5). It is promoted to a
#      persistent buffer so the trained values load and round-trip.
# T11  the rotary period buffer must be genuinely **bfloat16**: timm derives
#      the working precision of the whole rotary embedding from its dtype, and
#      the reference computed sin/cos in bf16. Values-only copying into a
#      float32 buffer shifts the output probability by ~1e-1.
# eps  the LayerNorm epsilon is 1e-6 (DINOv3's `layernorm` default); timm's
#      dinov3 entries hard-code 1e-5, which perturbs all 25 LayerNorms.
# rope the reference quantises the *whole* q/k tensors (prefix tokens
#      included) to bf16 around rotation; timm rotates only the patch slice at
#      float32. The attention override below reproduces the reference order.

#: ``feature_entry_point.variant`` value that selects this builder. The HF
#: installer (quantem.registry.hf_install) writes it into the synthesised
#: checkpoint_index.json.
QUANTEM_TIMM_VARIANT = "quantem_dinov3"


def _promote_k_bias(model: nn.Module) -> None:
    """T10: make ``k_bias`` a persistent buffer so trained values load into it."""
    for blk in model.blocks:
        attn = blk.attn
        attn.register_buffer("k_bias", torch.zeros_like(attn.q_bias), persistent=True)


def _install_reference_attention(model: nn.Module) -> None:
    """Give every block the reference's rotary-embedding dtype semantics.

    The reference (dinov3 ``SelfAttention.apply_rope``) casts the whole q and k
    tensors to the rope dtype (bfloat16), rotates the patch slice, concatenates
    the -- also quantised -- prefix tokens back, and casts to the input dtype.
    timm rotates only the patch slice and leaves the prefix at float32. Those
    five prefix tokens are attended to by every patch token, so the discrepancy
    spreads across the whole map (measured up to 1e-1 in output probability,
    which flips pixels near the 0.5 threshold). Only the rotary section is
    reimplemented; projections and the attention call are timm's.
    """
    import torch.nn.functional as F
    from timm.layers import apply_rot_embed_cat
    from timm.models.eva import EvaAttention

    class _ReferenceRopeAttention(EvaAttention):
        def forward(self, x, rope=None, attn_mask=None, is_causal=False):  # type: ignore[override]
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
        # These are Identity for the released configuration; the override skips
        # them, so refuse rather than silently dropping a real module if timm's
        # defaults ever change.
        for attr in ("q_norm", "k_norm", "norm"):
            m = getattr(a, attr, None)
            if m is not None and not isinstance(m, nn.Identity):
                raise EncoderUnavailable(
                    f"unexpected active {attr} in EvaAttention; refusing to override"
                )
        if getattr(a, "gate", None) is not None:
            raise EncoderUnavailable("unexpected gated EvaAttention; refusing to override")
        if a.qkv is None or a.q_bias is None:
            raise EncoderUnavailable("expected a fused qkv with separate q/k/v biases")
        a.__class__ = _ReferenceRopeAttention


def _set_rope_periods_bf16(model: nn.Module, periods: torch.Tensor | None) -> None:
    """T11: install the rotary period buffer as true bfloat16, then fix attention.

    timm computes the whole rotary embedding in ``self.periods.dtype``. The
    checkpoint's values are already bf16; when absent, timm's own fp32 periods
    truncated through bf16 reproduce the reference exactly (measured 0.0).
    """
    p = model.rope.periods
    src = p if periods is None else periods.to(device=p.device)
    model.rope.periods = src.to(device=p.device, dtype=torch.bfloat16)
    model.rope.pos_embed_cached = None
    # A bf16 rope buffer only reproduces the reference if q/k are cast down to
    # it as well.
    _install_reference_attention(model)


def build_quantem_timm_encoder(
    weight_path: str | Path | None,
    manifest: EncoderManifest,
    apply_encoder_norm: bool = True,
    *,
    skeleton_state: dict | None = None,
    state_dict: dict | None = None,
) -> FrozenEncoder:
    """Tier (b) for the QuantEM family: the DINOv3 ViT-B through timm.

    Selected by ``feature_entry_point.variant == "quantem_dinov3"`` in a pack's
    ``checkpoint_index.json`` -- the shape an HF install writes. The weight
    file is the published trunk safetensors (timm-named, ``encoder.`` prefix);
    for the ``last_n`` packs it deliberately lacks the fine-tuned blocks named
    in ``feature_entry_point.overlay_blocks``, which the pack's own head
    carries and the head loader overlays. Every parameter left uninitialised
    here is recorded on the returned encoder as ``pending_overlay``, and
    :func:`quantem.inference._fig3.load_head.build_and_load_head` refuses to
    hand back a model while any of them was not covered by the head --
    otherwise those blocks would run at random init and the output would look
    like a segmentation.

    ``skeleton_state`` serves ``quantem:er`` (``adapt: full``): its head embeds
    the whole fine-tuned encoder, no trunk is installed, and the module is
    built from the head's own timm-named tensors.

    ``state_dict`` hands over already-loaded, already-timm-named tensors and
    wins over both of the above. It is how
    :func:`build_quantem_timm_encoder_from_dinov3` reaches this builder with a
    pack whose weights are on disk in DINOv3's naming: the translation happens
    once, there, and everything downstream is the ordinary timm path.

    Input contract: the caller hands this encoder a tile already standardised
    with the corpus statistics (``input_mean``/``input_std`` below), exactly as
    the dinov3 tier -- not the OmniEM convention of normalising inside.
    """
    import timm

    fe = manifest.entry_point
    model_name = fe.get("timm_model")
    if not model_name:
        raise EncoderUnavailable(
            f"encoder {manifest.run_id!r} declares variant {QUANTEM_TIMM_VARIANT!r} but no "
            "'timm_model' in its feature_entry_point; cannot build it."
        )

    kwargs: dict[str, Any] = {}
    norm_eps = fe.get("norm_eps")
    if norm_eps is not None:
        from functools import partial

        # timm's dinov3 entries hard-code eps=1e-5 (right for Meta's released
        # weights, wrong for ours); see the tier comment above.
        kwargs["norm_layer"] = partial(nn.LayerNorm, eps=float(norm_eps))

    model = timm.create_model(
        str(model_name),
        pretrained=False,
        num_classes=0,
        in_chans=int(fe.get("in_chans", manifest.input_channels or 1)),
        img_size=int(fe.get("img_size_build", 512)),
        **kwargs,
    )
    _promote_k_bias(model)

    if state_dict is not None:
        sd = dict(state_dict)
    elif weight_path is not None:
        sd = _load_state_dict_any(weight_path)
        strip = str(fe.get("strip_prefix") or "")
        if strip:
            stripped = {k[len(strip) :]: v for k, v in sd.items() if k.startswith(strip)}
            sd = stripped or sd
    elif skeleton_state:
        sd = {
            k[len("backbone.") :]: v for k, v in skeleton_state.items() if k.startswith("backbone.")
        }
        if not sd:
            raise EncoderUnavailable(
                "no encoder blob installed and the head carries no backbone tensors to "
                "build the encoder from."
            )
    else:
        raise EncoderUnavailable(
            f"encoder {manifest.run_id!r} needs its weight file, which is not installed."
        )

    periods = sd.pop("rope.periods", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    overlay_blocks = [int(i) for i in (fe.get("overlay_blocks") or [])]
    overlay_prefixes = tuple(f"blocks.{i}." for i in overlay_blocks)
    pending = [k for k in missing if k.startswith(overlay_prefixes)] if overlay_prefixes else []
    hard_missing = [k for k in missing if k not in pending]
    if hard_missing or unexpected:
        raise EncoderUnavailable(
            f"unclean load of {Path(weight_path).name if weight_path else 'the head backbone'} "
            f"into timm {model_name!r}: missing={hard_missing[:8]} "
            f"unexpected={list(unexpected)[:8]}. Refusing a partially initialised encoder."
        )

    if fe.get("rope_periods_bf16"):
        _set_rope_periods_bf16(model, periods)
    elif periods is not None:
        model.rope.periods = periods.to(device=model.rope.periods.device)
        model.rope.pos_embed_cached = None

    want_prefix = int(fe.get("n_prefix_tokens", 5))
    have_prefix = int(getattr(model, "num_prefix_tokens", 1))
    if have_prefix != want_prefix:
        raise EncoderUnavailable(
            f"prefix-token mismatch: timm reports {have_prefix}, the index expects "
            f"{want_prefix}. Feature and LoRA token splitting would be silently wrong."
        )
    model.eval()

    enc = FrozenEncoder(
        backbone=model,
        framework="timm_vit",
        depth=manifest.depth,
        embedding_dim=manifest.embedding_dim,
        patch_size=manifest.patch_size,
        # The caller standardises the tile with the corpus statistics, exactly
        # as for the dinov3 tier; _preprocess is then the identity (1 channel,
        # no internal normalisation).
        input_mean=manifest.image_mean,
        input_std=manifest.image_std,
        apply_encoder_norm=apply_encoder_norm,
    )
    enc.ext_in_chans = 1
    # Parameters the trunk left uninitialised, named as the head loader sees
    # them. build_and_load_head verifies the head covers every one.
    enc.pending_overlay = [f"backbone.{k}" for k in sorted(pending)]
    return enc


# --- Tier (b), reached from a DINOv3-shaped index ---------------------------
#
# An HF-installed pack arrives with everything above already arranged for it:
# the installer synthesises `framework: timm_vit` + `variant: quantem_dinov3`
# and the published weights are timm-named. A pack installed from a release
# bundle, or from a local research directory, carries the *research* index
# instead -- `framework: dinov3`, `loader: dinov3_teacher` -- and its weights
# are named as Meta's model names them. Nothing but the names differed, and the
# names were enough to send it down a tier that needs a package this project
# does not ship, and from there to the CPU. MEASURED on a Quadro RTX 8000:
# quantem:mito 0.68 s per window on the processor against 0.24 s on the card,
# for want of a rename.
#
# The translation below is a port of quantem-core's
# `models/encoders/quantem_vit.remap_reference_state_dict` -- the same function
# the published HF artifacts were converted with, and the one whose fidelity
# that project's tests pin. Keeping the two in step matters: a key this misses
# is not an error, it is a tensor that quietly never loads.

#: DINOv3 keys that exist only for pretraining and have no timm counterpart.
_DINOV3_PRETRAIN_ONLY = ("mask_token",)

#: Where :func:`remap_dinov3_state_dict` parks the rotary period buffer. timm
#: registers ``rope.periods`` non-persistently, so it cannot travel in a state
#: dict; it is applied by hand afterwards (T11).
_ROPE_PERIODS_KEY = "rope.periods"

#: ``encoder.arch`` -> the timm entry that implements it. Only the ViT-B is
#: mapped because the QuantEM family is only the ViT-B; an unknown arch falls
#: through to Meta's package rather than guessing at a timm name.
_QUANTEM_TIMM_MODELS: dict[str, str] = {
    "vit_base": "vit_base_patch16_dinov3_qkvb",
}


def remap_dinov3_state_dict(src: dict) -> dict:
    """DINOv3 backbone naming -> timm ``Eva`` naming.

    ``src`` is already stripped of the ``teacher.``/``backbone.`` prefixes.

    Deliberately **not** ``timm.models.eva.checkpoint_filter_fn``: that helper
    drops the k-bias, which is right for Meta's distilled weights (trained with
    ``mask_k_bias=True``) and wrong for ours (the k-bias was live for all 675k
    steps; see T10 in the tier comment above). Every trained tensor is kept.
    """
    out: dict = {}
    for k, v in src.items():
        if k in _DINOV3_PRETRAIN_ONLY:
            continue
        if k == "rope_embed.periods":
            out[_ROPE_PERIODS_KEY] = v
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


def remap_dinov3_overlay(state: dict) -> dict:
    """A head's ``encoder_trainable`` block, in the timm module's naming.

    The released heads were saved from the reference encoder, so a ``last_n``
    pack's replaced blocks arrive as ``backbone.blocks.8.attn.qkv.bias`` and a
    ``full`` pack's whole encoder the same way. Keys outside ``backbone.`` (the
    OmniEM LoRA modules) pass through untouched.
    """
    passthrough = {k: v for k, v in state.items() if not k.startswith("backbone.")}
    backbone = {k[len("backbone.") :]: v for k, v in state.items() if k.startswith("backbone.")}
    remapped = remap_dinov3_state_dict(backbone)
    # A buffer, never a trained head tensor; it is installed from the trunk.
    remapped.pop(_ROPE_PERIODS_KEY, None)
    out = {f"backbone.{k}": v for k, v in remapped.items()}
    out.update(passthrough)
    return out


def _timm_entry_point_for_dinov3(manifest: EncoderManifest, backbone_sd: dict) -> dict:
    """The ``feature_entry_point`` an HF install would have written for this pack.

    Every value here is the one
    :func:`quantem.registry.hf_install._checkpoint_index` writes for the QuantEM
    family, so a pack installed from a bundle and the same pack installed from
    Hugging Face build the *same* module. The one thing read from the checkpoint
    rather than fixed is the prefix-token count, because it is derivable
    (1 class token + however many storage tokens this encoder was trained with)
    and a wrong value would split features silently rather than loudly.
    """
    model_name = _QUANTEM_TIMM_MODELS.get(manifest.arch)
    if not model_name:
        raise EncoderUnavailable(
            f"encoder arch {manifest.arch!r} has no timm equivalent recorded here; "
            "only the QuantEM ViT-B does."
        )
    storage = backbone_sd.get("storage_tokens")
    n_prefix = 1 + int(storage.shape[1]) if storage is not None else 5
    return {
        "loader": "timm_dinov3_remap",
        "variant": QUANTEM_TIMM_VARIANT,
        "forward": "forward_intermediates",
        "timm_model": model_name,
        "in_chans": int(manifest.input_channels or 1),
        "img_size_build": 512,
        "norm_eps": 1e-06,
        "rope_periods_bf16": True,
        "n_prefix_tokens": n_prefix,
        # The trunk this reads is the research checkpoint, which carries all
        # twelve blocks; the head then overwrites the fine-tuned ones. Nothing
        # is left for the head to *provide*, unlike the HF trunk.
        "overlay_blocks": [],
    }


def build_quantem_timm_encoder_from_dinov3(
    weight_path: str | Path | None,
    manifest: EncoderManifest,
    apply_encoder_norm: bool = True,
    *,
    skeleton_state: dict | None = None,
) -> FrozenEncoder:
    """Build a DINOv3-indexed QuantEM pack through timm, by renaming its tensors.

    The bridge between an index written by the research tree and the builder
    above. It reads the checkpoint the index describes, translates the names,
    synthesises the timm entry point that index implies, and hands both to
    :func:`build_quantem_timm_encoder`. The encoder that comes back is the same
    object an HF install produces, with one addition: ``overlay_remap``, because
    the pack's *head* is written in the same DINOv3 naming and has to make the
    same journey.
    """
    fe = manifest.entry_point
    prefix = str(fe.get("backbone_prefix", "backbone."))
    checkpoint_key = str(fe.get("checkpoint_key", "teacher"))

    if weight_path is not None:
        src = _dinov3_backbone_state(weight_path, checkpoint_key, prefix)
    elif skeleton_state:
        src = {k[len(prefix) :]: v for k, v in skeleton_state.items() if k.startswith(prefix)}
        if not src:
            raise EncoderUnavailable(
                "no encoder blob installed and the head carries no backbone tensors to "
                "build the encoder from."
            )
    else:
        raise EncoderUnavailable(
            f"encoder {manifest.run_id!r} needs its weight file, which is not installed."
        )

    state = remap_dinov3_state_dict(src)
    shim = replace(
        manifest,
        framework="timm_vit",
        entry_point=_timm_entry_point_for_dinov3(manifest, src),
    )
    enc = build_quantem_timm_encoder(None, shim, apply_encoder_norm, state_dict=state)
    enc.overlay_remap = remap_dinov3_overlay
    return enc


# --- Tier (c): dinov3 -------------------------------------------------------


def dinov3_available() -> bool:
    """True when Meta's DINOv3 package can be imported (after the env hint)."""
    try:
        _import_dinov3("dinov3.models.vision_transformer")
    except EncoderUnavailable:
        return False
    return True


def _import_dinov3(module_name: str) -> ModuleType:
    """Import from Meta's DINOv3 package, honouring the dev-only path hint.

    What counts as a usable hint is decided in one place
    (:func:`quantem.inference.dinov3_hint.hint_provides_dinov3`) so that this
    and the Models screen's probe cannot mean different things by the same
    environment variable.
    """
    hint = dinov3_hint()
    if hint and hint not in sys.path and hint_provides_dinov3(hint):
        sys.path.insert(0, hint)
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise EncoderUnavailable(
            "The QuantEM family's encoder is a DINOv3 ViT-B and needs Meta's `dinov3` "
            f"package ({module_name}), which QuantEM does not redistribute and does not "
            "depend on.\n"
            "The supported path is not to install it here but to export the pack once, on "
            "a machine that has it, and ship the artifact:\n"
            "  1. point QUANTEM_DINOV3_PATH at a checkout of "
            "https://github.com/facebookresearch/dinov3\n"
            "  2. python -m quantem.inference.export <pack-id>\n"
            f"     -> writes {EXPORTED_ENCODER_NAME} beside the pack's head.pt\n"
            "After that the pack loads through tier (a) and `dinov3` is never needed again. "
            "The four OmniEM packs need none of this: they build through timm.\n"
            f"Original error: {exc!r}"
        ) from exc


def infer_dinov3_build_kwargs(backbone_sd: dict, base_kwargs: dict | None = None) -> dict:
    """Reconstruct the ViT block config from the checkpoint's own keys.

    Ported from ``em_ssl/utils/checkpoint_index.py`` (research tree, read-only).
    The bare ``vit_base`` factory's defaults differ from the SSL training
    config -- LayerScale is **off** by default but was on in training -- so a
    naive build has nowhere to put the checkpoint's ``ls1``/``ls2`` gammas and
    silently produces corrupted features. Deriving the kwargs from the
    checkpoint means any future run rebuilds correctly without a config file.
    """
    kw = dict(base_kwargs or {})
    keys = list(backbone_sd.keys())
    if "storage_tokens" in backbone_sd:
        kw["n_storage_tokens"] = int(backbone_sd["storage_tokens"].shape[1])
    if any(".ls1." in k or ".ls2." in k for k in keys):
        kw.setdefault("layerscale_init", 1.0e-05)  # overwritten by the loaded gamma
    if any(k.startswith("cls_norm.") for k in keys):
        kw["untie_cls_and_patch_norms"] = True
    return kw


def _dinov3_backbone_state(weight_path: str | Path, checkpoint_key: str, prefix: str) -> dict:
    ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    sd = ckpt.get(checkpoint_key, ckpt) if isinstance(ckpt, dict) else ckpt
    backbone_sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    if not backbone_sd:
        raise EncoderUnavailable(f"No {prefix!r} keys in {Path(weight_path).name}")
    return backbone_sd


def build_dinov3_encoder(
    weight_path: str | Path | None,
    manifest: EncoderManifest,
    apply_encoder_norm: bool = True,
    *,
    skeleton_state: dict | None = None,
) -> FrozenEncoder:
    """Tier (c): build the QuantEM ViT-B through Meta's DINOv3 package.

    Development only -- see the module docstring. ``skeleton_state`` lets a pack
    whose head embeds a fully fine-tuned encoder (``quantem:er``, ``adapt:
    full``) build the right module shape from the head's own parameter names
    when the shared encoder blob is not installed; the weights are overwritten
    by the head either way.
    """
    fe = manifest.entry_point
    build = dict(fe.get("build") or {})
    module_name = str(build.get("module", "dinov3.models.vision_transformer"))
    factory_name = str(build.get("factory", manifest.arch))
    prefix = str(fe.get("backbone_prefix", "backbone."))
    checkpoint_key = str(fe.get("checkpoint_key", "teacher"))

    module = _import_dinov3(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise EncoderUnavailable(
            f"{module_name} has no factory {factory_name!r} (dinov3 version mismatch)."
        )

    if weight_path is not None:
        backbone_sd = _dinov3_backbone_state(weight_path, checkpoint_key, prefix)
    elif skeleton_state:
        backbone_sd = {
            k[len(prefix) :]: v for k, v in skeleton_state.items() if k.startswith(prefix)
        }
        if not backbone_sd:
            raise EncoderUnavailable(
                "no encoder blob installed and the head carries no backbone tensors to "
                "infer the architecture from."
            )
    else:
        raise EncoderUnavailable("no encoder weights and no skeleton state to build from.")

    kwargs = dict(build.get("kwargs") or {})
    kwargs.setdefault("patch_size", manifest.patch_size)
    kwargs.setdefault("in_chans", manifest.input_channels or 1)
    kwargs = infer_dinov3_build_kwargs(backbone_sd, kwargs)
    model = factory(**kwargs)

    if weight_path is not None:
        missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
        if missing or unexpected:
            raise EncoderUnavailable(
                f"unclean load of {Path(weight_path).name} into {factory_name}: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}. The rebuild "
                "config does not match this checkpoint; refusing a corrupted encoder."
            )
    model.eval()

    return FrozenEncoder(
        backbone=model,
        framework="dinov3",
        depth=manifest.depth,
        embedding_dim=manifest.embedding_dim,
        patch_size=manifest.patch_size,
        # This encoder was trained on tiles standardised by the corpus stats.
        input_mean=manifest.image_mean,
        input_std=manifest.image_std,
        apply_encoder_norm=apply_encoder_norm,
    )


# --- The tiered entry point -------------------------------------------------


@dataclass(frozen=True)
class EncoderBundle:
    module: nn.Module
    contract: EncoderContract


def build_encoder(
    *,
    manifest: EncoderManifest | None,
    encoder_path: str | Path | None,
    export_path: str | Path | None = None,
    apply_encoder_norm: bool = True,
    device: str = "cpu",
    skeleton_state: dict | None = None,
    allow_eager: bool = True,
) -> EncoderBundle:
    """Build a pack's encoder, preferring the exported artifact.

    Args:
        manifest: the family's parsed ``checkpoint_index.json``. May be None
            only when ``export_path`` exists, since the artifact is self-describing.
        encoder_path: the shared encoder blob, or None for a pack whose head
            embeds a fully fine-tuned encoder.
        export_path: candidate tier-(a) artifact; used when it exists.
        skeleton_state: head parameters to infer the module shape from when
            there is no encoder blob.
        allow_eager: set False to require tier (a) -- used by tests that assert
            the app can run with no research dependency installed.

    Raises:
        EncoderUnavailable: with an explanation naming the export route.
    """
    if export_path is not None and Path(export_path).exists():
        bundle = load_exported_encoder(export_path, device=device)
        return EncoderBundle(
            module=bundle,
            contract=EncoderContract(
                input_mean=bundle.input_mean,
                input_std=bundle.input_std,
                patch_size=bundle.patch_size,
                embedding_dim=bundle.embedding_dim,
                depth=bundle.depth,
                framework="exported",
                tier="exported",
                pack_id=bundle.pack_id,
            ),
        )

    if not allow_eager:
        raise EncoderUnavailable(
            f"no exported encoder at {export_path} and eager construction was disallowed. "
            f"Run: python -m quantem.inference.export <pack-id>"
        )
    if manifest is None:
        raise EncoderUnavailable(
            "no exported encoder and no checkpoint_index.json for this pack; nothing "
            "describes the architecture to rebuild."
        )

    if manifest.framework == "timm_vit":
        if manifest.entry_point.get("variant") == QUANTEM_TIMM_VARIANT:
            # The QuantEM ViT-B published as timm-named safetensors (an HF
            # install). Handles its own no-blob case: quantem:er builds from
            # the head's embedded encoder via skeleton_state.
            enc = build_quantem_timm_encoder(
                encoder_path, manifest, apply_encoder_norm, skeleton_state=skeleton_state
            )
        else:
            if encoder_path is None:
                raise EncoderUnavailable(
                    f"encoder {manifest.run_id!r} needs its weight file, which is not installed."
                )
            enc = build_timm_encoder(encoder_path, manifest, apply_encoder_norm)
        tier = "timm"
    elif manifest.framework == "dinov3":
        # timm first, Meta's package only if timm cannot. The index says
        # "dinov3" because the research tree wrote it, not because that is the
        # only way to build this encoder: timm implements the same architecture
        # (Apache-2.0, timm/models/eva.py) and only the tensor *names* differ,
        # which build_quantem_timm_encoder_from_dinov3 translates.
        #
        # The order is what closes the GPU gap, and it is chosen on evidence
        # rather than preference. A traced DINOv3 encoder cannot be replayed on
        # another device (see exported_encoder_name), so on an accelerator the
        # shipped artifact fails and the eager rebuild is the only route left --
        # and if that route needs a package this project does not redistribute,
        # the pack lands on the processor. MEASURED on a Quadro RTX 8000:
        # quantem:mito 0.68 s per window there against 0.24 s on the card. timm
        # is already a dependency, so the fast route is the one that is always
        # available and the reference package is the fallback.
        #
        # It is not a different model. quantem-core pins the five adaptations
        # timm needs to reproduce the reference forward, and this build is
        # verified against the DINOv3 one on the released weights (see
        # inference/tests/test_dinov3_timm_bridge.py).
        #
        # This belongs here and not at install time: rewriting a pack's
        # checkpoint_index.json on the way in would make the file disagree with
        # the digest the installer recorded for it, so `quantem models verify`
        # would then call every QuantEM pack corrupt.
        try:
            enc = build_quantem_timm_encoder_from_dinov3(
                encoder_path, manifest, apply_encoder_norm, skeleton_state=skeleton_state
            )
            tier = "timm"
        except EncoderUnavailable as timm_exc:
            logger.info(
                "%s: could not build this encoder through timm (%s); falling back "
                "to Meta's dinov3 package.",
                manifest.run_id or manifest.arch,
                timm_exc,
            )
            enc = build_dinov3_encoder(
                encoder_path, manifest, apply_encoder_norm, skeleton_state=skeleton_state
            )
            tier = "dinov3"
    else:
        raise EncoderUnavailable(
            f"encoder framework {manifest.framework!r} is not supported "
            "(QuantEM builds 'timm_vit' and 'dinov3')."
        )

    return EncoderBundle(
        module=enc,
        contract=EncoderContract(
            input_mean=enc.input_mean,
            input_std=enc.input_std,
            patch_size=enc.patch_size,
            embedding_dim=enc.embedding_dim,
            depth=enc.depth,
            # What was *built*, not what the index said. They differ for a
            # DINOv3-indexed pack built through timm, and this field is read as
            # provenance -- the honest answer is the module in hand.
            framework=enc.framework,
            tier=tier,
        ),
    )
