"""Runtime monkeypatches that adapt upstream DINOv3 to single-channel EM SSL.

Call ``apply_em_patches()`` once before launching DINOv3 training. The four that change the recipe:

  1. build_model: inject ``in_chans`` (DINOv3 drops cfg.student.in_chans in the SSL path).
  2. SSLMetaArch.build_data_augmentation_dino: return the EM single-channel multi-crop aug.
  3. make_dataset: dispatch ``EMShards:...`` to the streaming EM shard dataset (WebDataset-layout
     tars, read with stdlib ``tarfile``).
  4. do_test: export the EMA teacher checkpoint + update checkpoint_index, and skip DINOv3's own
     eval harness — evaluation runs separately, giving clean periodic teacher checkpoints.

The rest adapt upstream to this setup without changing what is optimised: an iterable-dataset sampler,
a compile strategy that keeps bf16 scatter ops off Triton atomics, FLOP accounting in the logged
metrics, DTensor-aware warm-start and checkpoint loading, and the FINO guide-loss graft, which is inert
unless a run enables metadata conditioning.

Nothing is imported from dinov3 until apply/verify is called, so importing this module is
cheap and dinov3-free.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

# Set by the runner so the patched do_test can append teacher checkpoints to the index.
ACTIVE_CKPT_INDEX = None  # type: Any
# Path to a previous-stage teacher_checkpoint.pth for continuation/high-res warm-start.
WARM_START_PATH = None  # type: Any
# FINO runtime (em_ssl.fino.factors.FinoRuntime) — when set, the EM dataset emits per-sample
# metadata so the upstream GuidedSSLMetaArch guide heads can be trained. None ⇒ plain SSL.
ACTIVE_FINO = None  # type: Any
_APPLIED = False

def set_fino_runtime(runtime) -> None:
    """Install the resolved FINO factors for this run (read by make_em_dataset on every rank)."""
    global ACTIVE_FINO
    ACTIVE_FINO = runtime

def set_warm_start(path) -> None:
    """Warm-start the next run's student+teacher backbone from a previous teacher .pth.

    Used for continuation / high-resolution adaptation stages (512 -> 768 -> 1024). RoPE is
    resolution-agnostic and the patch stem is unchanged, so backbone weights transfer
    directly (strict=False; non-backbone heads re-init)."""
    global WARM_START_PATH
    WARM_START_PATH = str(path) if path else None

def _patch_init_weights_for_warm_start() -> None:
    from dinov3.train.ssl_meta_arch import SSLMetaArch

    if getattr(SSLMetaArch.init_weights, "_em_warm_patched", False):
        return
    _orig = SSLMetaArch.init_weights

    def init_weights(self):
        out = _orig(self)
        if WARM_START_PATH:
            _load_backbone_warm_start(self, WARM_START_PATH)
        return out

    init_weights._em_warm_patched = True  # type: ignore[attr-defined]
    SSLMetaArch.init_weights = init_weights

def _load_backbone_warm_start(model, path: str) -> None:
    """Copy a previous stage's backbone into the (already FSDP2-sharded) student+teacher.

    Runs from the patched ``init_weights``, after ``prepare_for_distributed_training``, so the target
    params are sharded DTensors while the checkpoint holds full tensors. Copying per-tensor and
    resharding each source to the param's mesh/placement avoids the mixed-type load. Backbone only —
    heads re-initialise — with strict=False semantics.
    """
    import torch
    from torch.distributed.tensor import DTensor, distribute_tensor

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("teacher", ckpt) if isinstance(ckpt, dict) else ckpt
    # Keep only backbone.* keys, strip the prefix.
    backbone_sd = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
    if not backbone_sd:
        warnings.warn(f"warm-start: no backbone.* keys in {path}; skipping.")
        return
    loaded = 0
    for target in ("student", "teacher"):
        mod = getattr(model, target, None)
        if mod is None or not hasattr(mod, "backbone"):
            continue
        copied = 0
        with torch.no_grad():
            for name, p in list(mod.backbone.named_parameters()) + list(mod.backbone.named_buffers()):
                src = backbone_sd.get(name)
                if src is None:
                    continue
                src = src.to(dtype=p.dtype)
                if isinstance(p, DTensor):
                    # Reshard the full checkpoint tensor to match this sharded param before copy_.
                    src = distribute_tensor(src.to(p.device), p.device_mesh, p.placements)
                else:
                    src = src.to(p.device)
                p.copy_(src)
                copied += 1
        loaded += 1
        warnings.warn(f"warm-start: copied {copied} backbone tensors into {target}.backbone from {path}.")
    if loaded == 0:
        warnings.warn(f"warm-start: no student/teacher backbone found; skipping {path}.")

def _patch_init_fsdp_checkpoint_loader() -> None:
    """Make dinov3's ``init_fsdp_model_from_checkpoint`` FSDP2/DTensor-aware.

    This is the loader behind ``student.resume_from_teacher_chkpt`` (which transfers backbone and
    heads) and the gram-teacher load. Upstream distributes every checkpoint tensor with a fixed
    heuristic, but FSDP2 ``fully_shard`` shards small params such as ``backbone.cls_token`` as
    ``Shard(0)``, so copying a replicated source into a sharded param raises. Redistributing each
    source to the target param's own mesh/placement first keeps the in-place copy placement-stable.
    """
    import dinov3.train.ssl_meta_arch as ssl_mod

    if getattr(ssl_mod.init_fsdp_model_from_checkpoint, "_em_dtensor_patched", False):
        return
    _orig = ssl_mod.init_fsdp_model_from_checkpoint

    import torch
    from pathlib import Path
    from torch.distributed.tensor import DTensor, distribute_tensor

    def init_fsdp_model_from_checkpoint(model, checkpoint_path, skip_load_keys=None,
                                        keys_not_sharded=None, process_group=None):
        skip_load_keys = skip_load_keys or []
        if Path(checkpoint_path).is_dir():  # DCP checkpoints are unaffected -> defer upstream
            return _orig(model, checkpoint_path, skip_load_keys=skip_load_keys,
                         keys_not_sharded=keys_not_sharded, process_group=process_group)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        chkpt = loaded["teacher"] if isinstance(loaded, dict) and "teacher" in loaded else loaded
        model_sd = model.state_dict()
        new_sd = {}
        for key, tensor in chkpt.items():
            if any(s in key for s in skip_load_keys):
                continue
            tgt = model_sd.get(key)
            if tgt is None:  # strict=False semantics: skip keys absent from the model
                continue
            tensor = tensor.to(dtype=tgt.dtype)
            if isinstance(tgt, DTensor):
                # Reshard the full source to this param's own mesh/placement (e.g. Shard(0) for
                # cls_token/mask_token) so the in-place copy needs no placement change.
                new_sd[key] = distribute_tensor(tensor.to(tgt.device), tgt.device_mesh, tgt.placements)
            else:
                new_sd[key] = tensor.to(tgt.device)
        model.load_state_dict(new_sd, strict=False)
        warnings.warn(f"init_fsdp_model_from_checkpoint (DTensor-aware): loaded {len(new_sd)} "
                      f"tensors from {checkpoint_path}.")

    init_fsdp_model_from_checkpoint._em_dtensor_patched = True  # type: ignore[attr-defined]
    ssl_mod.init_fsdp_model_from_checkpoint = init_fsdp_model_from_checkpoint
    import importlib
    for _modname in ("dinov3.checkpointer", "dinov3.checkpointer.checkpointer"):
        try:
            setattr(importlib.import_module(_modname), "init_fsdp_model_from_checkpoint",
                    init_fsdp_model_from_checkpoint)
        except Exception:
            pass

# --------------------------------------------------------------------------- #
# 1. in_chans injection
# --------------------------------------------------------------------------- #
def _patch_build_model() -> None:
    import dinov3.models as M
    import dinov3.models.vision_transformer as vits

    if getattr(M.build_model, "_em_patched", False):
        return
    _orig = M.build_model

    def build_model(args, only_teacher=False, img_size=224, device=None):
        in_chans = int(getattr(args, "in_chans", 3) or 3)
        arch = getattr(args, "arch", None)
        factory = vits.__dict__.get(arch) if arch else None
        if factory is None or in_chans == 3:
            return _orig(args, only_teacher=only_teacher, img_size=img_size, device=device)

        def wrapped(*a, _f=factory, _ic=in_chans, **kw):
            kw.setdefault("in_chans", _ic)
            return _f(*a, **kw)

        vits.__dict__[arch] = wrapped
        try:
            return _orig(args, only_teacher=only_teacher, img_size=img_size, device=device)
        finally:
            vits.__dict__[arch] = factory

    build_model._em_patched = True  # type: ignore[attr-defined]
    M.build_model = build_model

# --------------------------------------------------------------------------- #
# 2. EM augmentation
# --------------------------------------------------------------------------- #
def em_aug_config_from_cfg(cfg) -> "Any":
    from ..transforms import EMAugmentationConfig

    em = getattr(cfg, "em", None)
    if em is None:
        return EMAugmentationConfig()
    fields = EMAugmentationConfig().__dataclass_fields__
    kw = {}
    for k in fields:
        if k in em:
            try:
                kw[k] = em[k]
            except Exception:
                pass
    return EMAugmentationConfig(**kw)

def build_em_data_augmentation(cfg):
    """Construct EMDataAugmentationDINO from a DINOv3 cfg (1-channel mean/std + EM color ops)."""
    from ..transforms import EMDataAugmentationDINO

    crops = cfg.crops
    mean = list(crops.rgb_mean)
    std = list(crops.rgb_std)
    if len(mean) != 1 or len(std) != 1:
        warnings.warn(
            f"EM run expects single-channel crops.rgb_mean/std (len 1); got mean={mean}, std={std}. "
            "Set them to 1-element lists for in_chans=1."
        )
    return EMDataAugmentationDINO(
        global_crops_scale=tuple(crops.global_crops_scale),
        local_crops_scale=tuple(crops.local_crops_scale),
        local_crops_number=int(crops.local_crops_number),
        global_crops_size=int(crops.global_crops_size),
        local_crops_size=int(crops.local_crops_size),
        gram_teacher_crops_size=(int(crops.gram_teacher_crops_size) if crops.gram_teacher_crops_size else None),
        gram_teacher_no_distortions=bool(getattr(crops, "gram_teacher_no_distortions", False)),
        teacher_no_color_jitter=bool(getattr(getattr(cfg, "em", object()), "teacher_no_color_jitter", False)),
        mean=mean,
        std=std,
        em=em_aug_config_from_cfg(cfg),
        expected_channels=1,
    )

def _patch_build_data_augmentation() -> None:
    from dinov3.train.ssl_meta_arch import SSLMetaArch

    if getattr(SSLMetaArch.build_data_augmentation_dino, "_em_patched", False):
        return

    def build_data_augmentation_dino(self, cfg):
        return build_em_data_augmentation(cfg)

    build_data_augmentation_dino._em_patched = True  # type: ignore[attr-defined]
    SSLMetaArch.build_data_augmentation_dino = build_data_augmentation_dino

# --------------------------------------------------------------------------- #
# 3. make_dataset dispatch
# --------------------------------------------------------------------------- #
def _patch_make_dataset() -> None:
    import dinov3.data.loaders as loaders
    from .em_dataset import is_em_dataset_str, make_em_dataset

    if getattr(loaders.make_dataset, "_em_patched", False):
        return
    _orig = loaders.make_dataset

    def make_dataset(*, dataset_str, transform=None, target_transform=None, transforms=None):
        if is_em_dataset_str(dataset_str):
            return make_em_dataset(dataset_str, transform=transform, target_transform=target_transform)
        return _orig(
            dataset_str=dataset_str, transform=transform, target_transform=target_transform, transforms=transforms
        )

    make_dataset._em_patched = True  # type: ignore[attr-defined]
    # Patch every binding (modules that did `from dinov3.data import make_dataset`).
    loaders.make_dataset = make_dataset
    import dinov3.data as ddata

    ddata.make_dataset = make_dataset
    try:
        import dinov3.train.train as traintrain

        traintrain.make_dataset = make_dataset
    except Exception:
        pass

# --------------------------------------------------------------------------- #
# 3b. data loader: the streaming EM dataset uses no index-sampler and has no len()
# --------------------------------------------------------------------------- #
def _patch_make_sampler() -> None:
    """Skip DINOv3's index-sampler for the streaming EM dataset.

    EMShardDataset is an IterableDataset: it does its own per-rank/per-worker shard sharding and
    infinite reshuffle and exposes no ``len()``, which upstream ``_make_sampler`` requires.
    Returning None makes ``make_data_loader`` build a plain iterable DataLoader instead.
    """
    import dinov3.data.loaders as loaders
    from torch.utils.data import IterableDataset

    if getattr(loaders._make_sampler, "_em_patched", False):
        return
    _orig = loaders._make_sampler

    def _make_sampler(*, dataset, type=None, shuffle=False, seed=0, size=-1, advance=0):
        if isinstance(dataset, IterableDataset):
            return None
        return _orig(dataset=dataset, type=type, shuffle=shuffle, seed=seed, size=size, advance=advance)

    _make_sampler._em_patched = True  # type: ignore[attr-defined]
    loaders._make_sampler = _make_sampler

# --------------------------------------------------------------------------- #
# 3c. compile: route bf16 scatter/atomic ops to eager ATen so the heads can compile
# --------------------------------------------------------------------------- #
# Heads compile alongside the backbone; ``EM_COMPILE_HEADS=0`` selects backbone-only compilation.
_FULL_COMPILE_DEFAULT = True

# Aten scatter/accumulate ops whose Inductor Triton lowering emits ``tl.atomic_add``. On a Triton
# build without bf16 atomics that raises "atomic_add does not support bf16" at kernel-compile time,
# a few steps into a compiled run. Routing these ops to the eager ATen kernel inside the Inductor
# graph removes the bad codegen while everything else stays compiled; they fire at most a few times
# per step and the eager kernel uses the same reduction, so the result matches eager.
_BF16_ATOMIC_OPS = (
    "index_put",
    "index_put_",
    "_unsafe_index_put",
    "index_add",
    "index_add_",
    "scatter_add",
    "scatter_add_",
    "scatter_reduce",
    "scatter_reduce_",
)

def _aten_op(name: str):
    """Return ``torch.ops.aten.<name>`` (an OpOverloadPacket) or None if absent in this torch."""
    try:
        from torch.ops import aten

        return getattr(aten, name)
    except Exception:
        return None

def _force_inductor_fallback(L, op_packet) -> bool:
    """Make Inductor lower ``op_packet`` (and its overloads) to the eager ATen kernel.

    Tries the public ``make_fallback`` first (it also wires realized-inputs / layout constraints
    the extern kernel needs); on a signature/assertion mismatch, directly overrides the lowering
    table per overload. Returns True if the op is now routed through a fallback handler.
    """
    # 1) Public API — handles overloads + needs-realized-inputs itself.
    for kwargs in ({"override_decomp": True}, {"warn": False}, {}):
        try:
            L.make_fallback(op_packet, **kwargs)
            return True
        except TypeError:
            continue  # this torch's make_fallback lacks that kwarg; try a simpler form
        except Exception:
            break  # already decomposed/registered differently -> direct override below
    # 2) Direct override of each concrete overload's lowering entry.
    ok = False
    try:
        overloads = [getattr(op_packet, o) for o in op_packet.overloads()]
    except Exception:
        overloads = []
    for ov in overloads:
        try:
            if hasattr(L, "add_needs_realized_inputs"):
                L.add_needs_realized_inputs(ov)
            try:
                handler = L.fallback_handler(ov, add_to_fallback_set=False)
            except TypeError:
                handler = L.fallback_handler(ov)
            L.lowerings[ov] = handler
            ok = True
        except Exception:
            continue
    return ok

def _patch_inductor_bf16_atomic_fallback() -> bool:
    """Route the bf16-atomic scatter ops in ``_BF16_ATOMIC_OPS`` to eager ATen inside Inductor.

    Must run before the first compile (apply_em_patches runs well before model build). Idempotent.
    Returns True if at least one op was routed. On any failure it warns and returns False so the
    caller can fall back to backbone-only rather than crash mid-run.
    """
    if getattr(_patch_inductor_bf16_atomic_fallback, "_em_done", False):
        return True
    try:
        from torch._inductor import lowering as L
    except Exception as exc:  # pragma: no cover - exercised only on a torch without _inductor
        warnings.warn(
            f"bf16-atomic fallback: torch._inductor.lowering unavailable ({exc!r}); "
            "full compile may hit 'atomic_add does not support bf16'. The backbone-only path "
            "(EM_COMPILE_HEADS=0) avoids it."
        )
        return False

    routed, missing = [], []
    for name in _BF16_ATOMIC_OPS:
        op = _aten_op(name)
        if op is None:
            missing.append(name)
            continue
        if _force_inductor_fallback(L, op):
            routed.append(name)
    _patch_inductor_bf16_atomic_fallback._em_done = True  # type: ignore[attr-defined]
    print(
        "[dinov3_patch] Inductor bf16-atomic fallback installed -> eager ATen for: "
        f"{', '.join(routed) or '(none)'}"
        + (f"  [absent in this torch: {', '.join(missing)}]" if missing else "")
    )
    return bool(routed)

def _patch_compile_strategy() -> None:
    """Select torch.compile coverage for the SSL model.

      * full compile (default; EM_COMPILE_HEADS unset or 1): ``wrap_compile_block`` is left
        upstream so the DINO/iBOT heads compile too, with the bf16-atomic ops routed to eager ATen.

      * backbone-only (EM_COMPILE_HEADS=0): ``wrap_compile_block`` becomes a no-op for non-backbone
        modules, so only the transformer blocks compile and the heads run eager.

    EM_COMPILE_BF16_FALLBACK forces the fallback on (``1``) or off (``0``) independently of regime.
    """
    import os

    v = os.environ.get("EM_COMPILE_HEADS")
    full_compile = (v == "1") if v is not None else _FULL_COMPILE_DEFAULT

    fb_env = os.environ.get("EM_COMPILE_BF16_FALLBACK")
    install_fallback = full_compile if fb_env is None else (fb_env == "1")
    fallback_ok = _patch_inductor_bf16_atomic_fallback() if install_fallback else False

    if full_compile:
        if install_fallback and not fallback_ok:
            warnings.warn(
                "Full compile requested but the bf16-atomic fallback failed to install; the heads "
                "will likely crash with 'atomic_add does not support bf16'. Setting "
                "EM_COMPILE_HEADS=0 selects the backbone-only path."
            )
        print(
            "[dinov3_patch] full compile: backbone + DINO/iBOT heads "
            "(EM_COMPILE_HEADS=1; bf16-atomic ops -> eager ATen)."
        )
        return  # leave upstream wrap_compile_block intact -> heads compile

    # Backbone-only: no-op wrap_compile_block for non-backbone modules.
    import dinov3.fsdp.ac_compile_parallelize as acp

    if getattr(acp.wrap_compile_block, "_em_patched", False):
        return
    _orig = acp.wrap_compile_block

    def wrap_compile_block(module, use_cuda_graphs, is_backbone_block):
        if not is_backbone_block:
            return module  # heads eager -> no bf16-atomic scatter kernel in a head graph
        return _orig(module, use_cuda_graphs, is_backbone_block)

    wrap_compile_block._em_patched = True  # type: ignore[attr-defined]
    acp.wrap_compile_block = wrap_compile_block
    print("[dinov3_patch] compiling backbone only (EM_COMPILE_HEADS=0; SSL heads eager). "
          "Unsetting EM_COMPILE_HEADS selects full compile, the default.")

# --------------------------------------------------------------------------- #
# 3d. cumulative-FLOPs logging (compute axis for FLOP-matched comparisons)
# --------------------------------------------------------------------------- #
def _as_int_size(v) -> int:
    """crops.*_crops_size is an int per stage, but may be a list under multi-res; take the first."""
    if isinstance(v, (list, tuple)):
        return int(v[0])
    return int(v)

def _meta_arch_flops(self):
    """Compute (and cache) the analytic FLOP breakdown for this run from cfg + the built model."""
    bd = getattr(self, "_em_flops_bd", None)
    if bd is not None:
        return bd
    import dinov3.distributed as distributed

    from ..arch import resolve_arch
    from ..utils.flops import dinov3_flops_per_step

    cfg = self.cfg
    arch = resolve_arch(cfg.student.arch)
    backbone = getattr(self.student, "backbone", None)
    n_storage = int(getattr(backbone, "n_storage_tokens", 0) or 0)
    try:
        n_gpus = distributed.get_world_size()
    except Exception:
        n_gpus = 1
    mr = cfg.ibot.mask_ratio_min_max
    bd = dinov3_flops_per_step(
        embed_dim=int(self.embed_dim),
        depth=arch.depth,
        ffn_ratio=arch.ffn_ratio,
        patch_size=int(cfg.student.patch_size),
        global_crops_size=_as_int_size(cfg.crops.global_crops_size),
        local_crops_size=_as_int_size(cfg.crops.local_crops_size),
        n_local_crops=int(cfg.crops.local_crops_number),
        batch_size_per_gpu=int(cfg.train.batch_size_per_gpu),
        n_gpus=int(n_gpus),
        dino_hidden=int(cfg.dino.head_hidden_dim),
        dino_bottleneck=int(cfg.dino.head_bottleneck_dim),
        dino_prototypes=int(cfg.dino.head_n_prototypes),
        dino_nlayers=int(cfg.dino.head_nlayers),
        ibot_hidden=int(cfg.ibot.head_hidden_dim),
        ibot_bottleneck=int(cfg.ibot.head_bottleneck_dim),
        ibot_prototypes=int(cfg.ibot.head_n_prototypes),
        ibot_nlayers=int(cfg.ibot.head_nlayers),
        mask_ratio_min=float(mr[0]),
        mask_ratio_max=float(mr[1]),
        mask_sample_probability=float(cfg.ibot.mask_sample_probability),
        n_storage_tokens=n_storage,
        activation_checkpointing=bool(getattr(cfg.train, "checkpointing", False)),
    )
    self._em_flops_bd = bd
    try:
        if distributed.is_main_process():
            print(
                f"[dinov3_patch] analytic step FLOPs (aggregate, {n_gpus} GPU): "
                f"model={bd.model_pflops_per_step * 1000:.2f} TFLOP/step, "
                f"hw={bd.hw_pflops_per_step * 1000:.2f} TFLOP/step "
                f"(hw is the standard axis; logged as cum_hw_pflops)."
            )
    except Exception:
        pass
    return bd

def _patch_flops_logging() -> None:
    """Inject per-step and cumulative FLOPs into the logged metrics (opt out with EM_LOG_FLOPS=0).

    Adds ``model_tflops_per_step`` / ``hw_tflops_per_step`` and ``cum_model_pflops`` /
    ``cum_hw_pflops`` to the metrics dict returned by ``forward_backward``, so they flow through the
    unchanged train loop into ``training_metrics.json``. ``hw_*`` includes activation-checkpoint
    recompute and is the axis for FLOP-matched comparisons; ``model_*`` is the recompute-free work.
    Failures self-disable and never break training.
    """
    import os

    if os.environ.get("EM_LOG_FLOPS") == "0":
        return
    from dinov3.train.ssl_meta_arch import SSLMetaArch

    classes = [SSLMetaArch]
    try:
        from dinov3.train.guided_ssl_meta_arch import GuidedSSLMetaArch

        classes.append(GuidedSSLMetaArch)  # overrides forward_backward; wrap it too (no super() call)
    except Exception:
        pass

    def _wrap(cls):
        if getattr(cls.forward_backward, "_em_flops_patched", False):
            return
        _orig = cls.forward_backward

        def forward_backward(self, data, *, teacher_temp, iteration=0, **kwargs):
            loss, metrics = _orig(self, data, teacher_temp=teacher_temp, iteration=iteration, **kwargs)
            try:
                bd = _meta_arch_flops(self)
                step = int(iteration) + 1
                metrics["model_tflops_per_step"] = bd.model_flops_per_step / 1e12
                metrics["hw_tflops_per_step"] = bd.hw_flops_per_step / 1e12
                metrics["cum_model_pflops"] = bd.model_flops_per_step * step / 1e15
                metrics["cum_hw_pflops"] = bd.hw_flops_per_step * step / 1e15
            except Exception as exc:  # pragma: no cover - defensive; never break training
                if not getattr(self, "_em_flops_warned", False):
                    warnings.warn(f"FLOPs logging disabled ({exc!r}).")
                    self._em_flops_warned = True  # type: ignore[attr-defined]
            return loss, metrics

        forward_backward._em_flops_patched = True  # type: ignore[attr-defined]
        cls.forward_backward = forward_backward

    for cls in classes:
        _wrap(cls)

# --------------------------------------------------------------------------- #
# 4. do_test -> teacher export + checkpoint index, skip eval harness
# --------------------------------------------------------------------------- #
def _parse_iteration(iteration: Any) -> int:
    s = str(iteration)
    for sep in ("_",):
        if sep in s:
            s = s.split(sep)[-1]
    try:
        return int(s)
    except Exception:
        return -1

def export_teacher_checkpoint(cfg, model, iteration) -> Path | None:
    """Replicate DINOv3's non-sharded teacher export and register it in the index."""
    import torch
    from dinov3 import distributed
    from torch.distributed.tensor import DTensor

    eval_dir = Path(cfg.train.output_dir) / "eval" / str(iteration)
    is_main = True
    try:
        is_main = distributed.is_subgroup_main_process()
    except Exception:
        pass

    state = model.model_ema.state_dict()
    for k, t in list(state.items()):
        if isinstance(t, DTensor):
            state[k] = t.full_tensor()
    if not is_main:
        return None
    eval_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = eval_dir / "teacher_checkpoint.pth"
    torch.save({"teacher": state}, ckpt_path)

    if ACTIVE_CKPT_INDEX is not None:
        try:
            step = _parse_iteration(iteration)
            ACTIVE_CKPT_INDEX.add(
                step=step,
                kind="teacher",
                path=str(ckpt_path),
                crop_size=int(cfg.crops.global_crops_size),
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"checkpoint_index update failed: {exc!r}")
    return ckpt_path

def _patch_do_test(skip_eval_harness: bool = True) -> None:
    import dinov3.train.train as traintrain

    if getattr(traintrain.do_test, "_em_patched", False):
        return
    _orig = traintrain.do_test

    def do_test(cfg, model, iteration, process_group=None, do_low_freq=False):
        ckpt = export_teacher_checkpoint(cfg, model, iteration)
        if skip_eval_harness:
            return  # no downstream decoder probe is configured here; keep the teacher checkpoint
        return _orig(cfg, model, iteration, process_group=process_group, do_low_freq=do_low_freq)

    do_test._em_patched = True  # type: ignore[attr-defined]
    traintrain.do_test = do_test

# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def apply_em_patches(skip_eval_harness: bool = True) -> None:
    """Install all EM patches (idempotent).

    Includes the FINO graft (mask-aware ``GuidedSSLMetaArch._compute_guide_losses``). Only a run
    with metadata factors selects ``GuidedSSLMetaArch`` as ``MODEL.META_ARCHITECTURE``, so the graft
    is inert for baseline (non-guided) runs.
    """
    global _APPLIED
    _patch_build_model()
    _patch_build_data_augmentation()
    _patch_make_dataset()
    _patch_make_sampler()
    _patch_compile_strategy()
    _patch_flops_logging()
    _patch_do_test(skip_eval_harness=skip_eval_harness)
    _patch_init_weights_for_warm_start()
    _patch_init_fsdp_checkpoint_loader()
    from ..fino.meta_arch_patch import apply_fino_grafts

    apply_fino_grafts()
    _APPLIED = True

def verify_one_channel(arch: str = "vit_small") -> bool:
    """Build a 1-channel ViT through the SSL build path and assert the conv stem is 1-channel."""
    apply_em_patches()
    from dinov3.configs import get_default_config
    from dinov3.models import build_model_from_cfg

    cfg = get_default_config()
    cfg.student.arch = arch
    cfg.student.in_chans = 1
    cfg.student.patch_size = 16
    cfg.crops.global_crops_size = 224
    student, teacher, embed_dim = build_model_from_cfg(cfg, only_teacher=False)
    w = student.patch_embed.proj.weight
    ok = tuple(w.shape)[1] == 1
    if not ok:
        raise AssertionError(f"in_chans patch failed: patch_embed.proj.weight={tuple(w.shape)} (want C_in=1)")
    print(f"[verify_one_channel] {arch}: patch_embed.proj.weight={tuple(w.shape)} embed_dim={embed_dim} -> 1-channel OK")
    return True
