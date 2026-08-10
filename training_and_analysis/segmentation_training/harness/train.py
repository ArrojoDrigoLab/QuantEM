"""Segmentation neck+decoder training loop (every arm launched through ``harness.run_seg`` trains here).

Assembles a :class:`~segmentation_training.models.base.SegModel` (encoder -> neck -> decoder) and trains the
neck + decoder, plus — when ``cfg.encoder.adapt`` is not ``frozen`` — the encoder-side LoRA adapters or
unfrozen base blocks that mode selects. The SegModel/registry design:

  * The decoder consumes the neck's feature pyramid, not raw taps — the neck turns the stride-16 taps
    into the STRIDES pyramid (contract in ``segmentation_training.models.base``).
  * The loss is a registry-built :class:`~segmentation_training.models.losses.CombinedLoss` (loss term stack) taking the
    decoder's deep-supervision / instance ``aux_logits`` as an optional third argument.
  * Optimizer param groups: neck + decoder at ``cfg.optim.lr`` (decoder scaled by ``decoder_lr_mult``);
    the image-style conditioner, when enabled, also at ``cfg.optim.lr``; trainable encoder params at
    ``cfg.optim.adapter_lr``. Linear-warmup + cosine to ``max_steps``.

No GPU is needed: imports only torch (+ numpy transitively via the dataset). Heavy neck/decoder arms fail
loudly at build via ``base.require``; the encoder runs under ``no_grad`` while it is fully frozen, and
with grad once an adaptation mode has made any of its params trainable.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models.base import SegModel
from ..models.decoders import build_decoder
from ..models.losses import build_loss
from ..models.necks import build_neck


# --------------------------------------------------------------------------- #
# Model assembly
# --------------------------------------------------------------------------- #
def build_segmodel(cfg, encoder, field_sizes: dict | None = None) -> SegModel:
    """Wire an ``encoder`` + neck + decoder (+ optional image-style conditioner) into a ``SegModel``.

    ``cfg.encoder.adapt`` decides what trains on the encoder side: the LoRA modes install adapters inside
    the otherwise-frozen backbone, ``last_n``/``full`` unfreeze base blocks, and any of them flips
    ``encoder_trainable`` so features run with grad. When ``cfg.cond.enabled`` the image-style conditioner
    (style encoder + FiLM + optional MixStyle/adversary) is built and attached — ``field_sizes`` (from the
    metadata vocab) sizes its embeddings/adversary heads.
    """
    layers = cfg.encoder.resolved_layers(encoder.depth)
    n_taps = len(layers)
    embed_dim = encoder.embedding_dim

    out_channels = int((cfg.neck.params or {}).get("out_channels", 256))
    neck = build_neck(cfg.neck, embed_dim, n_taps, encoder.patch_size, out_channels=out_channels)
    decoder = build_decoder(cfg.decoder, neck.out_channels, neck.strides, cfg.data.num_classes)

    # Encoder adaptation: driven by cfg.encoder.adapt (frozen|lora|lora_ln|cond_lora|last_n|full; the
    # mode -> trainable-params policy lives in hooks/encoder_adaptation.py). The trainable params (LoRA
    # adapters, or unfrozen base blocks) live on the encoder; base weights stay frozen unless the mode
    # says otherwise.
    from ..hooks.encoder_adaptation import apply_adaptation
    from ..hooks.film_conditioning import build_conditioner

    adapt = (getattr(cfg.encoder, "adapt", "frozen") or "frozen")
    encoder_trainable = apply_adaptation(encoder, adapt, cfg.encoder.adapt_params or {})

    conditioner = build_conditioner(cfg, neck, decoder, field_sizes=field_sizes, embed_dim=embed_dim)
    return SegModel(encoder, neck, decoder, layers, encoder_trainable=encoder_trainable,
                    conditioner=conditioner)


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #
def _lr_scale(step: int, max_steps: int, warmup: int) -> float:
    """Linear-warmup then cosine-to-zero multiplier in [0, 1] (scales each group's base lr)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / float(max(warmup, 1))
    prog = (step - warmup) / float(max(max_steps - warmup, 1))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))


# --------------------------------------------------------------------------- #
# Optimizer param groups
# --------------------------------------------------------------------------- #
def _param_groups(model: SegModel, cfg):
    """AdamW param groups: neck @ lr, decoder @ lr*decoder_lr_mult, conditioner @ lr (when one is
    attached), trainable encoder params @ adapter_lr.

    Each group carries its own ``base_lr`` so the shared warmup/cosine multiplier scales all groups
    together while preserving their relative ratios. Only params with ``requires_grad`` are included.
    """
    o = cfg.optim
    seen: set[int] = set()
    groups = []

    def _collect(module, base_lr):
        params = []
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        if params:
            groups.append({"params": params, "base_lr": float(base_lr), "lr": float(base_lr)})

    _collect(model.neck, o.lr)
    _collect(model.decoder, o.lr * float(o.decoder_lr_mult))
    if getattr(model, "conditioner", None) is not None:
        # The image-style conditioner (style encoder + FiLM heads + adversary) trains at the neck/decoder base lr.
        _collect(model.conditioner, o.lr)
    if model.encoder_trainable:
        # LoRA adapters (and any unfrozen base blocks) live on the encoder; base is frozen otherwise.
        _collect(model.encoder, o.adapter_lr)
    return groups


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_segmodel(cfg, encoder, train_records, data_root, device, logger=None, tag="",
                   run_dir=None) -> SegModel:
    """Train and return a ``SegModel`` for one arm on one frozen encoder + training record set.

    ``logger`` (optional) is called ``logger(step, {metric: value})`` every ~50 steps. An em_ssl
    ``MetricLogger`` is also driven best-effort if importable; its absence is never fatal.
    """
    from .dataset import SegTrainDataset  # imported lazily so importing this module stays light

    # Image-style conditioning metadata vocab (source id + categorical fields) — built from the training records, shared with eval,
    # saved into head.pt. Only when conditioning is enabled (else None, leaving arms without conditioning unchanged).
    vocab = None
    field_sizes = None
    cond_on = bool(getattr(getattr(cfg, "cond", None), "enabled", False))
    if cond_on:
        from .meta import MetaVocab, conditioning_fields
        vocab = MetaVocab.build(train_records, conditioning_fields(cfg.cond))
        field_sizes = vocab.sizes()

    model = build_segmodel(cfg, encoder, field_sizes=field_sizes).to(device)
    if model.conditioner is not None:
        model.conditioner.vocab = vocab
        model._meta_vocab = vocab

    # --- data ---------------------------------------------------------------
    ds = SegTrainDataset(train_records, data_root, cfg, encoder.image_mean, encoder.image_std,
                         patch_size=encoder.patch_size, vocab=vocab)
    if len(ds) == 0:
        raise ValueError(f"No training samples for {tag or cfg.name}")
    worker_init = None
    if cfg.num_workers > 0:
        try:
            from em_ssl.utils.reproducibility import worker_init_fn as worker_init
        except Exception:
            worker_init = None
    loader = DataLoader(
        ds, batch_size=cfg.optim.batch_size, shuffle=True, num_workers=cfg.num_workers,
        drop_last=False, pin_memory=str(device).startswith("cuda"),
        worker_init_fn=worker_init,
    )

    # --- optimizer / schedule / loss ---------------------------------------
    opt = torch.optim.AdamW(
        _param_groups(model, cfg), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    criterion = build_loss(cfg.loss, cfg.data.num_classes,
                           decoder_type=getattr(cfg.decoder, "type", None)).to(device)
    use_amp = bool(cfg.amp) and str(device).startswith("cuda")
    grad_clip = float(cfg.optim.grad_clip)
    max_steps = int(cfg.optim.max_steps)
    warmup = int(cfg.optim.warmup_steps)

    metric_logger = _make_metric_logger(tag or cfg.name)

    model.train()
    model.encoder.backbone.eval()  # frozen base backbone always in eval (BN/dropout off)
    data_iter = _cycle(loader)
    grad_accum = max(1, int(getattr(cfg.optim, "grad_accum", 1)))
    # image-style conditioning: gradient-reversed adversary ramp (0 disables). lambda ramps over global training progress.
    grl_max = float(getattr(getattr(cfg, "cond", None), "grad_reversal", 0.0) or 0.0)
    adversarial = cond_on and grl_max > 0 and model.conditioner is not None \
        and model.conditioner.adversary is not None
    if adversarial:
        from ..models.conditioning.grl import dann_lambda
    step = 0
    # Lightweight progress telemetry: a windowed it/s + ETA, printed to stdout (captured per-arm by
    # the caller) and mirrored to run_dir/progress.json for cheap polling. run_seg is otherwise silent
    # during training, so this is the only live signal into how fast an arm is going.
    _t0 = time.time()
    _win_t, _win_step = _t0, 0
    while step < max_steps:
        # ``step`` counts optimizer updates (not micro-batches), so max_steps / warmup_steps and the
        # samples-seen (= batch_size * grad_accum * step) are identical across arms regardless of
        # grad_accum. A memory-heavy arm sets a small batch_size + grad_accum > 1 to
        # match the others' effective batch (batch_size * grad_accum) without OOM, keeping the decoder
        # comparison a fair, same-effective-batch one.
        scale = _lr_scale(step, max_steps, warmup)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * scale

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        report: dict = {}
        for _micro in range(grad_accum):
            batch = next(data_iter)  # SegTrainDataset yields dict{image,target[,inst][,meta]}
            meta = batch.pop("meta", None)
            batch = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            x, y = batch["image"], batch["target"]

            # image-style conditioning context (metadata / source ids / DANN alpha / pooled-global foreground mask) for this micro-batch.
            if cond_on and model.conditioner is not None:
                meta = ({k: v.to(device) for k, v in meta.items()} if meta else None)
                alpha = (dann_lambda(step / max(max_steps, 1), lambda_max=grl_max)
                         if adversarial else 0.0)
                # pooled-global FiLM: teacher-force the appearance code from GT foreground (the "confident correct" regions).
                fg_mask = ((y == 1).float()[:, None]
                           if getattr(model.conditioner, "confident_style", None) is not None else None)
                model.set_conditioning_context(
                    meta=meta, source_ids=(meta.get("dataset") if meta else None), alpha=alpha,
                    fg_mask=fg_mask)

            # The query decoder carries its own reference set-criterion and exposes
            # ``decoder.compute_loss(batch, device)`` (Hungarian matching over instance targets), which it
            # advertises with ``uses_query_loss``; every other decoder is scored by the dense build_loss
            # criterion on the [B,K,H,W] semantic logits.
            use_decoder_loss = getattr(model.decoder, "uses_query_loss", False)
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _nullctx()
            with ctx:
                logits = model(x)  # populates the decoder's raw outputs + aux_logits for this batch
                if use_decoder_loss:
                    loss, report = model.decoder.compute_loss(batch, device)
                else:
                    # inst (instance-id map) drives the instance-head losses (affinity / panoptic_instance);
                    # None for arms/data without instance labels -> those terms no-op.
                    loss, report = criterion(logits, y, model.aux_logits, inst=batch.get("inst"))
                # image-style conditioning gradient-reversed source adversary: CE against the batch metadata; the GRL
                # (inside the adversary) already negates+scales the gradient into the style encoder.
                if adversarial and meta is not None and model.conditioner.last_adv_logits:
                    adv = _adversary_loss(model.conditioner.last_adv_logits, meta)
                    loss = loss + adv
                    report = {**report, "adv": float(adv.detach())}
            (loss / grad_accum).backward()  # scale so the accumulated grad == mean over the effective batch
            accum_loss += float(loss.detach()) / grad_accum

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                (p for grp in opt.param_groups for p in grp["params"]), grad_clip
            )
        opt.step()

        if step % 50 == 0 or step == max_steps - 1:
            record = {**report, "loss": accum_loss, "lr": opt.param_groups[0]["lr"]}
            if logger is not None:
                logger(step, record)
            if metric_logger is not None:
                _log_metrics(metric_logger, step, record)
        if step % 100 == 0 or step == max_steps - 1:
            now = time.time()
            win_rate = (step - _win_step) / max(now - _win_t, 1e-6)  # throughput over the last window
            cum_rate = (step + 1) / max(now - _t0, 1e-6)
            eta = (max_steps - step) / max(win_rate, 1e-9)
            print(f"[{tag}] step {step}/{max_steps} | {now - _t0:.0f}s | {win_rate:.2f} it/s "
                  f"(cum {cum_rate:.2f}) | ETA {eta / 60:.1f}min | loss {accum_loss:.4f}", flush=True)
            if run_dir is not None:
                try:
                    (Path(run_dir) / "progress.json").write_text(json.dumps(
                        {"step": step, "max_steps": max_steps, "elapsed_s": round(now - _t0, 1),
                         "it_per_s": round(win_rate, 3), "eta_min": round(eta / 60, 2),
                         "loss": round(float(accum_loss), 4)}))
                except OSError:
                    pass
            _win_t, _win_step = now, step
        step += 1

    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _adversary_loss(adv_logits: dict, meta: dict):
    """Mean cross-entropy of the gradient-reversed source adversary over its targets."""
    import torch.nn.functional as F

    terms = [F.cross_entropy(logits, meta[t]) for t, logits in adv_logits.items() if t in meta]
    if not terms:
        return torch.zeros((), device=next(iter(adv_logits.values())).device)
    return sum(terms) / len(terms)


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_metric_logger(tag: str):
    """Best-effort em_ssl MetricLogger; returns None if unavailable, which is never fatal."""
    try:
        from em_ssl.utils import MetricLogger  # type: ignore

        return MetricLogger(delimiter="  ")
    except Exception:
        try:
            from em_ssl.utils.logging import MetricLogger  # type: ignore

            return MetricLogger(delimiter="  ")
        except Exception:
            return None


def _log_metrics(metric_logger, step: int, record: dict) -> None:
    try:
        metric_logger.update(**{k: float(v) for k, v in record.items()})
    except Exception:
        pass
