"""Train one decoder head on a pretrained encoder.

The encoder is frozen on the default path. When ``harness.encoder_adaptation.apply_adaptation`` has
already made some of its parameters trainable — LoRA adapters, the LayerNorms, the last N blocks or
the whole backbone — those parameters train alongside the decoder in a second optimizer group at
``cfg.adapter_lr``.

Loss = ``cfg.ce_weight`` x cross-entropy (ignore_index=255) + ``cfg.dice_weight`` x soft-Dice on the
foreground over valid pixels. AdamW with linear warmup + cosine decay. bf16 autocast on CUDA. Returns
the trained decoder.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..constants import IGNORE_INDEX
from .dataset import ProbeTrainDataset
from .decoder import build_decoder

def _probe_seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn (module-level so it pickles): re-seed the dataset's RNG per worker so
    augmentation is diverse across record revisits yet reproducible per (seed, worker)."""
    import torch.utils.data as _tud
    info = _tud.get_worker_info()
    if info is not None:
        info.dataset.reseed(worker_id + 1)

def _center_crop(t: torch.Tensor, s: int) -> torch.Tensor:
    """Central ``s x s`` crop of the last two dims; a no-op when already ``s``."""
    H, W = t.shape[-2], t.shape[-1]
    if H == s and W == s:
        return t
    oh, ow = max(0, (H - s) // 2), max(0, (W - s) // 2)
    return t[..., oh:oh + s, ow:ow + s]

def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice on the foreground class over valid (non-ignore) pixels."""
    valid = (target != IGNORE_INDEX).float()
    probs = F.softmax(logits, dim=1)[:, 1]  # [B,H,W] P(foreground)
    tgt = (target == 1).float()
    p = probs * valid
    t = tgt * valid
    num = 2 * (p * t).sum(dim=(1, 2)) + eps
    den = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) + eps
    return (1 - num / den).mean()

def _lr_at(step: int, max_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(max_steps - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(prog, 1.0)))

def train_head(encoder, train_records, cfg, derived_root, layers, device, logger=None, tag=""):
    """Train and return the decoder head for one organelle on one encoder + label fraction.

    ``build_decoder`` picks the class from ``cfg.decoder``: ``SegDecoder`` for ``linear`` /
    ``light_conv``, ``UPerNetDecoder`` for ``upernet``, ``UNetDecoder`` for ``unet``."""
    encoder = encoder.to(device)
    decoder = build_decoder(
        embedding_dim=encoder.embedding_dim, n_layers=len(layers),
        num_classes=cfg.num_classes, patch_size=encoder.patch_size, mode=cfg.decoder,
    ).to(device)

    ds = ProbeTrainDataset(train_records, derived_root, cfg, encoder.image_mean, encoder.image_std)
    if len(ds) == 0:
        raise ValueError(f"No training samples for {tag}")
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        drop_last=False, pin_memory=(device != "cpu"),
        worker_init_fn=_probe_seed_worker if cfg.num_workers > 0 else None,
    )
    # Encoder adaptation: apply_adaptation (run by the worker after building the encoder) may have
    # installed LoRA adapters or unfrozen the LayerNorms / the last N blocks / the whole backbone ->
    # those params carry requires_grad. Collect them into a second optimizer group at cfg.adapter_lr;
    # whatever apply_adaptation left untouched stays frozen. When nothing is trainable on the encoder,
    # this is a plain decoder-only optimizer.
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    enc_trainable = len(enc_params) > 0
    param_groups = [{"params": list(decoder.parameters()), "lr": cfg.lr, "base_lr": cfg.lr}]
    if enc_trainable:
        param_groups.append({"params": enc_params, "lr": cfg.adapter_lr, "base_lr": cfg.adapter_lr})
    opt = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    use_amp = bool(cfg.amp) and str(device).startswith("cuda")
    # A GradScaler supplies loss scaling: back-propagating through the encoder in fp16 without it
    # underflows and collapses the prediction to all-background, while a decoder-only backward is
    # unaffected. It is therefore enabled only when QUANTEM_FORCE_FP16=1 marks the run as fp16 and the
    # encoder is trainable; otherwise it is a pass-through around backward()/step(). The autocast
    # context below always requests bf16.
    _fp16_forced = os.environ.get("QUANTEM_FORCE_FP16") == "1"
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and _fp16_forced and enc_trainable))

    cmp = cfg.effective_compare()  # the decoder predicts and is supervised on the central compare region
    decoder.train()
    step = 0
    data_iter = _cycle(loader)
    while step < cfg.max_steps:
        x, y = next(data_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # One shared warmup+cosine, scaled per group by its base LR (decoder @ lr, adapters @ adapter_lr).
        sched = _lr_at(step, cfg.max_steps, cfg.warmup_steps, 1.0)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * sched
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _nullctx()
        if not enc_trainable:
            with torch.no_grad():
                feats = encoder.extract(x, layers)  # frozen: central-token-cropped when compare<tile
        with ctx:
            if enc_trainable:
                feats = encoder.extract_train(x, layers)  # grad flows to LoRA / last-N encoder params
            logits = decoder([f.float() for f in feats], out_hw=(cmp, cmp))
            y_cmp = _center_crop(y, cmp)  # supervise only the common region (== y when compare==tile)
            ce = F.cross_entropy(logits, y_cmp, ignore_index=IGNORE_INDEX)
            dice = soft_dice_loss(logits, y_cmp)
            loss = cfg.ce_weight * ce + cfg.dice_weight * dice
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if enc_trainable:  # stabilize the encoder fine-tune: unscale (if fp16-scaled) then clip grads
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for grp in opt.param_groups for p in grp["params"]], max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        if logger is not None and step % 50 == 0:
            logger(step, {"loss": float(loss.detach()), "ce": float(ce.detach()),
                          "dice": float(dice.detach()), "lr": opt.param_groups[0]["lr"]})
        step += 1
    decoder.eval()
    return decoder

def _cycle(loader):
    while True:
        for b in loader:
            yield b

class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
