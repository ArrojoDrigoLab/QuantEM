"""Trainer + evaluator for the shared DoDNet organelle-conditioned model.

One shared frozen+LoRA base + one shared DoDNet head trained on the mixed multi-organelle dataset: each step
sets the head's organelle code from the crop's ``org_code`` and supervises the binary-per-organelle target
(+ instance target for mito). ``evaluate_multi`` evals per organelle by setting the code and running the
standard sliding-window eval (``evaluate_head``, so both metrics: semantic and true-instance
``inst_*`` for mito, semantic for ER), then reports each via ``common.eval_report``.

Mirrors ``scale/two_scale.py::train_two_scale`` for the baseline-matched optim (AdamW param groups, warmup→cosine,
``build_loss``, picklable DataLoader worker-init, empty-loader guard). The model is a plain ``SegModel`` assembled by
``harness.train.build_segmodel`` with ``decoder.type == "dodnet"`` (registered by ``dodnet_head`` at
import) — so LoRA adaptation + neck + the dynamic head are wired by the standard factory. The only novelty is
setting ``model.decoder.set_organelle_code(...)`` each forward.

Runs without a GPU: torch only at module top; heavy imports are lazy. num_workers=0 for CPU smoke.
"""

from __future__ import annotations

import math

import torch

# Importing dodnet_head registers the "dodnet" decoder into segmentation training's DECODERS dict (side effect at import).
from . import dodnet_head as _dodnet  # noqa: F401  (import registers the decoder)
from .mixed_dataset import build_mixed_dataset


def _worker_reseed(worker_id: int) -> None:
    """Picklable DataLoader worker-init (Windows spawn-safe): reseed the per-worker dataset RNG."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(worker_id)


def _lr_scale(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / float(max(1, warmup))
    prog = (step - warmup) / float(max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))


def build_shared_dodnet_model(cfg, encoder, *, n_organelles: int = 2, mid_channels: int = 8,
                              n_dynamic: int = 3, mechanism: str = "dynamic"):
    """Assemble the shared SegModel with a ``dodnet`` decoder (LoRA + neck from cfg, like build_segmodel).

    Forces ``cfg.decoder.type='dodnet'`` and injects the DoDNet params so the registered builder sizes the
    controller for K organelles. Returns the ``SegModel``; its ``.decoder`` is the ``DoDNetHead``.
    """
    import copy

    from ...harness.train import build_segmodel

    c = copy.deepcopy(cfg)
    c.decoder.type = "dodnet"
    params = dict(c.decoder.params or {})
    params.update({"n_organelles": int(n_organelles), "mid_channels": int(mid_channels),
                   "n_dynamic": int(n_dynamic), "mechanism": mechanism})
    c.decoder.params = params
    c.data.num_classes = 2  # binary-per-organelle
    model = build_segmodel(c, encoder)
    model._n_organelles = int(n_organelles)
    return model


def train_multi(cfg, encoder, per_organelle: dict, data_root_unused=None, device: str = "cpu", *,
                n_organelles: int = 2, mid_channels: int = 8, n_dynamic: int = 3, mechanism: str = "dynamic",
                tasks: dict | None = None, balance: str = "raw", logger=None):
    """Train the shared DoDNet model on the mixed multi-organelle dataset, with baseline-matched optim/steps.

    ``per_organelle``: ``{organelle: (records, data_root)}`` (from ``mixed_dataset.load_per_organelle``).
    Returns the trained ``SegModel`` (``.eval()``). ``data_root_unused`` is accepted for signature symmetry
    with the other trainers (the mixed dataset carries its own per-organelle roots).
    """
    from torch.utils.data import DataLoader

    from ...models.losses import build_loss

    model = build_shared_dodnet_model(cfg, encoder, n_organelles=n_organelles, mid_channels=mid_channels,
                                      n_dynamic=n_dynamic, mechanism=mechanism).to(device)
    ds = build_mixed_dataset(per_organelle, cfg, encoder.image_mean, encoder.image_std,
                             patch_size=encoder.patch_size, n_organelles=n_organelles, tasks=tasks,
                             balance=balance)
    if len(ds) == 0:
        raise ValueError("train_multi: mixed dataset is empty (no records for any organelle).")
    slot_to_org = {slot: org for org, slot in ds.code_slot.items()}   # for per-organelle gradient-step counting

    nw = int(cfg.num_workers)
    bs = min(int(cfg.optim.batch_size), max(1, len(ds)))       # never > dataset (else drop_last empties it)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=(len(ds) >= int(cfg.optim.batch_size)),
                        num_workers=nw, collate_fn=ds.collate,
                        worker_init_fn=(_worker_reseed if nw > 0 else None))

    criterion = build_loss(cfg.loss, 2).to(device)
    # Param groups: neck+decoder @ lr; LoRA adapters @ adapter_lr (mirrors train_segmodel / two_scale).
    core = list(model.neck.parameters()) + list(model.decoder.parameters())
    groups = [{"params": [p for p in core if p.requires_grad], "base_lr": cfg.optim.lr}]
    if getattr(model, "encoder_trainable", False) and getattr(encoder, "_conv_lora", None) is not None:
        groups.append({"params": list(encoder._conv_lora.parameters()), "base_lr": cfg.optim.adapter_lr})
    opt = torch.optim.AdamW([{**g, "lr": g["base_lr"]} for g in groups], weight_decay=cfg.optim.weight_decay)

    use_amp = bool(cfg.amp) and str(device).startswith("cuda")
    total, warmup = int(cfg.optim.max_steps), int(cfg.optim.warmup_steps)
    model.train(); encoder.backbone.eval()
    org_crops_seen = {org: 0 for org in ds.organelles}     # per-organelle gradient-step exposure
    step = 0
    while step < total:
        n_in_epoch = 0
        for batch in loader:
            n_in_epoch += 1
            if step >= total:
                break
            for slot in batch["org_idx"].tolist():         # count crops per organelle actually trained on
                org_crops_seen[slot_to_org.get(int(slot), "?")] = org_crops_seen.get(slot_to_org.get(int(slot), "?"), 0) + 1
            x = batch["image"].to(device)
            y = batch["target"].to(device)
            inst = batch.get("inst")
            inst = inst.to(device) if inst is not None else None
            # DoDNet: the per-crop organelle code is set on the head before the forward.
            model.decoder.set_organelle_code(batch["org_code"].to(device))
            scale = _lr_scale(step, warmup, total)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss, report = criterion(logits, y, model.aux_logits, inst=inst)
            loss.backward()
            if cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()
            if logger and step % 50 == 0:
                logger(step, float(loss.detach()))
            step += 1
        if n_in_epoch == 0:
            raise RuntimeError(f"train_multi: DataLoader yielded no batches "
                               f"(len(ds)={len(ds)}, batch_size={cfg.optim.batch_size}).")
    # Report the step allocation and the dataset imbalance alongside the sharing verdict; both are
    # properties of the sampling rather than of the seed. Under raw balance the majority organelle
    # gets most gradient steps.
    total_crops = sum(org_crops_seen.values()) or 1
    model._train_stats = {
        "balance": balance, "total_steps": total, "batch_size": bs,
        "per_organelle_crops_trained": org_crops_seen,
        "per_organelle_step_fraction": {o: c / total_crops for o, c in org_crops_seen.items()},
        "dataset_ratio": ds.ratio_report(),
        "note": ("matched-per-organelle-steps (balanced sampling)" if balance == "balanced"
                 else "matched-total-steps (raw concat) — the majority organelle dominates gradients, so "
                      "the balanced-sampling variant is reported alongside; the sharing verdict can "
                      "differ between the two"),
    }
    return model.eval()


def evaluate_multi(model, per_organelle_records: dict, cfg, device, mean: float, std: float, *,
                   n_organelles: int = 2, tasks: dict | None = None) -> dict:
    """Eval the shared DoDNet model per organelle: set the code, run ``evaluate_head`` (both metrics).

    ``per_organelle_records``: ``{organelle: (records, data_root)}`` for the eval split. For each organelle
    it deep-copies cfg with that organelle's ``data.organelle`` + ``data.task`` (so instance-eval
    fires for mito and the semantic path for ER), sets the head's organelle code, and scores. Returns
    ``{organelle: {"summary":..., "per_crop":...}}`` (the shape ``eval_report.assemble_report`` consumes).
    """
    import copy

    from .mixed_dataset import one_hot, subset_code_map
    from ...harness.evaluate import evaluate_head

    default_task = {"mito": "instance", "ld": "instance", "er": "semantic", "nucleus": "semantic"}
    # Subset-local code slots, matching training (MixedOrganelleDataset.code_slot); otherwise the eval code
    # points at a different (or out-of-range) head slot than the crops were trained with.
    code_slot = subset_code_map(list(per_organelle_records.keys()))
    out: dict = {}
    for org, (records, data_root) in per_organelle_records.items():
        if not records:
            continue
        c = copy.deepcopy(cfg)
        c.data.organelle = org
        c.data.task = (tasks or {}).get(org, default_task.get(org, "semantic"))
        c.data.num_classes = 2
        oidx = code_slot[org.lower()]
        model.decoder.set_organelle_code(one_hot(oidx, n_organelles))
        ev_recs = records
        if len(ev_recs) > 300:   # cap eval cost (mutex watershed over every mito crop is slow); stratified+seeded
            from ...harness.dataset import subset_fraction
            ev_recs = subset_fraction(ev_recs, 300 / len(ev_recs), seed=int(getattr(c.optim, "seed", 0) or 0))
        out[org] = evaluate_head(model, ev_recs, c, data_root, device, mean=mean, std=std)
    return out
