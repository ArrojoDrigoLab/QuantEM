"""The shared config template every experiment line generates its arms from.

Keeps all arms matched to the baseline they are compared against on steps, seed, adaptation and
evaluation settings, so a measured difference belongs to the arm rather than to an incidental
configuration change. The values are those of the trained adapted baseline for each organelle.

Per organelle the baseline recipe is a ``resnet34_detail`` neck, ``dpt`` for ER and ``affinity_mws``
for mitochondria, soft-Dice + BCE loss, and rank-8 convolutional LoRA on the encoder.
``encoder.run_dir`` is left null and supplied at launch, so the same template serves either encoder.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

# Per-organelle baseline recipe (neck/decoder/task/num_classes).
ORG_RECIPE = {
    "er": {"neck": "resnet34_detail", "decoder": "dpt", "task": "semantic", "num_classes": 2},
    "mito": {"neck": "resnet34_detail", "decoder": "affinity_mws", "task": "instance", "num_classes": 2},
    # Secondary organelles follow the primary of the same task: LD reuses the mito instance setup,
    # nucleus the ER semantic one (the same split mixed_dataset applies).
    "ld": {"neck": "resnet34_detail", "decoder": "affinity_mws", "task": "instance", "num_classes": 2},
    "nucleus": {"neck": "resnet34_detail", "decoder": "dpt", "task": "semantic", "num_classes": 2},
}


def base_config(organelle: str, *, name: str | None = None, seed: int = 0) -> dict:
    """A full SegConfig-shaped dict = the baseline recipe for ``organelle`` (run_dir supplied at launch)."""
    o = ORG_RECIPE[organelle]
    return {
        "name": name or f"e2base_{organelle}",
        "notes": f"baseline-matched adapted-base recipe for {organelle} (OmniEM ViT-L + Conv-LoRA rank8).",
        "device": "cuda", "amp": True, "num_workers": 8,
        "encoder": {
            "run_dir": None, "checkpoint_step": None, "tile_size": 512,
            "feature_layers": "last4", "apply_encoder_norm": True,
            "adapt": "lora", "adapt_params": {"rank": 8, "conv": True},
        },
        "neck": {"type": o["neck"], "params": {}},
        "decoder": {"type": o["decoder"], "params": {}},
        "loss": {"terms": [{"type": "dice_bce", "weight": 1.0, "params": {}}], "ignore_index": 255},
        "data": {
            "organelle": organelle, "canonical_nm": None, "group": None, "bucket": "canonical",
            "train_split": "train", "val_split": "val", "test_split": "test",
            "num_classes": o["num_classes"], "task": o["task"], "min_fg_frac_keep": 0.1,
            "manifest_name": "manifest.jsonl",
            "aug_flip": True, "aug_rot90": True, "aug_elastic": True, "aug_elastic_alpha": 20.0,
            "aug_elastic_sigma": 6.0, "aug_brightness": 0.15, "aug_contrast": 0.15,
            "aug_gamma": 0.2, "aug_noise_std": 0.02,
        },
        "optim": {
            "max_steps": 5000, "warmup_steps": 625, "batch_size": 8, "lr": 1e-3, "weight_decay": 1e-4,
            "decoder_lr_mult": 1.0, "adapter_lr": 1e-3, "grad_checkpoint": False, "grad_clip": 0.0,
            "grad_accum": 1, "seed": seed,
        },
        "eval": {
            "overlap": 0.25, "boundary_theta_frac": 0.0075, "boundary_dilation_ratio": 0.02,
            "fg_threshold": 0.5, "auprc_bins": 256, "hd95_pct": 95.0, "instance_min_size": 16,
            "max_region_px": 4_000_000, "bootstrap_n": 1000, "bootstrap_ci": 95.0,
        },
    }


def with_overrides(base: dict, **paths) -> dict:
    """Deep-copy ``base`` and apply dotted-path overrides, e.g. ``with_overrides(cfg,
    **{"decoder.type": "upernet", "data.canonical_nm": 1.0, "name": "..."})``."""
    d = copy.deepcopy(base)
    for dotted, val in paths.items():
        node = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return d


def write_config(cfg: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path
