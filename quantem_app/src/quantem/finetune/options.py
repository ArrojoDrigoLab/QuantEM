"""Validated training controls and checkpoint selection inside a training fold."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from quantem.finetune.adapt import (
    HEAD_MODULES,
    MAX_ADAPT_STEPS,
    AdaptConfig,
    build_patches,
    train_head,
)


def parse_options(data: dict) -> AdaptConfig:
    """Validate controls before enqueueing a potentially long GPU job."""

    def integer(name, default, low, high):
        value = data.get(name, default)
        try:
            number = int(value)
            if isinstance(value, bool) or float(value) != number or not low <= number <= high:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a whole number from {low} to {high}.") from exc
        return number

    def real(name, default, low, high):
        try:
            number = float(data.get(name, default))
            if not math.isfinite(number) or not low <= number <= high:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be from {low} to {high}.") from exc
        return number

    schedule = data.get("schedule", "quick")
    if schedule not in {"quick", "extended"}:
        raise ValueError("Choose a quick or extended training schedule.")
    return AdaptConfig(
        steps=integer("steps", AdaptConfig.steps, 1, MAX_ADAPT_STEPS),
        lr=real("lr", AdaptConfig.lr, 1e-7, 0.01),
        weight_decay=real("weight_decay", AdaptConfig.weight_decay, 0, 1),
        seed=integer("seed", AdaptConfig.seed, 0, 2**32 - 1),
        schedule=schedule,
        validation_fraction=real("validation_fraction", 0.0, 0, 0.5),
        eval_interval=integer("eval_interval", 100, 1, MAX_ADAPT_STEPS),
        patience=integer("patience", 0, 0, 1000),
        min_delta=real("min_delta", 1e-4, 0, 1),
    )


def validation_split(crops, fraction: float, seed: int):
    """Image-disjoint inner split. Outer evaluation images never enter here."""
    keys = sorted({c.image_key for c in crops})
    if fraction <= 0 or len(keys) < 2:
        return list(crops), []
    order = np.random.default_rng(seed).permutation(keys)
    held = set(order[: min(len(keys) - 1, max(1, int(math.ceil(len(keys) * fraction))))])
    return [c for c in crops if c.image_key not in held], [c for c in crops if c.image_key in held]


def train_with_validation(
    module,
    crops,
    scaled,
    tile,
    *,
    image_mean,
    image_std,
    device,
    config,
    on_progress=None,
    should_cancel=None,
):
    """Select a step count on inner validation, then refit all training areas.

    The returned model sees every outer-training annotation. Its outer test
    images are untouched by both checkpoint selection and the refit.
    """
    fit, validation = validation_split(crops, config.validation_fraction, config.seed)

    def patches(items):
        return build_patches(
            [scaled[c.name] for c in items],
            tile,
            image_mean=image_mean,
            image_std=image_std,
            config=config,
        )

    all_patches = patches(crops)
    if not validation:
        result = train_head(
            module,
            all_patches,
            device=device,
            config=config,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        return result, {
            "validation_crop_names": [],
            "selection": None,
            "note": "Checkpoint selection needs at least two training images.",
        }

    initial = {
        name: {k: v.detach().cpu().clone() for k, v in getattr(module, name).state_dict().items()}
        for name in HEAD_MODULES
    }
    selection = train_head(
        module,
        patches(fit),
        validation_patches=patches(validation),
        device=device,
        config=config,
        on_progress=on_progress,
        should_cancel=should_cancel,
    )
    if should_cancel and should_cancel():
        return selection, {"selection": selection.as_dict(), "cancelled": True}
    for name in HEAD_MODULES:
        getattr(module, name).load_state_dict(initial[name])
    # The final pass always visits every window, including negative-only ones.
    refit_steps = max(len(all_patches), selection.selected_step)

    def progress(tick):
        if on_progress:
            on_progress(replace(tick, step=config.steps + tick.step, total_steps=2 * config.steps))

    result = train_head(
        module,
        all_patches,
        device=device,
        config=replace(config, steps=refit_steps, validation_fraction=0),
        on_progress=progress,
        should_cancel=should_cancel,
    )
    return result, {
        "validation_crop_names": [c.name for c in validation],
        "selection_train_crop_names": [c.name for c in fit],
        "selection": selection.as_dict(),
        "refit_steps": refit_steps,
        "optimizer_steps": selection.steps + result.steps,
    }
