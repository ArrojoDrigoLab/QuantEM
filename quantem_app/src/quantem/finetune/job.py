"""The guided fine-tuning job: crops in, a calibrated adapter out.

Two rungs, and the cheap one is not a consolation prize:

``threshold_only``
    Sweeps 19 thresholds over probability maps the app has already computed and
    keeps the Dice-maximising one. numpy only — no torch, no GPU, seconds. This
    is the rung the manuscript names, and it is the one that must work on every
    machine QuantEM installs on, including a laptop with no CUDA and no torch.

``head``
    Everything above, plus 300 steps of neck + decoder training first
    (:mod:`quantem.finetune.adapt`). Needs torch and a model whose head can be
    separated from its encoder.

Both fit on the user's *training* crops and only ever *score* on the held-out
ones, and both report the split mode, the crops the threshold was fit on, and
the per-crop oracle ceiling alongside every number. Those are requirements from
``API_CONTRACT.md`` §"Honesty rules", not decorations: a held-out Dice with no
split mode beside it is a number someone will put in a paper.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from quantem.finetune import calibrate
from quantem.finetune.adapt import (
    AdaptConfig,
    AdaptProgress,
    build_patches,
    load_head,
    masks_to_model_scale,
    save_head,
    tile_for,
    to_model_scale_crop,
    torch_available,
    train_head,
)
from quantem.finetune.storage import adapter_head_path, relative_head_path
from quantem.inference import resample
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.services.adapt import (
    AnnotatedCrop,
    CompletedRoiRequired,
    plan_split,
    require_crops,
)

logger = logging.getLogger(__name__)

MODE_THRESHOLD_ONLY = "threshold_only"
MODE_HEAD = "head"
MODES = (MODE_THRESHOLD_ONLY, MODE_HEAD)


# ---------------------------------------------------------------------------
# Adapter record (optional)
# ---------------------------------------------------------------------------


def _adapter_model() -> Any:
    """The ``Adapter`` model, or None when ``quantem.finetune`` is not installed.

    The job's whole result is returned to the caller and stored on the job row,
    so it stays useful either way; this only decides whether it is *also*
    recorded as an adapter. Import is guarded rather than assumed because
    importing a model of an app missing from ``INSTALLED_APPS`` raises.
    """
    try:
        from quantem.finetune.models import Adapter

        return Adapter
    except Exception:  # pragma: no cover - only before settings is updated
        logger.debug("quantem.finetune is not an installed app", exc_info=True)
        return None


def _update_adapter(adapter_id: str | None, **fields: object) -> None:
    if not adapter_id:
        return
    model = _adapter_model()
    if model is None:
        return
    model.objects.filter(id=adapter_id).update(**fields)


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _resolve_mode(payload: dict) -> str:
    mode = str(payload.get("mode") or MODE_THRESHOLD_ONLY).strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)} (got {mode!r})")
    if mode == MODE_HEAD and not torch_available():
        raise ValueError(
            "Head training needs PyTorch, which is not installed here. "
            "Threshold calibration works without it."
        )
    return mode


def _segmentation(payload: dict) -> ImageSegmentation:
    segmentation_id = str(payload.get("segmentation_id") or "").strip()
    if not segmentation_id:
        raise ValueError("payload.segmentation_id is required")
    segmentation = (
        ImageSegmentation.objects.select_related("asset", "segmentation_type")
        .filter(id=segmentation_id)
        .first()
    )
    if segmentation is None:
        raise ValueError(f"Segmentation {segmentation_id} not found")
    return segmentation


# ---------------------------------------------------------------------------
# Crops -> calibration crops
# ---------------------------------------------------------------------------


def _calibration_crops(
    crops: Sequence[AnnotatedCrop],
) -> list[calibrate.Crop]:
    return [
        calibrate.Crop(name=c.name, prob=c.prob, gt=c.gt, valid=c.valid)
        for c in crops
        if c.prob is not None
    ]


def _split_for_scoring(
    crops: Sequence[AnnotatedCrop],
) -> tuple[list[AnnotatedCrop], list[AnnotatedCrop], str]:
    train, heldout, mode = plan_split(list(crops))
    if not train:
        raise ValueError(
            "No annotated region has a probability map to calibrate against."
        )
    return train, heldout, mode


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


def adapter_job(payload: dict, reporter: Any, cancel: Any) -> dict:
    """Fit an adapter to the user's annotations and report it honestly.

    Args:
        payload: ``segmentation_id``, ``base_model``, ``mode``, and optionally
            ``adapter_id``, ``steps``, ``lr``, ``seed``, ``name``.
        reporter: job reporter (``update(progress=, message=)``, ``log``).
        cancel: cancel token (``check_cancelled()``).

    Returns:
        The adapter payload described in ``API_CONTRACT.md`` §"Guided
        fine-tuning", which is also what ``GET /api/adapters/<id>/`` serves.
    """
    adapter_id = str(payload.get("adapter_id") or "").strip() or None
    try:
        result = _run(payload, reporter, cancel, adapter_id)
    except Exception as exc:
        _update_adapter(adapter_id, status="FAILED", error=str(exc))
        raise
    return result


def _run(payload: dict, reporter: Any, cancel: Any, adapter_id: str | None) -> dict:
    cancel.check_cancelled()
    segmentation = _segmentation(payload)
    base_model = str(payload.get("base_model") or "").strip()
    if not base_model:
        raise ValueError("payload.base_model is required")
    mode = _resolve_mode(payload)
    config = AdaptConfig(
        steps=_as_int(payload.get("steps"), AdaptConfig.steps),
        lr=_as_float(payload.get("lr"), AdaptConfig.lr),
        seed=_as_int(payload.get("seed"), AdaptConfig.seed),
    )

    _update_adapter(adapter_id, status="RUNNING", error="")
    reporter.update(progress=5.0, message="reading your annotated regions")

    try:
        crop_set = require_crops(
            segmentation,
            load_em=(mode == MODE_HEAD),
            load_prob=(mode == MODE_THRESHOLD_ONLY),
            # Head training predicts its own maps; only calibration is stuck
            # without one already on disk.
            require_probability=(mode == MODE_THRESHOLD_ONLY),
        )
    except CompletedRoiRequired as exc:
        # Not an internal error: it is the one hard precondition, and the user
        # can fix it in the viewer in a few seconds.
        reporter.log("warning", str(exc))
        raise
    for warning in crop_set.warnings:
        reporter.log("warning", warning)

    cancel.check_cancelled()
    if mode == MODE_HEAD:
        result = _run_head(
            reporter,
            cancel,
            crop_set=crop_set,
            base_model=base_model,
            config=config,
            adapter_id=adapter_id,
        )
    else:
        result = _run_threshold_only(
            reporter,
            crop_set=crop_set,
            base_model=base_model,
        )

    result["id"] = adapter_id
    result["name"] = str(payload.get("name") or "").strip()
    result["segmentation_id"] = str(segmentation.id)
    result["warnings"] = list(crop_set.warnings)
    result["status"] = "SUCCESS"

    _update_adapter(
        adapter_id,
        status="SUCCESS",
        sweep=result["sweep"],
        calibrated_threshold=result["sweep"]["calibrated_threshold"],
        split_mode=result["split_mode"],
        head_path=result.get("head_path") or "",
        verified_reload=bool(result.get("verified_reload")),
        trainable_params=result.get("trainable_params"),
        train_seconds=result.get("train_seconds"),
    )
    reporter.update(progress=100.0, message="adaptation complete")
    return result


def _run_threshold_only(
    reporter: Any, *, crop_set: Any, base_model: str
) -> dict:
    """The always-available rung: one scalar, fit on the training crops."""
    usable = [c for c in crop_set.crops if c.prob is not None]
    if not usable:
        raise ValueError(
            "No probability map covers the completed area. Run the model on this "
            "image first."
        )
    train, heldout, split_mode = _split_for_scoring(usable)

    reporter.update(progress=45.0, message="sweeping the threshold")
    sweep = calibrate.sweep_threshold(
        _calibration_crops(train), heldout_crops=_calibration_crops(heldout)
    )
    reporter.update(progress=90.0, message="threshold calibrated")

    return {
        "base_model": base_model,
        "mode": MODE_THRESHOLD_ONLY,
        "steps": 0,
        "trainable_params": 0,
        "split_mode": split_mode,
        "train_crop_names": [c.name for c in train],
        "heldout_crop_names": [c.name for c in heldout],
        "sweep": sweep.as_dict(),
        "verified_reload": False,
        "caveats": _caveats(split_mode, sweep, mode=MODE_THRESHOLD_ONLY, verified=False),
    }


def _run_head(
    reporter: Any,
    cancel: Any,
    *,
    crop_set: Any,
    base_model: str,
    config: AdaptConfig,
    adapter_id: str | None,
) -> dict:
    """Train the neck + decoder, then calibrate on top of the adapted model.

    The base model is scored first, with its own freshly computed probability
    maps rather than whatever is cached on disk, so "base 0.817 -> adapted 0.870"
    compares two runs of the same code over the same pixels.
    """
    from quantem.inference import engine  # noqa: PLC0415 -- torch-heavy, lazy

    usable = [c for c in crop_set.crops if c.em is not None]
    if not usable:
        raise ValueError("No annotated region could be read from its image.")
    train, heldout, split_mode = _split_for_scoring(usable)

    reporter.update(progress=8.0, message=f"loading {base_model}")
    model = engine.load_model(base_model)
    # Training mutates this module in place. Dropping it from the shared cache
    # first means a later segmentation run cannot silently pick up an adapted
    # head the user never applied.
    engine.clear_model_cache()
    spec = model.spec

    # Everything past this point happens on the grid the model predicts on.
    scaled = {
        c.name: to_model_scale_crop(c, canonical_nm=spec.canonical_nm) for c in usable
    }

    cancel.check_cancelled()
    reporter.update(progress=12.0, message="scoring the base model")
    base_probs = {c.name: _predict(engine, model, c) for c in usable}
    base_sweep = calibrate.sweep_threshold(
        _scoring_crops(train, base_probs, scaled),
        heldout_crops=_scoring_crops(heldout, base_probs, scaled),
    )

    cancel.check_cancelled()
    tile = tile_for(spec.tile_size, spec.patch_size)
    patches = build_patches(
        [scaled[c.name] for c in train],
        tile,
        image_mean=spec.image_mean,
        image_std=spec.image_std,
        config=config,
    )
    reporter.update(
        progress=18.0,
        message=f"training on {len(patches)} window(s) from {len(train)} region(s)",
    )

    def on_progress(progress: AdaptProgress) -> None:
        reporter.update(
            progress=18.0 + 52.0 * progress.fraction,
            message=(
                f"step {progress.step + 1}/{progress.total_steps} "
                f"loss {progress.loss:.3f} ETA ~{progress.eta_s:.0f}s"
            ),
        )

    training = train_head(
        model.module,
        patches,
        device=model.device,
        config=config,
        on_progress=on_progress,
        should_cancel=lambda: _cancelled(cancel),
    )

    cancel.check_cancelled()
    reporter.update(progress=72.0, message="scoring the adapted model")
    adapted_probs = {c.name: _predict(engine, model, c) for c in usable}
    sweep = calibrate.sweep_threshold(
        _scoring_crops(train, adapted_probs, scaled),
        heldout_crops=_scoring_crops(heldout, adapted_probs, scaled),
    )

    reporter.update(progress=88.0, message="saving the adapted head")
    head_file = adapter_head_path(adapter_id or "unsaved")
    save_head(
        model.module,
        head_file,
        meta={
            "base_model": base_model,
            "steps": training.steps,
            "tile": tile,
            "calibrated_threshold": sweep.calibrated_threshold,
            "split_mode": split_mode,
        },
    )

    # The reference's last act, kept: reload the saved head onto a fresh encoder
    # and confirm it reproduces the held-out Dice. An adapter that cannot
    # reproduce its own number is not a deliverable.
    verified = False
    reload_heldout = None
    if heldout:
        reporter.update(progress=93.0, message="verifying the saved head")
        fresh = engine.load_model(base_model)
        engine.clear_model_cache()
        load_head(fresh.module, head_file)
        reload_probs = {c.name: _predict(engine, fresh, c) for c in heldout}
        reload_heldout = calibrate.mean_dice(
            _scoring_crops(heldout, reload_probs, scaled), sweep.calibrated_threshold
        )
        verified = _close(reload_heldout, sweep.heldout_dice_at_calibrated)
        if not verified:
            reporter.log(
                "warning",
                "The saved head scored "
                f"{reload_heldout} on reload against {sweep.heldout_dice_at_calibrated} "
                "during training; the numbers below are from the in-memory model.",
            )

    result = {
        "base_model": base_model,
        "mode": MODE_HEAD,
        "steps": training.steps,
        "trainable_params": training.trainable_params,
        "train_seconds": round(training.seconds, 2),
        "split_mode": split_mode,
        "train_crop_names": [c.name for c in train],
        "heldout_crop_names": [c.name for c in heldout],
        "sweep": sweep.as_dict(),
        "base_sweep": base_sweep.as_dict(),
        "training": training.as_dict(),
        "tile": tile,
        "head_path": relative_head_path(head_file),
        "verified_reload": verified,
        "reloaded_heldout_dice": reload_heldout,
        "caveats": _caveats(split_mode, sweep, mode=MODE_HEAD, verified=verified),
    }
    return result


def _cancelled(cancel: Any) -> bool:
    try:
        cancel.check_cancelled()
    except Exception:
        return True
    return False


def _predict(engine: Any, model: Any, crop: AnnotatedCrop) -> np.ndarray:
    """Probability for one crop, at model scale."""
    prediction = engine.predict_region(
        model, crop.em, pixel_size_nm=crop.pixel_size_nm
    )
    return prediction.prob


def _scoring_crops(
    crops: Sequence[AnnotatedCrop],
    probs: dict[str, np.ndarray],
    scaled: dict[str, Any],
) -> list[calibrate.Crop]:
    """Pair each crop's model-scale labels with its freshly predicted map.

    ``predict_region`` returns the map on the grid the model actually predicted
    on, which for a resampled model is not the crop's native grid. The labels
    move to the prediction, never the other way round — thresholding an
    upsampled probability map re-decides the boundary on interpolated values the
    model never produced.
    """
    scoring: list[calibrate.Crop] = []
    for crop in crops:
        prob = probs[crop.name]
        labels = scaled[crop.name]
        gt, valid = labels.gt, labels.valid
        if prob.shape != gt.shape:
            # Defensive: the two paths plan the same resample, so this should
            # not fire. Rounding is the only way it could.
            context = resample.ResampleContext(
                factor=prob.shape[0] / max(1, gt.shape[0]),
                native_shape=(int(gt.shape[0]), int(gt.shape[1])),
                model_shape=(int(prob.shape[0]), int(prob.shape[1])),
            )
            gt, valid = masks_to_model_scale(gt, valid, context)
        scoring.append(calibrate.Crop(name=crop.name, prob=prob, gt=gt, valid=valid))
    return scoring


def _close(a: float | None, b: float | None, tol: float = 1e-4) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _caveats(
    split_mode: str,
    sweep: calibrate.SweepResult,
    *,
    mode: str,
    verified: bool,
) -> list[str]:
    """The sentences that must travel with these numbers."""
    notes = [
        "The threshold was fit on the training crops only; the held-out score is "
        "reported at that threshold, never used to choose it."
    ]
    if split_mode == "within-image":
        notes.append(
            "The held-out crops come from the same image as the training crops, "
            "so this is a within-image score and does not measure generalisation "
            "to a new image."
        )
    elif split_mode == "no-heldout":
        notes.append(
            "Every annotated region was used to fit the threshold, so there is no "
            "held-out score at all."
        )
    if sweep.heldout_oracle is not None:
        notes.append(
            "The oracle is the best achievable with a threshold chosen per crop "
            "using the answers. It is a ceiling, not a target."
        )
    if mode == MODE_HEAD and not verified:
        notes.append(
            "The saved head was not re-scored after reloading, so these numbers "
            "are from the in-memory model only."
        )
    return notes


#: The name ``quantem.jobs.handlers`` imports. Kept as an alias so the job entry
#: point and the module's own vocabulary can differ without a shim in between.
train_organelle_adapter_job = adapter_job
