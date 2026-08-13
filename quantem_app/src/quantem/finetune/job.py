"""The guided fine-tuning job: crops in, a calibrated adapter out.

Two rungs, and the cheap one is not a consolation prize:

``threshold_only``
    Sweeps 19 thresholds over probability maps the app has already computed and
    keeps the Dice-maximising one. numpy only — no torch, no GPU, seconds. This
    is the rung the manuscript names, and it is the one that must work on every
    machine QuantEM installs on, including a laptop with no CUDA and no torch.

``head``
    Everything above, plus neck + decoder training first
    (:mod:`quantem.finetune.adapt`). Each round uses 20 steps per training tile,
    clamped to 300--600 steps. Needs torch and a model whose head can be
    separated from its encoder.

Both fit on the user's *training* crops and only ever *score* on the held-out
ones, and both report the split mode, the crops the threshold was fit on, and
the per-crop oracle ceiling alongside every number. Those are requirements from
the API contract's Honesty section rules", not decorations: a held-out Dice with no
split mode beside it is a number someone will put in a paper.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from quantem.finetune import calibrate
from quantem.finetune.adapt import (
    AdaptConfig,
    AdaptProgress,
    HeadAdaptationUnavailable,
    build_patches,
    load_head,
    masks_to_model_scale,
    save_head,
    tile_for,
    to_model_scale_crop,
    torch_available,
    train_head,
)
from quantem.finetune.preflight import check_head_size
from quantem.finetune.scope import TrainingFold, count_tiles, plan_folds, plan_step_counts
from quantem.finetune.storage import (
    adapter_head_path,
    discard_staged_head,
    promote_head,
    relative_head_path,
    staged_head_path,
    unsaved_head_path,
)
from quantem.inference import resample
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.services.adapt import (
    AnnotatedCrop,
    CompletedRoiRequired,
    plan_split,
    require_crops,
    require_crops_for_scope,
)

logger = logging.getLogger(__name__)

MODE_THRESHOLD_ONLY = "threshold_only"
MODE_HEAD = "head"
MODES = (MODE_THRESHOLD_ONLY, MODE_HEAD)

#: Said when an overwrite fails. The point of the sentence is that nothing was
#: lost: the previous weights were never touched, because a new head is written
#: beside the live one and only moved over it once the run has finished.
OVERWRITE_SAFE_SUFFIX = " The previous version of this fine-tune is untouched and still in use."


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
        raise ValueError("No annotated region has a probability map to calibrate against.")
    return train, heldout, mode


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


def adapter_job(payload: dict, reporter: Any, cancel: Any) -> dict:
    """Fit an adapter to the user's annotations and report it honestly.

    Two shapes of payload, one entry point:

    * the **single-segmentation** one, still used by the labeling view's Improve
      panel: ``segmentation_id``, ``base_model``, ``mode``;
    * the **scoped** one, from the Fine-Tune dialog: ``segmentation_type_id``,
      ``asset_ids``, ``training_mode``, ``cv_benchmark``. Recognised by
      ``asset_ids`` being present, so an old queued row still runs the old way.

    Args:
        payload: as above, plus optionally ``adapter_id``, ``steps``, ``lr``,
            ``seed``, ``name``.
        reporter: job reporter (``update(progress=, message=)``, ``log``).
        cancel: cancel token (``check_cancelled()``).

    Returns:
        The adapter payload described in the API contract's Guided section
        fine-tuning", which is also what ``GET /api/adapters/<id>/`` serves.
    """
    adapter_id = str(payload.get("adapter_id") or "").strip() or None
    scoped = bool(payload.get("asset_ids")) and bool(payload.get("segmentation_type_id"))
    try:
        if scoped:
            result = _run_scoped(payload, reporter, cancel, adapter_id)
        else:
            result = _run(payload, reporter, cancel, adapter_id)
    except Exception as exc:
        message = str(exc)
        if payload.get("preserves_live_version"):
            message += OVERWRITE_SAFE_SUFFIX
        _update_adapter(adapter_id, status="FAILED", error=message)
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
        base_model=base_model,
        preserves_live_version=False,
        sweep=result["sweep"],
        calibrated_threshold=result["sweep"]["calibrated_threshold"],
        split_mode=result["split_mode"],
        head_path=result.get("head_path") or "",
        verified_reload=bool(result.get("verified_reload")),
        trainable_params=result.get("trainable_params"),
        train_seconds=result.get("train_seconds"),
    )
    result["apply_and_rerun"] = _apply_and_rerun(
        adapter_id,
        base_model=base_model,
        calibrated=result["sweep"]["calibrated_threshold"],
        requested=bool(payload.get("apply_and_rerun")),
        reporter=reporter,
    )
    reporter.update(progress=100.0, message="adaptation complete")
    return result


def _apply_and_rerun(
    adapter_id: str | None,
    *,
    base_model: str,
    calibrated: float | None,
    requested: bool,
    reporter: Any,
) -> dict:
    """Put the new include level to work, and say exactly what that did.

    Two separate things, deliberately not merged:

    * **Applying** stamps the adapter so the *next* run uses it. It writes no
      object, so nothing on screen moves and nothing the user did by hand is
      touched. It is instant and it is what ``requested`` asks for.
    * **Re-running** is what actually re-finds the objects, costs real time, and
      belongs to the run machinery — not to a job that told the user it would
      take about a second. So this reports that the re-run is pending; it does
      not queue one behind the user's back.

    The re-run itself is safe by construction:
    :func:`quantem.seg_core.db.extraction.extract_and_save_segments` deletes only
    its own generated candidates and then drops any new guess that lands on a
    kept or removed object. Saying so here, in the result the panel renders, is
    what makes the guarantee visible rather than merely true.
    """
    default = _published_threshold(base_model)
    changes = (
        calibrated is not None
        and default is not None
        and abs(float(calibrated) - float(default)) >= 5e-3
    )
    applied_at = None
    if requested and adapter_id:
        from django.utils import timezone  # noqa: PLC0415 -- Django-only path

        applied_at = timezone.now()
        _update_adapter(adapter_id, applied_at=applied_at)
        reporter.update(message="using the new include level for the next run")
    return {
        "requested": requested,
        "applied": applied_at is not None,
        "applied_at": applied_at.isoformat() if applied_at is not None else None,
        "include_level": calibrated,
        "previous_include_level": default,
        "changes_objects": bool(changes),
        "rerun_pending": bool(applied_at is not None and changes),
        "preserves_manual_work": True,
        "preservation": (
            "Nothing you have kept, removed or drawn by hand changes when the "
            "model runs again. Only my own guesses are replaced."
        ),
    }


def _published_threshold(base_model: str) -> float | None:
    """The pack's own cut-off, so a new one can be reported as a change."""
    from quantem.inference.specs import MODEL_SPECS  # noqa: PLC0415 -- cheap, local

    spec = MODEL_SPECS.get(base_model)
    return float(spec.threshold) if spec is not None else None


def _run_threshold_only(reporter: Any, *, crop_set: Any, base_model: str) -> dict:
    """The always-available rung: one scalar, fit on the training crops."""
    usable = [c for c in crop_set.crops if c.prob is not None]
    if not usable:
        raise ValueError(
            "No probability map covers the completed area. Run the model on this image first."
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
    scaled = {c.name: to_model_scale_crop(c, canonical_nm=spec.canonical_nm) for c in usable}

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
    if not patches:
        # ``train_head`` refuses this too, but in terms of its own coverage
        # rule -- "every completed area is smaller than the model's 20 %
        # coverage rule for one tile" is true and tells a microscopist nothing
        # they can act on. The pre-flight knows the two spans, so say them.
        # Reaching here at all means the geometry changed between the check at
        # the door and the run; both ends now give the same sentence.
        verdict = check_head_size(train, base_model)
        raise HeadAdaptationUnavailable(
            f"{verdict.reason} No training window survived, so there would be "
            "nothing to train on. Matching my cut-off to your marks works at "
            "any size."
            if verdict is not None and verdict.reason
            else "No training window survived: every checked area is too small "
            "for this model. Matching my cut-off to your marks works at any size."
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
    # Written beside the live file and moved over it at the end, never straight
    # onto it: a run that dies while saving must leave whatever was there
    # loadable. An unrecorded run gets a unique scratch path -- this used to be
    # a single shared ``unsaved`` folder, so two of them overwrote each other.
    head_file = staged_head_path(adapter_id) if adapter_id else unsaved_head_path()
    final_head = adapter_head_path(adapter_id) if adapter_id else head_file
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

    promote_head(head_file, final_head)

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
        "head_path": relative_head_path(final_head),
        "verified_reload": verified,
        "reloaded_heldout_dice": reload_heldout,
        "caveats": _caveats(split_mode, sweep, mode=MODE_HEAD, verified=verified),
    }
    return result


# ---------------------------------------------------------------------------
# The scoped run: a named fine-tune over a chosen set of images
# ---------------------------------------------------------------------------


def _null_scope(total: int, label: str):
    """A unit-progress scope for a caller that has no job row behind it."""
    from quantem.jobs.reporter import NullUnitProgressScope  # noqa: PLC0415

    return NullUnitProgressScope(total=total, label=label)


def _unit_scope(reporter: Any, *, total: int, label: str, stage: str, detail: dict):
    """``reporter.unit_scope`` when there is one, a no-op scope otherwise.

    The trainer runs under the queue in production and under a plain object in
    tests and at a REPL; giving the second case a scope with the same surface is
    what keeps the round loop free of "is anything watching" branches.
    """
    factory = getattr(reporter, "unit_scope", None)
    if factory is None:
        return _null_scope(total, label)
    return factory(total=total, label=label, stage=stage, detail=detail)


def _stage(reporter: Any, stage: str, message: str | None = None) -> None:
    update = getattr(reporter, "update", None)
    if update is None:
        return
    try:
        update(message=message, stage=stage)
    except TypeError:
        # A reporter from before stages existed. The message still lands.
        update(message=message)


def _fold_metrics(
    fold: TrainingFold,
    probs: dict[str, Any],
    scaled: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Dice and IoU for one round, on the area held back from it.

    ``None`` where undefined -- neither prediction nor truth has any foreground
    inside the valid region -- and never zero, which would drag a mean down for
    an area that contains nothing to find.
    """
    scored = _scoring_crops(fold.heldout, probs, scaled)
    return {
        "fold": fold.index,
        "held_out_asset_id": fold.held_out_asset_id,
        "dice": calibrate.mean_dice(scored, threshold),
        "iou": calibrate.mean_iou(scored, threshold),
        "n_tiles": len(fold.train),
    }


def _cv_results(folds: list[dict[str, Any]], names: dict[str, str]) -> dict[str, Any]:
    """Fold rows, the mean over them, and the per-image breakdown R13 requires.

    The mean is over the folds that produced a number. Per-image rows are
    grouped by the image that was held out, so a run whose average looks fine
    but which cannot do one of the images says so on that image's row -- which
    is the whole reason the owner asked for per-image results beside the mean.
    """
    dice = [f["dice"] for f in folds if f["dice"] is not None]
    iou = [f["iou"] for f in folds if f["iou"] is not None]
    per_image: dict[str, dict[str, Any]] = {}
    for fold in folds:
        asset_id = fold.get("held_out_asset_id")
        if not asset_id:
            continue
        row = per_image.setdefault(
            str(asset_id),
            {
                "asset_id": str(asset_id),
                "name": names.get(str(asset_id), ""),
                "dice": None,
                "iou": None,
                "n_tiles": 0,
            },
        )
        row["dice"] = fold["dice"]
        row["iou"] = fold["iou"]
        row["n_tiles"] = fold["n_tiles"]
    return {
        "folds": folds,
        "mean": {
            "dice": float(np.mean(dice)) if dice else None,
            "iou": float(np.mean(iou)) if iou else None,
        },
        "per_image": [per_image[key] for key in sorted(per_image)],
    }


def _asset_names(asset_ids: Sequence[str]) -> dict[str, str]:
    from quantem.assets.models import Asset  # noqa: PLC0415 -- Django app registry

    return {
        str(asset_id): display_name
        for asset_id, display_name in Asset.objects.filter(
            id__in=[str(value) for value in asset_ids]
        ).values_list("id", "display_name")
    }


def _run_scoped(payload: dict, reporter: Any, cancel: Any, adapter_id: str | None) -> dict:
    """Train a named fine-tune over an explicit set of images.

    One round per hold-out unit under cross-validation, one otherwise. Every
    round starts from the released weights, not from the round before it: a fold
    that inherited the previous fold's training would have already seen the area
    it is about to be scored on, and its number would be worthless.

    The **shipped** head is the last round's. Under cross-validation the folds
    are a measurement of the recipe on this data, and the average is reported as
    that; the head the user gets is one real model, and the fold it belongs to
    is named in the result so the two are never confused.
    """
    from quantem.finetune.models import (  # noqa: PLC0415 -- Django app registry
        TRAINING_MODE_USE_ALL,
    )
    from quantem.inference import engine  # noqa: PLC0415 -- torch-heavy, lazy
    from quantem.jobs.models import (  # noqa: PLC0415 -- Django app registry
        STAGE_EVALUATING,
        STAGE_LOADING_MODEL,
        STAGE_PREPARING,
        STAGE_SAVING,
        STAGE_TRAINING,
        UNIT_STEP,
    )
    from quantem.jobs.reporter import unit_window  # noqa: PLC0415

    cancel.check_cancelled()
    base_model = str(payload.get("base_model") or "").strip()
    if not base_model:
        raise ValueError("payload.base_model is required")
    if not torch_available():
        raise ValueError(
            "Fine-tuning needs PyTorch, which is not installed here. Matching my "
            "cut-off to your marks works without it."
        )
    segmentation_type_id = str(payload.get("segmentation_type_id") or "").strip()
    asset_ids = [str(value) for value in (payload.get("asset_ids") or [])]
    training_mode = str(payload.get("training_mode") or TRAINING_MODE_USE_ALL)
    cv_benchmark = bool(payload.get("cv_benchmark"))
    config = AdaptConfig(
        lr=_as_float(payload.get("lr"), AdaptConfig.lr),
        seed=_as_int(payload.get("seed"), AdaptConfig.seed),
    )

    _update_adapter(adapter_id, status="RUNNING", error="")
    _stage(reporter, STAGE_PREPARING, "reading the annotations you chose")

    try:
        crop_set = require_crops_for_scope(segmentation_type_id, asset_ids, load_em=True)
    except CompletedRoiRequired as exc:
        reporter.log("warning", str(exc))
        raise
    for warning in crop_set.warnings:
        reporter.log("warning", warning)

    usable = [c for c in crop_set.crops if c.em is not None]
    if not usable:
        raise ValueError("No annotated region could be read from its image.")

    folds, split_mode = plan_folds(usable, training_mode=training_mode, cv_benchmark=cv_benchmark)
    total_rounds = len(folds)
    tile_count = count_tiles(usable, base_model, config=config)
    fixed_value = payload.get("fixed_steps")
    if fixed_value is None and "planned_steps" not in payload:
        # Backward compatibility for older and internal scoped payloads, whose
        # ``steps`` field meant a fixed count for every round.
        fixed_value = payload.get("steps")
    fixed_steps = _as_int(fixed_value, AdaptConfig.steps) if fixed_value is not None else None
    steps_by_round = plan_step_counts(
        folds,
        base_model,
        fixed_steps=fixed_steps,
        config=config,
    )
    training_tiles_by_round = [count_tiles(fold.train, base_model, config=config) for fold in folds]
    total_steps = sum(steps_by_round)

    cancel.check_cancelled()
    _stage(reporter, STAGE_LOADING_MODEL, f"loading {base_model}")
    model = engine.load_model(base_model)
    engine.clear_model_cache()
    spec = model.spec
    scaled = {c.name: to_model_scale_crop(c, canonical_nm=spec.canonical_nm) for c in usable}
    tile = tile_for(spec.tile_size, spec.patch_size)

    # The "before" number, from this code over these pixels, so the improvement
    # is a comparison of two runs and not of a run against a published figure.
    base_probs = {c.name: _predict(engine, model, c) for c in usable}
    base_sweep = calibrate.sweep_threshold(_scoring_crops(usable, base_probs, scaled))

    fold_rows: list[dict[str, Any]] = []
    training = None
    sweep = None
    shipped_fold = folds[-1]
    completed_steps = 0
    for index, fold in enumerate(folds):
        cancel.check_cancelled()
        if index:
            # Fresh weights per round. See the docstring.
            model = engine.load_model(base_model)
            engine.clear_model_cache()

        round_config = replace(config, steps=steps_by_round[index])
        patches = build_patches(
            [scaled[c.name] for c in fold.train],
            tile,
            image_mean=spec.image_mean,
            image_std=spec.image_std,
            config=round_config,
        )
        if not patches:
            verdict = check_head_size(fold.train, base_model)
            raise HeadAdaptationUnavailable(
                f"{verdict.reason} No training window survived, so there would be "
                "nothing to train on. Matching my cut-off to your marks works at "
                "any size."
                if verdict is not None and verdict.reason
                else "No training window survived: every checked area is too small "
                "for this model. Matching my cut-off to your marks works at any "
                "size."
            )

        base_units = completed_steps
        message = (
            f"Round {index + 1} of {total_rounds}: training on {len(patches)} "
            f"window(s) from {len(fold.train)} area(s) for {round_config.steps} steps"
        )
        reporter.update(message=message)
        # One window per round, offset into the grand total, so the bar and the
        # ETA are over the whole run rather than restarting every fold.
        with (
            unit_window(base_units, total_steps),
            _unit_scope(
                reporter,
                total=round_config.steps,
                label=UNIT_STEP,
                stage=STAGE_TRAINING,
                detail={
                    "round": index + 1,
                    "total_rounds": total_rounds,
                    "tiles": len(patches),
                },
            ) as steps_done,
        ):

            def on_progress(progress: AdaptProgress, sink=steps_done) -> None:
                sink.set(progress.step + 1)

            training = train_head(
                model.module,
                patches,
                device=model.device,
                config=round_config,
                on_progress=on_progress,
                should_cancel=lambda: _cancelled(cancel),
            )

        completed_steps += training.steps

        cancel.check_cancelled()
        _stage(
            reporter,
            STAGE_EVALUATING,
            f"Round {index + 1} of {total_rounds}: scoring the held-out area",
        )
        adapted_probs = {c.name: _predict(engine, model, c) for c in usable}
        sweep = calibrate.sweep_threshold(
            _scoring_crops(fold.train, adapted_probs, scaled),
            heldout_crops=_scoring_crops(fold.heldout, adapted_probs, scaled),
        )
        if fold.heldout:
            fold_rows.append(_fold_metrics(fold, adapted_probs, scaled, sweep.calibrated_threshold))
        shipped_fold = fold

    assert sweep is not None and training is not None  # one round always runs

    _stage(reporter, STAGE_SAVING, "saving the fine-tuned model")
    staged = staged_head_path(adapter_id) if adapter_id else unsaved_head_path()
    final_head = adapter_head_path(adapter_id) if adapter_id else staged
    try:
        save_head(
            model.module,
            staged,
            meta={
                "base_model": base_model,
                "steps": training.steps,
                "steps_by_round": steps_by_round,
                "total_steps": total_steps,
                "tile": tile,
                "calibrated_threshold": sweep.calibrated_threshold,
                "split_mode": split_mode,
            },
        )
        promote_head(staged, final_head)
    except Exception:
        discard_staged_head(staged)
        raise

    names = _asset_names(asset_ids)
    cv_results = _cv_results(fold_rows, names) if fold_rows else {}

    result = {
        "id": adapter_id,
        "name": str(payload.get("name") or "").strip(),
        "base_model": base_model,
        "mode": MODE_HEAD,
        "training_mode": training_mode,
        "cv_benchmark": cv_benchmark,
        "steps": training.steps,
        "steps_by_round": steps_by_round,
        "total_steps": total_steps,
        "step_policy": "fixed" if fixed_steps is not None else "20_per_tile_clamped_300_600",
        "rounds": total_rounds,
        "trainable_params": training.trainable_params,
        "train_seconds": round(training.seconds, 2),
        "split_mode": split_mode,
        "asset_ids": asset_ids,
        "annotation_count": crop_set.annotation_count,
        "tile_count": tile_count,
        "training_tiles_by_round": training_tiles_by_round,
        "train_crop_names": [c.name for c in shipped_fold.train],
        "heldout_crop_names": [c.name for c in shipped_fold.heldout],
        "sweep": sweep.as_dict(),
        "base_sweep": base_sweep.as_dict(),
        "training": training.as_dict(),
        "tile": tile,
        "head_path": relative_head_path(final_head),
        "verified_reload": False,
        "cv_results": cv_results,
        "warnings": list(crop_set.warnings),
        "status": "SUCCESS",
        "caveats": _caveats(split_mode, sweep, mode=MODE_HEAD, verified=False),
    }

    _update_adapter(
        adapter_id,
        status="SUCCESS",
        base_model=base_model,
        preserves_live_version=False,
        sweep=result["sweep"],
        calibrated_threshold=result["sweep"]["calibrated_threshold"],
        split_mode=split_mode,
        head_path=result["head_path"],
        verified_reload=False,
        trainable_params=result["trainable_params"],
        train_seconds=result["train_seconds"],
        cv_results=cv_results,
        params={
            "steps": training.steps,
            "steps_by_round": steps_by_round,
            "total_steps": total_steps,
            "step_policy": result["step_policy"],
            "lr": config.lr,
            "seed": config.seed,
        },
        # Written by the run and not only by the view that started it, so the row
        # records what was actually done rather than what was asked for.
        training_mode=training_mode,
        cv_benchmark=cv_benchmark,
    )
    reporter.update(progress=100.0, message="Fine-tune complete")
    return result


def _cancelled(cancel: Any) -> bool:
    try:
        cancel.check_cancelled()
    except Exception:
        return True
    return False


def _predict(engine: Any, model: Any, crop: AnnotatedCrop) -> np.ndarray:
    """Probability for one crop, at model scale."""
    prediction = engine.predict_region(model, crop.em, pixel_size_nm=crop.pixel_size_nm)
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
