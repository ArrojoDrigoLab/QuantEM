"""
Generic DB-Aware Inference
============================

Run organelle inference for a segmentation, loading images from DB models
and persisting probability maps. Parameterized by BaseSegmenter.

This is also where a guided fine-tuning result is put to work. An adapter that
the user applied through ``POST /api/adapters/<id>/apply/`` is looked up here
(:func:`apply_active_adapter`) and handed to the segmenter before its weights
are loaded. Until that wiring existed, a calibrated threshold and a trained head
were computed, verified, reported -- and then never used by a single run.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

import numpy as np

from quantem.assets.asset_openable import get_asset_openable
from quantem.assets.models import ImageROI
from quantem.assets.task_utils import load_image_array, load_image_roi_array
from quantem.seg_core.base_segmenter import BaseSegmenter
from quantem.seg_core.types import InferenceResult
from quantem.segmentation.models import ImageSegmentation, SegmentationConfig

from .prob_maps import (
    load_prob_map_from_path,
    prob_map_file_exists,
    save_probability_map,
)

logger = logging.getLogger(__name__)

_DETAIL_PROGRESS_DELTA = 0.05
_DETAIL_MIN_SECONDS = 1.0

#: Fallback tile edge used only for the progress estimate when a segmenter does
#: not implement ``estimate_dl_tile_count``. The real tile size is a per-model
#: fact (512 for QuantEM ViT-B, 518 for OmniEM ViT-L) owned by the segmenter.
_FALLBACK_TILE_PX = 512


def apply_active_adapter(
    segmenter: BaseSegmenter,
    segmentation: ImageSegmentation,
    on_detail: Callable[[str], None] | None = None,
) -> str | None:
    """Hand the segmentation's applied adapter to the segmenter, if there is one.

    ``quantem.finetune`` is imported **lazily**: it is an optional Django app,
    and ``seg_core`` must keep running on an install that has no guided
    fine-tuning and no torch. An install without it simply has no adapters.

    Three ways this correctly does nothing, each reported rather than silent:

    * no adapter has been applied to this segmentation;
    * the applied adapter was fitted on a different model than the one this run
      uses -- a threshold calibrated on ``quantem:mito`` describes that model's
      probability distribution and is meaningless for ``omniem:mito``;
    * the segmenter does not support adapters at all.

    Anything else -- a missing head file, an unreadable one -- is an error the
    run must not swallow. A user who applied an adapter and got an uncalibrated
    run with no complaint would have no way to know their numbers changed.

    Returns:
        The adapter id that was applied, or None.
    """
    if not hasattr(segmentation, "_meta"):
        # Not a persisted row, so nothing can have been applied to it. (The
        # cache/streaming tests below drive this function with a stand-in.)
        return None

    try:
        from quantem.finetune.models import active_adapter_for
    except Exception:  # finetune not installed, or no Django app registry
        logger.debug("quantem.finetune unavailable; no adapter lookup", exc_info=True)
        return None

    try:
        adapter = active_adapter_for(segmentation)
    except Exception:
        # An install that never migrated. Loud rather than silent: a user who
        # applied an adapter and gets an uncalibrated run deserves to know.
        logger.warning(
            "Could not read the applied adapter for segmentation %s; running the "
            "released model.",
            segmentation.id,
            exc_info=True,
        )
        return None
    if adapter is None:
        return None

    pack_id = getattr(segmenter, "source_model", None)
    if not getattr(segmenter, "supports_adapters", False):
        logger.info(
            "Adapter %s is applied to segmentation %s but %s cannot use one; "
            "running the released model.",
            adapter.id, segmentation.id, type(segmenter).__name__,
        )
        return None
    if adapter.base_model != pack_id:
        message = (
            f"Applied adapter was fitted on {adapter.base_model}, but this run uses "
            f"{pack_id}; running the released model at its published threshold."
        )
        logger.warning("segmentation %s: %s", segmentation.id, message)
        if on_detail is not None:
            on_detail(message)
        return None

    segmenter.apply_adapter(
        adapter_id=str(adapter.id),
        base_model=adapter.base_model,
        calibrated_threshold=adapter.calibrated_threshold,
        head_file=adapter.head_file,
    )
    if on_detail is not None:
        threshold = adapter.calibrated_threshold
        on_detail(
            f"Using your adapted model '{adapter.name or adapter.base_model}' "
            + (
                f"at its calibrated threshold {threshold:.2f}"
                if threshold is not None
                else "at the published default threshold"
            )
            + (" with a trained head" if adapter.head_file else "")
        )
    return str(adapter.id)


def _normalize_prob_map_metadata(metadata: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def _stage_key(name: str) -> str:
    """Collapse a model or stage name to a compact progress-stage key.

    ``"DINO"`` -> ``"dino"``, ``"combine:foreground"`` -> ``"combine"``.
    """
    lowered = (name or "").strip().lower()
    if not lowered:
        return ""
    base = lowered.split(":", 1)[0]
    return base.replace(" ", "").replace("-", "").replace("_", "")


def _estimate_model_tile_count(
    segmenter: BaseSegmenter,
    image_shape: tuple[int, int],
) -> int:
    """Tiles a segmenter is expected to run, for progress reporting only."""
    estimator = getattr(segmenter, "estimate_dl_tile_count", None)
    if callable(estimator):
        estimated = estimator(image_shape)
        if estimated:
            return max(int(estimated), 1)
    return max(
        int(
            math.ceil(image_shape[0] / _FALLBACK_TILE_PX)
            * math.ceil(image_shape[1] / _FALLBACK_TILE_PX)
        ),
        1,
    )


def run_inference_for_segmentation(
    segmenter: BaseSegmenter,
    segmentation: ImageSegmentation,
    config: SegmentationConfig,
    roi: ImageROI | None = None,
    on_status: Callable[..., None] | None = None,
    on_detail: Callable[[str], None] | None = None,
    force_recompute_prob_maps: bool = False,
    **kwargs,
) -> tuple[InferenceResult, np.ndarray]:
    """Run organelle inference for a segmentation. Returns (result, image_array).

    Args:
        segmenter: The organelle segmenter instance.
        segmentation: The ImageSegmentation instance.
        config: The SegmentationConfig for this segmentation. Currently unused;
            kept so the positional call signature stays stable.
        roi: Optional ROI to restrict inference to.
        on_status: Optional status callback.
        on_detail: Optional callback for human-readable progress messages.
        force_recompute_prob_maps: Ignore any cached probability maps.

    Returns:
        Tuple of (InferenceResult, image_array).
    """
    # Per-run inference settings do not come from `config`: the threshold and
    # the head come from the adapter the user applied (below), and min_area /
    # tile size are per-organelle and per-model facts owned by the segmenter.
    # The parameter stays because quantem.segmentation.organelle_tasks passes it
    # positionally.
    _ = config

    if not segmentation.asset_id:
        raise ValueError("Segmentation has no target asset")

    # Before anything is loaded: a guided fine-tuning result the user applied
    # changes which weights and which threshold this run uses.
    applied_adapter_id = apply_active_adapter(segmenter, segmentation, on_detail)
    target_image = get_asset_openable(segmentation.asset)
    roi_id = str(roi.id) if roi else None
    prefix = segmenter.prob_map_prefix
    persist_probability_maps = bool(
        getattr(segmenter, "persist_probability_maps", True)
    )
    if applied_adapter_id and persist_probability_maps:
        # A map on disk was produced by the released head. Reusing it under an
        # adapter would report the adapted model's provenance over the base
        # model's pixels. (The organelle segmenters do not persist maps, so this
        # is a guard rather than a code path they take.)
        force_recompute_prob_maps = True

    use_image_file_prediction = bool(
        roi is None and getattr(segmenter, "supports_image_file_prediction", False)
    )

    # Load image
    if roi:
        img_array = load_image_roi_array(
            target_image, roi.x, roi.y, roi.width, roi.height
        )
    elif use_image_file_prediction:
        # Streaming segmenters read their own tiles; never materialize the
        # full image (a gigapixel asset does not fit in RAM).
        img_array = np.zeros((1, 1), dtype=np.uint8)
    else:
        img_array, _ = load_image_array(target_image)

    analysis_shape = (
        (int(roi.height), int(roi.width))
        if roi is not None
        else (int(target_image.height), int(target_image.width))
    )

    cached_prob_maps: dict[str, np.ndarray | None] = {}
    if persist_probability_maps:
        for model_name in segmenter.get_dl_model_names():
            if force_recompute_prob_maps:
                cached_prob_maps[model_name] = None
                continue
            cached_prob_maps[model_name] = load_prob_map_from_path(
                segmentation, model_name, prefix, roi_id
            )
    else:
        for model_name in segmenter.get_dl_model_names():
            cached_prob_maps[model_name] = None

    dl_model_names = segmenter.get_dl_model_names()

    missing_prob_maps: list[tuple[str, str, int]] = []
    stage_plan: list[tuple[str, float]] = []
    stage_totals: dict[str, int] = {}
    stage_labels: dict[str, str] = {}
    stage_item_labels: dict[str, str] = {}

    for model_name in dl_model_names:
        if cached_prob_maps.get(model_name) is not None:
            continue
        stage_key = _stage_key(model_name)
        tiles = _estimate_model_tile_count(segmenter, analysis_shape)
        stage_plan.append((stage_key, float(tiles)))
        stage_totals[stage_key] = tiles
        stage_labels[stage_key] = model_name
        stage_item_labels[stage_key] = "Tile"
        missing_prob_maps.append((model_name, stage_key, tiles))

    direct_units = (
        segmenter.estimate_image_file_prediction_units(analysis_shape)
        if use_image_file_prediction
        else None
    )
    stage_plan.append(("combine", float(max(direct_units or 1, 1))))
    if direct_units:
        stage_totals["combine"] = int(max(direct_units, 1))
        stage_item_labels["combine"] = "Tile"
    stage_labels["combine"] = (
        "Tiled direct candidate generation"
        if use_image_file_prediction
        else "Direct candidate generation"
        if not persist_probability_maps
        else "Probability map combination"
    )

    total_units = sum(units for _, units in stage_plan)
    if total_units <= 0:
        total_units = 1.0

    stage_ranges: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for stage_key, units in stage_plan:
        start = cursor / total_units
        cursor += units
        end = cursor / total_units
        stage_ranges[stage_key] = (start, end)

    if on_detail is not None:
        if not persist_probability_maps:
            on_detail("This segmenter produces direct candidates and skips probability-map persistence")
            if use_image_file_prediction:
                on_detail(
                    f"Full-image inference will stream tiles from source data (~{stage_totals.get('combine', 1)} tiles)"
                )
        elif missing_prob_maps:
            tiles_summary = ", ".join(
                f"{model_name}: ~{tiles} tiles"
                for model_name, _stage_key, tiles in missing_prob_maps
            )
            on_detail(
                f"Generating {len(missing_prob_maps)} probability map(s): {tiles_summary}"
            )
        elif force_recompute_prob_maps:
            on_detail("Force recompute enabled; regenerating all probability maps")
        else:
            on_detail("All probability maps are cached; skipping map generation")

    stage_start_times: dict[str, float] = {}
    last_detail_fraction: dict[str, float] = {}
    last_detail_time: dict[str, float] = {}

    def on_progress(stage: str, fraction: float):
        if on_status is None and on_detail is None:
            return
        stage_key = _stage_key(stage)
        if stage_key not in stage_ranges:
            return

        bounded_fraction = max(0.0, min(float(fraction), 1.0))
        start, end = stage_ranges[stage_key]
        progress_fraction = start + ((end - start) * bounded_fraction)
        progress_pct = max(0.0, min(progress_fraction * 100.0, 100.0))

        label = stage_labels.get(stage_key, stage_key.replace("_", " ").title())
        total_items = stage_totals.get(stage_key)
        completed = (
            max(0, min(total_items, int(round(total_items * bounded_fraction))))
            if total_items is not None
            else None
        )
        if (
            completed is not None
            and total_items is not None
            and 0.0 < bounded_fraction < 1.0
            and completed == 0
        ):
            completed = 1
        item_label = stage_item_labels.get(stage_key, "Step")

        status_message: str | None = None
        if roi is None:
            if total_items is None:
                status_message = f"{label}: {bounded_fraction * 100.0:.0f}%"
            elif completed is not None:
                status_message = (
                    f"{label}: {bounded_fraction * 100.0:.0f}% "
                    f"({item_label} {completed}/{total_items})"
                )

        if on_status is not None:
            if status_message is not None:
                try:
                    on_status("RUNNING_INFERENCE", progress_pct, status_message)
                except TypeError as exc:
                    if "positional" not in str(exc):
                        raise
                    on_status("RUNNING_INFERENCE", progress_pct)
            else:
                on_status("RUNNING_INFERENCE", progress_pct)

        if on_detail is None:
            return

        now = time.perf_counter()
        if stage_key not in stage_start_times:
            stage_start_times[stage_key] = now
            last_detail_fraction[stage_key] = 0.0
            last_detail_time[stage_key] = 0.0

        prev_fraction = last_detail_fraction.get(stage_key, 0.0)
        prev_time = last_detail_time.get(stage_key, 0.0)
        should_emit = (
            bounded_fraction >= 1.0
            or (bounded_fraction - prev_fraction) >= _DETAIL_PROGRESS_DELTA
            or (
                (now - prev_time) >= _DETAIL_MIN_SECONDS
                and bounded_fraction > prev_fraction
            )
        )
        if not should_emit:
            return

        if total_items is None:
            on_detail(f"{label}: {bounded_fraction * 100.0:.0f}%")
            last_detail_fraction[stage_key] = bounded_fraction
            last_detail_time[stage_key] = now
            return

        elapsed = now - stage_start_times.get(stage_key, now)
        if completed is None:
            completed = max(
                0,
                min(total_items, int(round(total_items * bounded_fraction))),
            )
        item_progress = f"{item_label} {completed}/{total_items}"
        if 0.0 < bounded_fraction < 1.0 and elapsed > 0.0:
            eta_seconds = elapsed * (1.0 - bounded_fraction) / bounded_fraction
            on_detail(
                f"{label}: {bounded_fraction * 100.0:.0f}% ({item_progress}, ETA ~{eta_seconds:.0f}s)"
            )
        else:
            on_detail(f"{label}: {bounded_fraction * 100.0:.0f}% ({item_progress})")
        last_detail_fraction[stage_key] = bounded_fraction
        last_detail_time[stage_key] = now

    # Load models and run prediction
    segmenter.load_models()
    if use_image_file_prediction:
        result = segmenter.predict_from_image_file(
            target_image,
            cached_prob_maps=cached_prob_maps,
            on_progress=on_progress,
            coordinate_offset=(0, 0),
            on_detail=on_detail,
            **kwargs,
        )
    else:
        result = segmenter.predict(
            img_array,
            cached_prob_maps=cached_prob_maps,
            on_progress=on_progress,
            coordinate_offset=(int(roi.x), int(roi.y)) if roi is not None else (0, 0),
            **kwargs,
        )

    if persist_probability_maps:
        for model_name in segmenter.get_dl_model_names():
            should_save = force_recompute_prob_maps or not prob_map_file_exists(
                segmentation,
                model_name,
                prefix,
                roi_id,
            )
            if should_save and model_name in result.prob_maps:
                save_probability_map(
                    segmentation,
                    model_name,
                    result.prob_maps[model_name],
                    prefix,
                    segmenter.generated_flag,
                    roi_id,
                    extra_metadata=_normalize_prob_map_metadata(
                        segmenter.get_probability_map_metadata(model_name)
                    ),
                )


    return result, img_array
