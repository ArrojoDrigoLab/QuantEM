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

Two entry points, and the difference between them is deliberate
---------------------------------------------------------------
:func:`run_inference_for_segmentation` runs the model. It always runs the model:
a stored probability map is never substituted for one, because "re-run" has to
mean the model saw the image again.

:func:`replay_stored_probability_map` runs no model at all. It reads the uint8
probability map a previous run stored -- already in the image's own pixel
coordinates -- and hands it back to the segmenter to be thresholded again at a
new value. That is exact rather than approximate: the fresh run thresholded
those same bytes with the same function, so a replay at threshold ``T`` produces
the objects a fresh run at ``T`` would have produced, not objects close to them.
See :mod:`quantem.inference.resample` for why the ordering makes that true by
construction, and :mod:`quantem.segmentation.prob_maps.persistence` for what is
stored.

Both then go through the same :func:`quantem.seg_core.db.extraction.
extract_and_save_segments`, so nothing downstream can tell -- or needs to tell --
which of the two produced a candidate set.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

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
from .tile_progress import TileProgressWriter

logger = logging.getLogger(__name__)

_DETAIL_PROGRESS_DELTA = 0.05
_DETAIL_MIN_SECONDS = 1.0

#: Fallback tile edge used only for the progress estimate when a segmenter does
#: not implement ``estimate_dl_tile_count``. The real tile size is a per-model
#: fact (512 for QuantEM ViT-B, 518 for OmniEM ViT-L) owned by the segmenter.
_FALLBACK_TILE_PX = 512

#: What the deep-learning pass is called on screen.
#:
#: A segmenter names its own outputs after the architecture that produces them
#: -- ``"DINO"`` for the foundation encoder -- and those names used to travel
#: verbatim into the job message, which the Tasks drawer renders. ``DINO: 57%
#: (Tile 32/56)`` is what a user saw. An internal model codename is on the
#: plan's never-show list, so the pass is named for what it does instead. The
#: architecture is still available where it belongs: the pack id and the
#: organelle go into ``progress_detail_json``, which is machine-readable and is
#: never rendered.
_DL_STAGE_LABEL = "Segmenting"

#: What follows the tiles, per segmenter kind. All three are the same phase of
#: the run seen from three implementations; none of them may name an internal
#: artefact ("probability map", "direct candidate") at a user.
_COMBINE_LABEL_STREAMING = "Finding objects"
_COMBINE_LABEL_DIRECT = "Finding objects"
_COMBINE_LABEL_PROB_MAPS = "Combining results"


@dataclass
class TileWindow:
    """Where one organelle's tiles sit inside a multi-organelle job's count.

    A run over three organelles is **one** job row with **one** tile count, so
    each organelle in turn has to report into a shared denominator rather than
    restarting one of its own. Without this the row would count 0-858, then jump
    back to 0-88 when the next organelle started, and the "never runs backwards"
    invariant would be broken once per organelle.

    ``base`` is how many tiles the organelles before this one actually walked;
    ``total`` is the sum of every organelle's plan. ``walked`` is written by the
    run as it goes, so the driver can advance ``base`` by *work that happened*
    rather than by work that was planned -- an organelle that failed at tile 18
    of 858 must not hand 858 to the next one, or the bar reaches 100 % having
    skipped most of the image.

    A single-organelle run passes no window at all and behaves exactly as it did
    before this existed: its own plan is its own denominator.
    """

    base: int = 0
    total: int = 0
    #: Tiles this leg has been seen to complete. Monotone; only ever raised.
    walked: int = field(default=0)

    def note_walked(self, done: int) -> None:
        self.walked = max(self.walked, max(int(done), 0))

    def advance(self) -> None:
        """Fold this leg's walked tiles into the base and start the next one."""
        self.base += self.walked
        self.walked = 0


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


def _active_job_reporter():
    """The reporter for the job this run is inside, or ``None``.

    Looked up lazily and defensively, exactly as
    :mod:`quantem.seg_core.db.tile_progress` does: ``seg_core`` runs from the
    CLI and from tests with no job behind it, and on an install where
    ``quantem.jobs`` is not migrated. Finding nothing means nobody is watching.
    """
    try:
        from quantem.jobs.reporter import active_reporter  # noqa: PLC0415
    except Exception:
        logger.debug("quantem.jobs.reporter unavailable", exc_info=True)
        return None
    try:
        return active_reporter()
    except Exception:
        logger.debug("Could not read the active job reporter", exc_info=True)
        return None


def report_device_notices(segmenter: BaseSegmenter) -> list[str]:
    """Put what the run had to do about hardware onto the job record.

    A model that cannot execute on the graphics card, a card that ran short of
    memory and shrank the batch, a run that moved to the processor part-way
    through: each of those is a plain sentence the segmenter already composed
    (``device_notices``, written by :mod:`quantem.inference.engine`'s fallback
    ladder), and until this existed **nothing read them**. A run that silently
    fell back to the processor took twenty minutes when the estimate said one
    and told the user nothing, which is the surprise that destroys trust in the
    estimate -- and is precisely what the fallback copy was written to prevent.

    Written with ``reporter.log`` rather than the ``on_detail`` channel on
    purpose. ``on_detail`` sets the job row's single ``message``, which the next
    progress line overwrites a fraction of a second later; the job log is where
    a sentence stays put long enough to be read after the run has finished.

    Done here rather than in the run task because this function is what runs
    the model: every caller of it -- a whole-image run, a patch run, a re-run --
    gets the notices, and a new caller cannot forget to ask for them.

    Returns:
        The sentences that were reported, for a caller that wants to say them
        somewhere else as well. Empty on the ordinary path, which is almost
        always.
    """
    notices = [
        str(sentence).strip()
        for sentence in (getattr(segmenter, "device_notices", ()) or ())
        if str(sentence).strip()
    ]
    if not notices:
        return []

    reporter = _active_job_reporter()
    for sentence in notices:
        logger.warning("%s", sentence)
        if reporter is None:
            continue
        try:
            reporter.log("warning", sentence)
        except Exception:
            # A log line must never fail a run that has already produced its
            # numbers.
            logger.debug("Could not record a device note on the job", exc_info=True)
    return notices


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


def _dl_stage_label(index: int, count: int) -> str:
    """Name the deep-learning pass without naming the architecture.

    Numbered only when a segmenter really does run more than one pass, so the
    ordinary single-model case reads as a sentence rather than as a machine
    log.
    """
    if count <= 1:
        return _DL_STAGE_LABEL
    return f"{_DL_STAGE_LABEL} (pass {index + 1} of {count})"


def _tile_phrase(done: int, total: int) -> str:
    """``"32 of 56 tiles"``.

    "tiles" is the one piece of machine vocabulary the plan keeps, on the
    condition that it never appears alone: every caller here puts it after a
    percentage, which glosses it by construction.
    """
    return f"{done} of {total} {'tile' if total == 1 else 'tiles'}"


def _time_left_phrase(seconds: float) -> str | None:
    """``"about 4 min left"``, or None when the estimate is not worth saying."""
    if not (seconds > 0) or seconds != seconds or seconds == float("inf"):
        return None
    if seconds < 10:
        return "a few seconds left"
    if seconds < 90:
        return f"about {int(round(seconds / 5.0) * 5)} seconds left"
    minutes = int(round(seconds / 60.0))
    return f"about {minutes} min left" if minutes > 1 else "about a minute left"


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
    tile_window: TileWindow | None = None,
    image_array: np.ndarray | None = None,
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
        tile_window: Where this run's tiles sit inside a bigger job's count.
            Passed by the multi-organelle driver so that three organelles report
            into one denominator on one row. See :class:`TileWindow`.
        image_array: Pixels already decoded by the caller. The multi-organelle
            driver reads the image **once** and hands the same array to every
            organelle; decoding a 60 MP asset four times is pure waste, and on
            the 8 GB laptop of R3 it is four peaks of the same allocation. None
            means "read it yourself", which is what every other caller does.

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
    if image_array is not None and roi is None and not use_image_file_prediction:
        # Already decoded by the caller, once, for every organelle in this job.
        # Not reused for an ROI (the crop differs) and not for a streaming
        # segmenter (which reads its own tiles and must never be handed a
        # gigapixel array).
        img_array = image_array
    elif roi:
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

    pending_dl_models = [
        model_name
        for model_name in dl_model_names
        if cached_prob_maps.get(model_name) is None
    ]
    for index, model_name in enumerate(pending_dl_models):
        stage_key = _stage_key(model_name)
        tiles = _estimate_model_tile_count(segmenter, analysis_shape)
        stage_plan.append((stage_key, float(tiles)))
        stage_totals[stage_key] = tiles
        # Not ``model_name``: see _DL_STAGE_LABEL. The architecture's own name
        # for itself is not the user's name for what is happening.
        stage_labels[stage_key] = _dl_stage_label(index, len(pending_dl_models))
        missing_prob_maps.append((model_name, stage_key, tiles))

    direct_units = (
        segmenter.estimate_image_file_prediction_units(analysis_shape)
        if use_image_file_prediction
        else None
    )
    stage_plan.append(("combine", float(max(direct_units or 1, 1))))
    if direct_units:
        stage_totals["combine"] = int(max(direct_units, 1))
    stage_labels["combine"] = (
        _COMBINE_LABEL_STREAMING
        if use_image_file_prediction
        else _COMBINE_LABEL_DIRECT
        if not persist_probability_maps
        else _COMBINE_LABEL_PROB_MAPS
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

    # The tile total the run is about to walk. Announced before the model loads
    # (below) so the denominator is on the row through the 4-20 s load, and used
    # here for the planning sentence. ``missing_prob_maps`` is the deep-learning
    # pass; a streaming segmenter counts its tiles on the combine stage instead.
    planned_tiles = next(
        (tiles for _model, _key, tiles in missing_prob_maps if tiles > 0),
        stage_totals.get("combine") or 0,
    )

    if on_detail is not None:
        # None of these may name a probability map, a direct candidate or a
        # model architecture: they land in ``Job.message``, which the Tasks
        # drawer shows a user.
        if planned_tiles:
            # "tiles" never stands unglossed: saying what they cover is what
            # makes the word mean something to somebody who has never met it.
            on_detail(
                f"Planning the pass: {planned_tiles} tiles across this "
                f"{'patch' if roi is not None else 'image'}"
            )
        elif force_recompute_prob_maps:
            on_detail("Starting again from the image rather than earlier results")
        else:
            on_detail("Reusing the results already computed for this image")

    stage_start_times: dict[str, float] = {}
    last_detail_fraction: dict[str, float] = {}
    last_detail_time: dict[str, float] = {}
    #: One writer per counting stage, so a segmenter that runs two tiled passes
    #: reports each pass's own count instead of the second one being read as the
    #: first going backwards.
    tile_writers: dict[str, TileProgressWriter] = {}

    #: The stage whose tiles are *this run's* headline count. Only that stage
    #: contributes to a multi-organelle job's shared denominator: the combine
    #: phase is counted inside the plan the window was built from, and counting
    #: it twice would push the wave past its own total.
    primary_stage = next(
        (key for _model, key, tiles in missing_prob_maps if tiles > 0),
        "combine",
    )

    def report_units(stage_key: str, completed: int, total_items: int) -> None:
        """Put ``completed of total_items`` onto the job row for this stage.

        The offsetting into a multi-organelle wave is **not** done here: it is
        done once, in :func:`quantem.jobs.reporter.apply_unit_window`, because
        the tiling loop's own ``UnitProgressScope`` writes the same three
        columns and the two have to agree. What this does is tell the window how
        far this organelle got, so the driver can advance the base by work that
        happened rather than by work that was planned.
        """
        if tile_window is not None:
            if stage_key != primary_stage:
                # The combine phase is already inside the plan the window was
                # built from; counting it again would push the wave past its
                # own total.
                return
            tile_window.note_walked(completed)
        writer = tile_writers.get(stage_key)
        if writer is None:
            writer = tile_writers[stage_key] = TileProgressWriter()
        writer.report(completed, total_items)

    def on_progress(stage: str, fraction: float):
        stage_key = _stage_key(stage)
        if stage_key not in stage_ranges:
            return

        bounded_fraction = max(0.0, min(float(fraction), 1.0))
        start, end = stage_ranges[stage_key]
        progress_fraction = start + ((end - start) * bounded_fraction)
        progress_pct = max(0.0, min(progress_fraction * 100.0, 100.0))

        label = stage_labels.get(stage_key, stage_key.replace("_", " ").title())
        total_items = stage_totals.get(stage_key)
        # Exact, not nudged. ``fraction`` is ``done / total`` straight out of
        # the tiling loop, so this reproduces the loop's own integer. The old
        # code forced a 0 up to 1 "so it does not look stuck", which is how the
        # text on screen came to disagree with the structured count beside it.
        completed = (
            max(0, min(total_items, int(round(total_items * bounded_fraction))))
            if total_items is not None
            else None
        )

        # The structured count, written on its own cadence. This -- not the
        # sentence below -- is what the UI reads; the sentence exists for the
        # job log and for job kinds that count nothing.
        if total_items is not None and completed is not None:
            report_units(stage_key, completed, total_items)

        if on_status is None and on_detail is None:
            return

        status_message: str | None = None
        if total_items is None:
            status_message = f"{label}: {bounded_fraction * 100.0:.0f}%"
        elif completed is not None:
            # One divisor. The percentage and the count in this sentence are
            # the same fraction, and it is the tiling plan's fraction -- not
            # ``progress``, which also carries the stages either side of the
            # tiles and therefore divides by more than the plan's tile count.
            status_message = (
                f"{label}: {bounded_fraction * 100.0:.0f}% "
                f"({_tile_phrase(completed, total_items)})"
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
        item_progress = _tile_phrase(completed, total_items)
        time_left = (
            _time_left_phrase(elapsed * (1.0 - bounded_fraction) / bounded_fraction)
            if 0.0 < bounded_fraction < 1.0 and elapsed > 0.0
            else None
        )
        if time_left is not None:
            on_detail(
                f"{label}: {bounded_fraction * 100.0:.0f}% "
                f"({item_progress}, {time_left})"
            )
        else:
            on_detail(f"{label}: {bounded_fraction * 100.0:.0f}% ({item_progress})")
        last_detail_fraction[stage_key] = bounded_fraction
        last_detail_time[stage_key] = now

    # The denominator goes on the job row *before* the model is loaded. Loading
    # an exported encoder takes 4-20 s (minutes on the rebuild fallback) and
    # reported nothing at all, so the run read as frozen; with the plan already
    # written, that window says "loading the model - 0 of 56 tiles".
    if planned_tiles:
        announced = tile_writers[primary_stage] = TileProgressWriter()
        announced.announce(int(planned_tiles))

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

    # Where the run actually ran, if that is not where it was asked to run. Read
    # here, immediately after the pass, because the segmenter clears these at
    # the start of the next one.
    report_device_notices(segmenter)

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


class ProbabilityMapMissing(RuntimeError):
    """Base for "the stored probability map cannot be used here".

    Exists for its **name**. :data:`quantem.core.error_codes._CLASS_NAMES` maps
    the class name ``ProbabilityMapMissing`` to
    :attr:`~quantem.core.error_codes.ErrorCode.PROBABILITY_MAP_MISSING`, and
    :func:`~quantem.core.error_codes.classify_exception` walks the MRO by name
    rather than importing anything -- this module has to keep working on an
    install with no torch. Until something inherited it, that catalogue entry
    matched no exception in the tree, so the one failure with a "Run inference
    again" control behind it reached the client as an uncoded sentence and
    rendered as red text with no way forward.
    """


class StoredMapUnavailable(ProbabilityMapMissing):
    """There is no stored probability map to replay for this segmentation.

    Carries a sentence a user can act on. Raised rather than returned so a
    caller cannot mistake "nothing to replay" for "replayed and found nothing":
    those need different words on screen, and the difference is whether the
    model has to run again.

    The sentence is **not** interchangeable between the cases that raise it. A
    map that was never written and a map from an older build both end in "run
    the model again", but only the second will keep being refused until it is
    replaced, and a user told the same thing either way cannot tell which they
    are in. See :mod:`quantem.segmentation.prob_maps.persistence`.
    """


def replay_stored_probability_map(
    segmenter: BaseSegmenter,
    segmentation: ImageSegmentation,
    *,
    threshold: float | None = None,
    roi: ImageROI | None = None,
    on_detail: Callable[[str], None] | None = None,
    on_stored_metadata: Callable[[dict[str, object]], None] | None = None,
) -> tuple[InferenceResult, np.ndarray]:
    """Re-threshold a stored probability map. No model is loaded or run.

    This is the backend of the accuracy dial. The map a previous run stored is
    already in the image's own pixel coordinates and is the array that run
    thresholded, so moving the threshold is genuinely the same operation the run
    performed -- :func:`quantem.inference.resample.binarize_quantized` on the
    same bytes -- and the objects that come out are the objects a fresh run at
    this threshold would have produced. Every object-level filter after it
    (closing radius, hole fill, labeling, minimum area) is unchanged and runs in
    the same native coordinates it always did.

    Args:
        segmenter: the segmenter whose model produced the map. It must be able
            to adopt a stored map (``adopt_native_probability_map``); one that
            cannot has no stored-map contract and must be re-run.
        segmentation: the segmentation the map belongs to.
        threshold: the new foreground threshold. ``None`` keeps the segmenter's
            own -- which is what a caller replaying at the run's own settings
            wants, e.g. to re-extract after the objects were cleared.
        roi: replay the map stored for this ROI rather than the full image.
        on_detail: optional job-log callback.
        on_stored_metadata: optional receiver for the model-run provenance
            saved beside the probability bytes.

    Returns:
        ``(InferenceResult, image_array)``, the same pair
        :func:`run_inference_for_segmentation` returns, so the caller hands it
        to :func:`quantem.seg_core.db.extraction.extract_and_save_segments`
        unchanged.

    Raises:
        StoredMapUnavailable: no map is stored, the stored map does not cover
            the region asked for, or this segmenter has no stored-map contract.
    """
    # Lazy: quantem.segmentation.prob_maps.persistence imports this package's
    # prob_maps module, and importing it at module scope closes that loop
    # through quantem.seg_core.db.__init__.
    from quantem.segmentation.prob_maps.persistence import (  # noqa: PLC0415
        NO_STORED_MAP_MESSAGE,
        load_stored_native_map,
    )

    report = on_detail or (lambda _message: None)

    if not segmentation.asset_id:
        raise ValueError("Segmentation has no target asset")

    adopt = getattr(segmenter, "adopt_native_probability_map", None)
    if not callable(adopt):
        raise StoredMapUnavailable(
            f"{getattr(segmenter, 'source_model', 'This model')} does not keep a "
            "reusable probability map, so its threshold cannot be changed "
            "without running it again."
        )

    model_names = list(segmenter.get_dl_model_names())
    if len(model_names) != 1:
        # A segmenter with several outputs combines them into the foreground map
        # it thresholds, and that combination is not what gets stored. Replaying
        # would have to redo it, which is a second decision procedure and
        # exactly what this ordering exists to avoid.
        raise StoredMapUnavailable(
            "Changing the threshold without re-running is only available for "
            "models that produce a single probability map."
        )
    model_name = model_names[0]

    stored = load_stored_native_map(
        segmentation=segmentation,
        segmenter=segmenter,
        model_name=model_name,
        roi=roi,
    )
    if stored is None:
        # One sentence, written once, in the module that owns the store. The
        # patch case says so, because "no stored result covers this patch" is
        # actionable in a way the whole-image sentence is not -- the map may
        # exist for the image and simply not cover this window.
        if roi is not None:
            raise StoredMapUnavailable(
                "No stored result covers this patch, so the include level "
                "cannot be moved here without running the model again."
            )
        raise StoredMapUnavailable(NO_STORED_MAP_MESSAGE)

    if on_stored_metadata is not None:
        on_stored_metadata(dict(stored.metadata))

    target_image = get_asset_openable(segmentation.asset)
    if roi is not None:
        img_array = load_image_roi_array(
            target_image, roi.x, roi.y, roi.width, roi.height
        )
    else:
        img_array, _ = load_image_array(target_image)

    if stored.shape != tuple(img_array.shape[:2]):
        # The image was replaced or re-imported under the same segmentation.
        # Stretching the map onto the new grid would be a third resampling and a
        # silently different answer.
        raise StoredMapUnavailable(
            f"The stored result is {stored.shape[1]}x{stored.shape[0]} but this "
            f"image is {img_array.shape[1]}x{img_array.shape[0]}; the model has "
            "to run again."
        )

    if threshold is not None:
        setter = getattr(segmenter, "set_fg_threshold", None)
        if not callable(setter):
            raise StoredMapUnavailable(
                f"{getattr(segmenter, 'source_model', 'This model')} does not "
                "accept a threshold change."
            )
        setter(float(threshold))

    adopt(stored.native)
    prob_maps = {model_name: stored.native.as_float()}
    prob = segmenter.combine_prob_maps(prob_maps)
    applied = getattr(segmenter, "fg_threshold", None)
    if applied is None:
        applied = threshold
    logger.info(
        "Replayed stored probability map for segmentation %s (%s) at threshold %s",
        segmentation.id,
        model_name,
        applied,
    )
    report(
        "Re-reading the stored result"
        + (f" at threshold {float(applied):.2f}" if applied is not None else "")
        # An em dash, not "--". This line is rendered in Tasks & Queues, where
        # a literal double hyphen reads as a command-line flag; I-12 names it.
        + " — the model does not need to run again"
    )
    return InferenceResult(prob_maps=prob_maps, prob=prob), img_array
