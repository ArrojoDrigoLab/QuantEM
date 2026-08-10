"""Segmentation execution helpers used by DB-backed job handlers."""

from __future__ import annotations

import logging
import time
import uuid
from typing import NoReturn

from quantem.assets.models import ImageROI
from quantem.seg_core.db.extraction import extract_and_save_segments, resolve_min_area
from quantem.seg_core.db.inference import run_inference_for_segmentation
from quantem.seg_core.model_errors import translate_model_error
from quantem.seg_core.registry import get_segmenter

from .instance_params import supports_instance_params
from .models import ImageSegmentation, SegmentationConfig, SegmentObject
from .prob_maps.persistence import persist_run_probability_maps
from .run_identity import read_run_identity, run_identity_from_segmenter
from .source_models import normalize_source_model, resolve_segmenter_internal_name

logger = logging.getLogger(__name__)


_STATUS_MIN_INTERVAL_SECONDS = 0.5
_STATUS_MIN_PROGRESS_DELTA = 0.5
_DETAIL_MIN_INTERVAL_SECONDS = 0.5

#: What to tell a user whose run completed and found nothing **in an image with
#: nothing labelled in it**. A run that ends at "Candidates ready" with zero
#: objects is indistinguishable from a silent failure unless the terminal status
#: says which one it was, so it says so.
NO_OBJECTS_MESSAGE = (
    "completed: no objects found. Check the image's pixel size first, then the "
    "detection threshold and the selected model."
)

#: The first thing to check, and it was not on the list at all.
#:
#: Every released pack declares a ``canonical_nm`` and the run resamples the
#: image to it (:func:`_build_segmenter_kwargs`), so the pixel size decides what
#: apparent size the model sees an organelle at. It is the one input that turns
#: a working model into one that finds nothing, and unlike the threshold it is
#: usually *wrong* rather than badly chosen: one reported image measured 0, 19,
#: 120 and 233 objects over byte-identical pixels at 5 nm, unset, 10 nm and
#: 20 nm.
#:
#: Lowering the threshold on a wrongly-scaled run does not recover the objects.
#: It produces different rubbish, so this is named before the threshold.
_PIXEL_SIZE_UNSET_STEP = (
    "This image has no pixel size, so the run used the model's native scale. "
    "Set the pixel size on the image and run again — it decides what size the "
    "model thinks these organelles are, and it is the likeliest reason a run "
    "finds nothing."
)


def _pixel_size_step(pixel_size_nm: float | None) -> str:
    if pixel_size_nm is None:
        return _PIXEL_SIZE_UNSET_STEP
    return (
        f"Check the image's pixel size ({pixel_size_nm:g} nm/px). It decides "
        "what size the model thinks these organelles are, and a wrong value "
        "makes a working model find nothing — check it before the threshold, "
        "because lowering the threshold on a wrongly-scaled run does not bring "
        "the objects back."
    )

#: Label states that suppress a new candidate landing on top of them. See
#: :func:`quantem.seg_core.db.extraction.extract_and_save_segments`: a candidate
#: overlapping a CONFIRMED object by >=30%, or an EXCLUDED one by >=80%, is
#: dropped rather than saved.
_SUPPRESSING_LABEL_STATES = ("CONFIRMED", "EXCLUDED")


def _labelled_and_uncalibrated(segmentation: ImageSegmentation) -> tuple[int, int]:
    """``(labelled here, how many of those were produced with no pixel size)``.

    Both counts come from one pass so the proofread branch of
    :func:`_zero_object_advice` costs one query rather than two. Only the stamp
    is trusted: an object with no run identity is hand-drawn or predates
    stamping, and neither says anything about the scale a run used.
    """
    labelled = 0
    uncalibrated = 0
    rows = (
        SegmentObject.objects.filter(
            segmentation=segmentation, label_state__in=_SUPPRESSING_LABEL_STATES
        )
        .values_list("features", flat=True)
        .iterator(chunk_size=1000)
    )
    for features in rows:
        labelled += 1
        stamp = read_run_identity(features)
        if stamp is not None and stamp.get("native_pixel_size_nm") is None:
            uncalibrated += 1
    return labelled, uncalibrated


def _stale_scale_step(segmentation: ImageSegmentation, uncalibrated: int) -> str:
    """Why this re-run could not lift the uncalibrated stamp, and what can.

    The gap this closes was reported end to end. The analysis bundle told the
    user "Set the image's pixel size and re-run inference"; they set it and
    re-ran; the run completed SUCCESS with ``segment_count: 0`` and told them
    the 41 objects they had already labelled were "exactly as they were". All of
    that is correct, and none of it mentioned that the objects keep
    ``native_pixel_size_nm: null`` for good, that every future bundle therefore
    carries the same caveat, or that the instruction they had just followed
    could never have worked.

    The endpoint is named because it is the only route there is: no screen
    calls it (the labeling screen's delete actions target CANDIDATE and INFERRED
    objects), and ``unique_segmentation_per_asset`` refuses a second
    segmentation of the same organelle on the same image, so "start a new one"
    is not a way round it either.
    """
    now = _asset_pixel_size_nm(segmentation)
    objects = f"{uncalibrated} object(s)"
    if now is None:
        opening = (
            f"{objects} here were produced while this image had no pixel size, "
            "and it still has none. Setting one now will not change them — a "
            "pixel size is applied when inference runs, not afterwards."
        )
    else:
        opening = (
            f"{objects} here were produced while this image had no pixel size. "
            f"It records {now:g} nm/px now, but that was set after they were "
            "made, so they were not measured at it and no analysis of them will "
            "be: every export of these objects carries the wrong-scale caveat, "
            "including the one that told you to re-run."
        )
    return (
        f"{opening} A re-run cannot replace them, for the reason above, so they "
        "have to be discarded first: POST /api/segmentations/"
        f"{segmentation.id}/labels/clear removes this segmentation's confirmed "
        "and excluded objects, and inference run after that produces a set "
        "stamped with the image's pixel size. No screen offers that yet."
    )


def _zero_object_advice(segmentation: ImageSegmentation) -> tuple[str, str, list[str]]:
    """``(job message, screen headline, next steps)`` for a run that made nothing.

    There are two ways to create nothing and they need opposite advice.

    *Nothing is labelled in this image.* The model genuinely saw nothing, and
    the scale, the threshold and the choice of model are the things to check --
    in that order; see :data:`_PIXEL_SIZE_UNSET_STEP` for why the scale leads a
    list it used not to appear on at all.

    *The image is already proofread.* Extraction drops any candidate that lands
    on an object a person has already confirmed or excluded, so a re-run over a
    finished image is **expected** to create nothing. Telling that user to
    "lower the detection threshold and run again" -- which is what this reported
    unconditionally -- pushes them to change a setting they should not touch,
    and to re-run a model over work that is already done, in order to fix
    something that is not broken.

    The job message and the screen headline say the same thing in the two
    registers their readers need: a queue row prefixed ``completed:``, and a
    sentence that stands on its own next to "Candidates ready".

    The proofread branch also has to answer *why the user ran this at all*. The
    commonest reason is the analysis bundle's own instruction to set the pixel
    size and re-run, and over a proofread image that instruction cannot work.
    See :func:`_stale_scale_step`.
    """
    labelled, uncalibrated = _labelled_and_uncalibrated(segmentation)

    if labelled <= 0:
        return (
            NO_OBJECTS_MESSAGE,
            "This run finished without finding any objects.",
            [
                _pixel_size_step(_asset_pixel_size_nm(segmentation)),
                "Lower the detection threshold and run again.",
                "Check that the selected model is trained for this organelle.",
            ],
        )

    next_steps = [
        f"Nothing changed: the {labelled} object(s) you have already labelled "
        "here are exactly as they were.",
        "A candidate that lands on an object you have already confirmed or "
        "excluded is not added again, so a re-run over a proofread image is "
        "expected to find nothing new.",
    ]
    if uncalibrated:
        next_steps.append(_stale_scale_step(segmentation, uncalibrated))
    next_steps.append(
        "If you think objects were missed, run over an area you have not "
        "labelled yet rather than lowering the threshold over one you have."
    )
    return (
        (
            f"completed: no new objects. The {labelled} object(s) already labelled "
            "in this image are unchanged."
        ),
        (
            f"This run added no new objects. The {labelled} object(s) already "
            "labelled in this image are unchanged."
        ),
        next_steps,
    )


def zero_object_outcome(segmentation: ImageSegmentation) -> tuple[str, list[str]]:
    """The job's terminal message and next steps. See :func:`_zero_object_advice`."""
    job_message, _, next_steps = _zero_object_advice(segmentation)
    return job_message, next_steps


def zero_object_notice(segmentation: ImageSegmentation) -> dict:
    """The same finding, addressed to a screen instead of to a job log.

    All of this used to exist and reach nobody. The advice was written into the
    job's log and result, which no screen in the application renders, while the
    labeling header and the viewer chip both read the segmentation's stage --
    and the stage a finished run leaves behind is ``CANDIDATES_READY``. So a run
    that produced nothing showed "Mitochondria — Candidates ready", the same
    words as a run that produced two hundred, and the one place that knew
    better was a log file.

    Served on the segmentation payload
    (``ImageSegmentationSerializer.run_notice``) so it arrives with the stage it
    qualifies, in the same response, with no second request to correlate.
    """
    _, headline, next_steps = _zero_object_advice(segmentation)
    return {
        "kind": "no_objects",
        "message": headline,
        "next_steps": next_steps,
    }


def _build_segmenter_kwargs(
    segmentation: ImageSegmentation,
    config: SegmentationConfig,
    *,
    segmenter_internal_name: str,
    source_model: str | None = None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if segmenter_internal_name.startswith("dino_"):
        # DINO segmenters pick the family (quantem/omniem) from the source model.
        kwargs["source_model"] = source_model

    # Instance-param support is a property of the *segmentation type*, not of the
    # segmenter that happens to serve it. QuantEM registers the DINO segmenters
    # as ``dino_<organelle>``, which need not match the segmentation type's own
    # name, so checking the registry name here silently dropped the user's
    # threshold and downsample settings.
    if supports_instance_params(segmentation.segmentation_type.internal_name):
        kwargs["instance_params"] = config.get_instance_params()

    # The asset's real pixel size, so a model with a ``canonical_nm`` resamples
    # to the scale it was trained and benchmarked at.
    #
    # Six of the eight released packs declare one -- 8 nm for mito and lipid
    # droplets, 25 nm for nuclei -- and without this the segmenter falls back to
    # native scale. That is not a small difference: a 5 nm/px image runs at 5 nm
    # instead of 8, the model sees objects at the wrong apparent size, and every
    # number downstream (counts, areas, densities, enrichments, calibrated
    # thresholds) inherits it. Nothing else in the app would have said so; the
    # only trace was a logger.warning.
    kwargs["pixel_size_nm"] = _asset_pixel_size_nm(segmentation)

    return kwargs


def _pack_id_for(segmenter) -> str | None:
    spec = getattr(segmenter, "model_spec", None)
    return getattr(spec, "pack_id", None) or normalize_source_model(
        getattr(segmenter, "source_model", None)
    ) or None


def _fail_segmentation(
    segmentation: ImageSegmentation,
    exc: BaseException,
    *,
    segmenter,
) -> NoReturn:
    """Record a failed run and re-raise, with a message aimed at the user.

    A model-availability failure arrives here carrying maintainer instructions:
    clone Meta's ``dinov3`` and run ``python -m quantem.inference.export``.
    Correct for whoever builds a release, and useless-to-alarming for the person
    who just clicked "Run segmentation" -- who cannot do either, and whose real
    fix (reinstall from a release bundle) is what the CLI and the Models screen
    already tell them.

    :func:`quantem.seg_core.model_errors.translate_model_error` swaps in that
    one shared answer. Everything else keeps its own message, and the original
    text survives in the log, which ``logger.exception`` has already written.

    Re-raised as the same exception class so retry classification and any
    ``except`` upstream behave exactly as before; the queue picks the new text
    up because it formats the job message from ``str(exc)``.
    """
    message = translate_model_error(
        exc,
        pack_id=_pack_id_for(segmenter),
        log_context=f"segmentation {segmentation.id}",
    )
    # A failed run does not invalidate what an earlier successful one found, and
    # a red "Failed" over 19 perfectly good objects reads as though it does.
    surviving = SegmentObject.objects.filter(segmentation=segmentation).count()
    if surviving:
        message = (
            f"{message}\n\n"
            f"The {surviving} object(s) already in this segmentation are from an "
            "earlier run and are unaffected."
        )
    segmentation.status_stage = "FAILED"
    segmentation.status_error = message
    segmentation.save(update_fields=["status_stage", "status_error"])

    # Stamp the exception so the queue's failure reconciler knows this FAILED
    # state is *this attempt's own conclusion* and keeps the message above. An
    # unmarked exception dying on an already-FAILED segmentation means the
    # FAILED belongs to an older attempt, and the reconciler overwrites it
    # with the newer failure instead of letting a stale error outlive it.
    from quantem.jobs.failure_reconcile import mark_domain_status_recorded

    if message == str(exc):
        raise mark_domain_status_recorded(exc)
    try:
        replacement = type(exc)(message)
    except Exception:
        replacement = RuntimeError(message)
    raise mark_domain_status_recorded(replacement) from exc


def _run_id(reporter) -> str:
    """The id this run is recorded under: its job's, or a fresh one.

    Objects from one run must share an id so a manifest can group them, and the
    job id is the one a user can look up in Tasks & Queues. A run driven outside
    the queue (a CLI call, a test) still gets a real id rather than ``None`` --
    the alternative is objects that claim to come from a model and cannot say
    which run, which is the gap this record closes.
    """
    job_id = getattr(reporter, "job_id", None)
    text = str(job_id).strip() if job_id is not None else ""
    return text or str(uuid.uuid4())


def _asset_pixel_size_nm(segmentation: ImageSegmentation) -> float | None:
    """The segmentation's asset pixel size in nm, or None if uncalibrated."""
    asset = getattr(segmentation, "asset", None)
    value = getattr(asset, "pixel_size_nm", None) if asset is not None else None
    try:
        size = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return size if size and size > 0 else None


def _make_status_callback(
    segmentation,
    reporter=None,
    *,
    include_detail_message: bool = False,
):
    last_stage = segmentation.status_stage
    last_progress = float(segmentation.status_progress or 0.0)
    last_saved_monotonic = 0.0

    def on_status(stage: str, progress: float, message: str | None = None):
        nonlocal last_progress, last_saved_monotonic, last_stage
        progress = max(0.0, min(float(progress), 100.0))
        now = time.monotonic()

        stage_changed = stage != last_stage
        progress_changed = abs(progress - last_progress) >= _STATUS_MIN_PROGRESS_DELTA
        final_progress = progress >= 100.0
        interval_elapsed = (now - last_saved_monotonic) >= _STATUS_MIN_INTERVAL_SECONDS

        should_save = stage_changed or final_progress or (
            progress_changed and interval_elapsed
        )
        if not should_save:
            return

        segmentation.status_stage = stage
        segmentation.status_progress = progress
        segmentation.status_error = ""
        segmentation.save(
            update_fields=["status_stage", "status_progress", "status_error"]
        )

        if reporter is not None:
            if include_detail_message and message:
                reporter.update(progress=progress, message=message)
            else:
                reporter.update(progress=progress)

        last_stage = stage
        last_progress = progress
        last_saved_monotonic = now

    return on_status


def _make_detail_callback(reporter):
    if reporter is None:
        return None

    last_message = ""
    last_sent_monotonic = 0.0

    def on_detail(message: str) -> None:
        nonlocal last_message, last_sent_monotonic
        cleaned = (message or "").strip()
        if not cleaned:
            return

        now = time.monotonic()
        same_message = cleaned == last_message
        interval_elapsed = (
            now - last_sent_monotonic
        ) >= _DETAIL_MIN_INTERVAL_SECONDS
        if same_message and not interval_elapsed:
            return

        reporter.update(message=cleaned)
        last_message = cleaned
        last_sent_monotonic = now

    return on_detail


def _load_segmentation(
    segmentation_id: str, segmentation_type: str
) -> tuple[ImageSegmentation, SegmentationConfig]:
    try:
        segmentation = ImageSegmentation.objects.select_related(
            "asset", "config", "segmentation_type"
        ).get(id=segmentation_id)
    except ImageSegmentation.DoesNotExist as exc:
        raise ValueError(f"Segmentation {segmentation_id} not found") from exc

    internal_name = segmentation.segmentation_type.internal_name
    if segmentation_type != internal_name:
        raise ValueError(
            "payload.segmentation_type must match segmentation internal_name "
            f"(expected {internal_name}, got {segmentation_type})"
        )

    config, _ = SegmentationConfig.objects.get_or_create(segmentation=segmentation)
    return segmentation, config


def _run_segmentation(
    segmentation_id: str,
    segmentation_type: str,
    *,
    roi_id: str | None,
    source_model: str | None = None,
    force_recompute_prob_maps: bool = False,
    reporter=None,
) -> int:
    """Run inference and save candidates. Returns the number of objects created."""
    segmentation, config = _load_segmentation(segmentation_id, segmentation_type)
    segmenter_internal_name = resolve_segmenter_internal_name(
        segmentation_type_internal_name=segmentation.segmentation_type.internal_name,
        source_model=source_model,
    )
    segmenter = get_segmenter(
        segmenter_internal_name,
        **_build_segmenter_kwargs(
            segmentation,
            config,
            segmenter_internal_name=segmenter_internal_name,
            source_model=source_model,
        ),
    )

    roi = None
    if roi_id:
        roi = ImageROI.objects.filter(id=roi_id).first()
        if roi is None:
            raise ValueError(f"ROI {roi_id} not found")

    on_status = _make_status_callback(
        segmentation,
        reporter=reporter,
        include_detail_message=roi is None,
    )
    on_detail = _make_detail_callback(reporter)
    resolved_source_model = normalize_source_model(source_model) or normalize_source_model(
        getattr(segmenter, "source_model", None)
    )
    logger.info(
        "%s inference starting for segmentation %s (asset=%s, roi=%s, source_model=%s)",
        segmenter.name.upper(),
        segmentation_id,
        getattr(segmentation, "asset_id", None),
        roi_id,
        resolved_source_model,
    )

    try:
        on_status("RUNNING_INFERENCE", 0)
        if on_detail is not None:
            on_detail("Preparing inference workload")
        result, img_array = run_inference_for_segmentation(
            segmenter,
            segmentation,
            config,
            roi,
            on_status=on_status,
            on_detail=on_detail,
            force_recompute_prob_maps=force_recompute_prob_maps,
        )
        # Surface a slow-path model load on the job record. The engine has
        # already logged the WARNING (and best-effort rewritten the export);
        # this line is what a user reading Tasks & Queues sees instead of an
        # unexplained multi-minute "Preparing inference workload".
        encoder_tier = getattr(segmenter, "encoder_tier", None)
        if encoder_tier and encoder_tier != "exported" and reporter is not None:
            try:
                reporter.log(
                    "warning",
                    f"The model's exported encoder (encoder_ts.pt) was missing, so it "
                    f"was rebuilt from raw weights (tier '{encoder_tier}') -- this is "
                    "why the model load took minutes instead of seconds. QuantEM has "
                    "tried to rewrite the export so the next run is fast again.",
                )
            except Exception:  # a log line must never fail the run
                logger.debug("Could not record the slow-path note", exc_info=True)
        # Store the map before extracting candidates. Guided fine-tuning scores
        # itself against what the model predicted here, and until this ran there
        # was no sequence of actions that produced one. See
        # quantem.segmentation.prob_maps.persistence for why it is not the
        # segmenter's own persist_probability_maps that does this.
        persist_run_probability_maps(
            segmentation=segmentation,
            segmenter=segmenter,
            prob_maps=result.prob_maps,
            roi=roi,
            on_detail=on_detail,
        )
        # Built after inference returns, so `fg_threshold` and `adapter_id` are
        # the values the run actually wore (apply_adapter can have replaced the
        # published threshold), and stamped onto every object extraction
        # creates.
        area_floor = resolve_min_area(segmenter, None)
        run_identity = run_identity_from_segmenter(
            segmenter,
            run_id=_run_id(reporter),
            pack_id_fallback=resolved_source_model or segmenter.name,
            native_pixel_size_nm=_asset_pixel_size_nm(segmentation),
            min_area=area_floor,
        )
        logger.info(
            "Run %s: pack=%s threshold=%s adapter=%s ran_at_nm=%s min_area=%s",
            run_identity["id"],
            run_identity["pack_id"],
            run_identity["threshold"],
            run_identity["adapter_id"],
            run_identity["ran_at_nm"],
            run_identity["min_area"],
        )
        count = extract_and_save_segments(
            segmenter,
            segmentation,
            result,
            img_array,
            roi,
            min_area=area_floor,
            on_status=on_status,
            on_detail=on_detail,
            run_identity=run_identity,
        )
        logger.info("Created %d %s segments", count, segmenter.name)
        if count == 0:
            logger.info(
                "%s run over segmentation %s found no objects",
                segmenter.name.upper(),
                segmentation_id,
            )
            zero_message, _ = zero_object_outcome(segmentation)
            # A re-run over a proofread image finding nothing is the expected
            # result, not something to warn about.
            level = "warning" if zero_message == NO_OBJECTS_MESSAGE else "info"
            if reporter is not None:
                reporter.log(level, zero_message)
            if on_detail is not None:
                on_detail(zero_message)
        elif on_detail is not None:
            on_detail(f"Created {count} candidate segments")
        on_status("CANDIDATES_READY", 100.0)
        return count
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception(
            "%s inference failed for segmentation %s: %s",
            segmenter.name.upper(),
            segmentation_id,
            exc,
        )
        _fail_segmentation(segmentation, exc, segmenter=segmenter)
    except Exception as exc:
        logger.exception(
            "Unexpected %s inference error for segmentation %s: %s",
            segmenter.name.upper(),
            segmentation_id,
            exc,
        )
        _fail_segmentation(segmentation, exc, segmenter=segmenter)


def run_segmentation_roi_task(
    segmentation_id: str,
    segmentation_type: str,
    roi_id: str | None = None,
    source_model: str | None = None,
    force_recompute_prob_maps: bool = False,
    reporter=None,
) -> int:
    """Run segmentation inference scoped to an ROI. Returns the object count."""
    return _run_segmentation(
        segmentation_id=segmentation_id,
        segmentation_type=segmentation_type,
        roi_id=roi_id,
        source_model=source_model,
        force_recompute_prob_maps=force_recompute_prob_maps,
        reporter=reporter,
    )


def run_segmentation_full_task(
    segmentation_id: str,
    segmentation_type: str,
    source_model: str | None = None,
    force_recompute_prob_maps: bool = False,
    reporter=None,
) -> int:
    """Run segmentation inference over the full image. Returns the object count."""
    return _run_segmentation(
        segmentation_id=segmentation_id,
        segmentation_type=segmentation_type,
        roi_id=None,
        source_model=source_model,
        force_recompute_prob_maps=force_recompute_prob_maps,
        reporter=reporter,
    )
