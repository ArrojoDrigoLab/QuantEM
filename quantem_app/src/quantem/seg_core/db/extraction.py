"""
Generic Segment Creation + Candidate Replacement
==================================================

Extract segments and save as CANDIDATE SegmentObjects.
Handles candidate replacement by deleting prior generated inferred/candidate
segments in the affected region, while preserving user-confirmed/excluded labels.

Parameterized by BaseSegmenter.

Geometry is plain shapely in image pixel space, persisted as WKB
(``SegmentObject.geometry_wkb``) alongside indexed float columns
(``bbox_minx/miny/maxx/maxy``, ``centroid_x/centroid_y``). There is no spatial
database: ROI filtering is a numeric bbox range query, refined in Python with
shapely where an exact answer is needed.

This module is the orchestration only. The two pieces of it that carry weight
of their own live beside it:
:mod:`quantem.seg_core.db.candidate_protection` decides what a pass may not
overwrite, and :mod:`quantem.seg_core.db.segment_writer` turns shapes into rows
and reports what the write did.

Two kinds of protection, and only one of them is about objects
--------------------------------------------------------------
``candidate_protection`` answers whether a new shape repeats a rejection. A
confirmed outline is deliberately *not* consulted while Preview materializes
the full probability map: the overlay paint order keeps that preview beneath
confirmed work, and whole-image confirmation later applies the 70% merge rule
or subtracts the confirmed geometry before accepting the remainder. It is not,
and cannot be, an answer to the other thing a user
tells this application -- that a **region** is finished.

A :class:`~quantem.segmentation.models.CompletedROI` polygon, and an
:class:`~quantem.segmentation.models.RoiSegmentationStatus` row with
``is_complete``, both say: *everything of this organelle inside here is already
outlined*. Under owner ruling R13 a new run "may add objects **outside** those
areas freely" -- the contrast is the rule. Adding candidates in the gaps
between a user's own outlines inside a finished area overwrites nothing and
deletes nothing, and still breaks the promise: the area is no longer as they
left it, and the rejection work that marking it done was meant to end is handed
straight back to them.

That is decided by :func:`finished_regions` and applied in
:func:`run_extraction`, before the shapes reach the writer, because it is a
question about *where the pass may write at all* rather than about which
existing row a shape collides with.

One coordinate system, all the way down
---------------------------------------
Every decision made here and in the ``extract_instances`` it calls is made in
**the image's own pixels**: the foreground threshold (on the stored uint8
probability map), the closing radius, the hole fill, the connected-component
labeling, the minimum-area floor, and the polygon that is finally written. The
model's resampled grid does not appear at any point.

That used to be true of everything after the threshold but not of the threshold
itself, which was taken on the model's grid and brought back as a binary mask.
It matters here for two reasons. ``min_area`` and ``close_radius`` are stated in
native pixels (:mod:`quantem.inference.specs`), so a filter applied on the
model's grid would mean a different physical size on every image. And a
candidate set is now reproducible from the stored map alone -- which is what
:func:`quantem.seg_core.db.inference.replay_stored_probability_map` uses to move
the threshold without running the model, feeding this same function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

from quantem.assets.models import ImageROI
from quantem.seg_core.base_segmenter import BaseSegmenter
from quantem.seg_core.types import InferenceResult
from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    RoiSegmentationStatus,
    SegmentationResultVersion,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff import queue_full_overlay_rebuild
from quantem.segmentation.source_models import (
    SOURCE_MODEL_MANUAL,
    default_source_model_for_organelle,
    normalize_source_model,
)

from .candidate_protection import build_protection_index
from .segment_writer import WriteResult, write_segments

logger = logging.getLogger(__name__)

#: Area floor used only when neither the caller nor the segmenter states one.
#: A segmenter that knows its organelle overrides this: see
#: :attr:`quantem.seg_core.base_segmenter.BaseSegmenter.min_area`.
FALLBACK_MIN_AREA = 100


def resolve_min_area(segmenter: object, min_area: int | None) -> int:
    """The native-pixel area floor a run will actually apply.

    Precedence: an explicit caller value, then the segmenter's own
    per-organelle floor, then :data:`FALLBACK_MIN_AREA`.

    This used to be a bare ``min_area: int = 100`` default that was passed to
    every segmenter unconditionally, which silently overrode the per-organelle
    floors the models were tuned with -- a nucleus run filtered at 100 px
    instead of 8000, so the objects a nucleus model is expected to drop as
    debris were saved as nuclei, and a mito run filtered at 100 instead of 60
    dropped small real ones. Deferring to the segmenter is also what makes the
    ``min_area`` recorded in the run identity a true statement about the run.
    """
    if min_area is not None:
        return int(min_area)
    segmenter_floor = getattr(segmenter, "min_area", None)
    if segmenter_floor is not None:
        try:
            return int(segmenter_floor)
        except (TypeError, ValueError):
            logger.warning(
                "Segmenter %s reported an unusable min_area %r; using %d.",
                type(segmenter).__name__,
                segmenter_floor,
                FALLBACK_MIN_AREA,
            )
    return FALLBACK_MIN_AREA


def resolve_source_model(segmenter: object, segmentation: ImageSegmentation) -> str:
    """Which model the objects a run writes are attributed to."""
    source_model = normalize_source_model(getattr(segmenter, "source_model", None))
    if not source_model:
        source_model = default_source_model_for_organelle(
            segmentation.segmentation_type.internal_name
        )
    return source_model


def delete_replaced_candidates(
    segmentation: ImageSegmentation,
    *,
    generated_flag: str,
    source_model: str,
    roi: ImageROI | None,
) -> int:
    """Remove this model's own previous candidates from the region being redone.

    Only rows this model generated and nobody has labeled are touched: a
    CONFIRMED or EXCLUDED object is a user decision and survives every re-run,
    and so does anything another model or the user's own hand produced.
    """
    delete_qs = SegmentObject.objects.filter(
        segmentation=segmentation,
        label_state__in=["CANDIDATE", "INFERRED"],
        source_model=source_model,
        **{f"features__{generated_flag}": True},
    )
    if roi:
        # bbox intersects ROI rectangle, as a numeric range query on the
        # indexed bbox columns (no spatial index, no ST_Intersects).
        delete_qs = delete_qs.filter(
            bbox_maxx__gte=float(roi.x),
            bbox_minx__lte=float(roi.x + roi.width),
            bbox_maxy__gte=float(roi.y),
            bbox_miny__lte=float(roi.y + roi.height),
        )
    deleted_count, _ = delete_qs.delete()
    return deleted_count


def finished_regions(segmentation: ImageSegmentation) -> list[BaseGeometry]:
    """Regions the user has declared exhaustively outlined for this organelle.

    Two records mean it, and both count (owner ruling R13, and the ground-truth
    contract :class:`~quantem.segmentation.models.RoiSegmentationStatus` already
    states in its own docstring):

    * every :class:`~quantem.segmentation.models.CompletedROI` polygon, which is
      an arbitrary shape the user drew round work they finished;
    * every ROI with an ``is_complete`` status **for this segmentation**, which
      is the same claim about a rectangle. Scoped per organelle on purpose: an
      ROI finished for mitochondria says nothing about its ER, and treating it
      as finished for both would silently stop the ER run writing anything
      there.

    A geometry that cannot be read is skipped with a warning rather than failing
    the run, matching
    :func:`~quantem.seg_core.db.candidate_protection.load_protected_geometries`:
    one corrupt row must not cost a whole pass, and the consequence -- that it
    protects nothing here -- is said out loud rather than swallowed.
    """
    regions: list[BaseGeometry] = []

    payloads = [
        bytes(wkb)
        for wkb in CompletedROI.objects.filter(segmentation=segmentation).values_list(
            "geometry_wkb", flat=True
        )
        if wkb
    ]
    if payloads:
        parsed = shapely.from_wkb(payloads, on_invalid="ignore")
        readable = [geometry for geometry in parsed if geometry is not None]
        unreadable = len(payloads) - len(readable)
        if unreadable:
            logger.warning(
                "Skipping %d unreadable completed-area outlines for segmentation "
                "%s; they protect nothing on this pass.",
                unreadable,
                segmentation.pk,
            )
        regions.extend(readable)

    rectangles = RoiSegmentationStatus.objects.filter(
        segmentation=segmentation,
        is_complete=True,
    ).values_list("image_roi__x", "image_roi__y", "image_roi__width", "image_roi__height")
    for x, y, width, height in rectangles:
        if not width or not height:
            continue
        left, top = float(x), float(y)
        regions.append(shapely_box(left, top, left + float(width), top + float(height)))

    return regions


def drop_inside_finished_regions(
    extracted: Sequence,
    regions: Sequence[BaseGeometry],
) -> tuple[list, int]:
    """``(shapes the pass may write, how many a finished region absorbed)``.

    **Membership is by the candidate's centre, not by overlap.** A shape whose
    centre is inside a finished region belongs to that region and the user has
    already accounted for it; one that merely clips the edge belongs to the
    unfinished ground next door, and the boundary is where the user stopped
    drawing rather than a claim about what straddles it. An overlap rule would
    eat real objects lying against the edge of every completed area, which is
    the failure a user could see and not explain.

    The centres go through an :class:`~shapely.STRtree` for the same reason the
    protection index does: a dense image extracts thousands of shapes, and a
    handful of regions tested pairwise against all of them is the wrong shape of
    work.
    """
    shapes = list(extracted)
    if not shapes or not regions:
        return shapes, 0

    centres = np.empty(len(shapes), dtype=object)
    centres[:] = [
        shapely.points(float(shape.centroid_xy[0]), float(shape.centroid_xy[1])) for shape in shapes
    ]
    tree = STRtree(list(regions))
    # ``predicate="intersects"`` on points is containment, and it is evaluated
    # inside the tree rather than as a second pass over the envelope hits.
    left, _right = tree.query(centres, predicate="intersects")
    absorbed = set(left.tolist())
    if not absorbed:
        return shapes, 0
    kept = [shape for index, shape in enumerate(shapes) if index not in absorbed]
    return kept, len(absorbed)


def run_extraction(
    segmenter: BaseSegmenter,
    segmentation: ImageSegmentation,
    result: InferenceResult,
    image: np.ndarray,
    roi: ImageROI | None = None,
    min_area: int | None = None,
    on_status: Callable[[str, float], None] | None = None,
    on_detail: Callable[[str], None] | None = None,
    run_identity: dict[str, object] | None = None,
    include_level: float | None = None,
) -> WriteResult:
    """Extract candidates, replace this model's previous ones, and report.

    Same work as :func:`extract_and_save_segments`, which is the thin
    count-returning wrapper over this. Callers that need to tell the user what
    the pass did -- how many candidates their confirmations absorbed, how much
    manual work was left alone -- want this one.

    ``include_level`` is the dial position this set was extracted at, and is
    recorded on the numbered result. It stays ``None`` for a model run, which is
    the ordinary case: a run's own threshold is a different fact from a level
    the user chose, and defaulting one to the other would show a dial position
    nobody set (see
    :meth:`~quantem.segmentation.models.SegmentationResultVersion.record_new_result`).
    """
    if on_status is not None:
        on_status("EXTRACTING_CANDIDATES", 0)
    if on_detail is not None:
        on_detail("Extracting candidate shapes from probability map")

    area_floor = resolve_min_area(segmenter, min_area)
    coordinate_offset = (float(roi.x), float(roi.y)) if roi else None

    # Extract segments using segmenter's instance extraction
    if on_status is not None:
        on_status("EXTRACTING_CANDIDATES", 5)
    if result.extracted_segments is not None:
        extracted = result.extracted_segments
        if on_detail is not None:
            on_detail(f"Using {len(extracted)} direct candidate shapes from the segmenter")
    else:
        extracted = segmenter.extract_instances(
            result.prob,
            image,
            result.prob_maps,
            min_area=area_floor,
            coordinate_offset=coordinate_offset,
            on_progress=(
                lambda fraction: on_status(
                    "EXTRACTING_CANDIDATES",
                    5.0 + (65.0 * max(0.0, min(float(fraction), 1.0))),
                )
            )
            if on_status is not None
            else None,
        )
    if on_detail is not None:
        on_detail(f"Shape extraction complete: {len(extracted)} raw candidates")
    if on_status is not None:
        on_status("EXTRACTING_CANDIDATES", 70.0)

    # Areas the user has already finished are not this pass's to write in. See
    # the module docstring: this is a question about where the pass may write,
    # which is why it is answered before the shapes reach the protection index.
    extracted, absorbed_by_finished = drop_inside_finished_regions(
        extracted, finished_regions(segmentation)
    )
    if absorbed_by_finished:
        logger.info(
            "Left %d candidate(s) unwritten inside finished areas of segmentation %s",
            absorbed_by_finished,
            segmentation.pk,
        )
        if on_detail is not None:
            # Read in Tasks & Queues by a biologist: says what was not done and
            # why, in the vocabulary of the tick that caused it.
            on_detail(
                f"Left {absorbed_by_finished} shape(s) alone inside areas you "
                "have already marked finished"
            )

    # Read before anything is deleted: which numbered result is this pass
    # replacing? Afterwards the question cannot be answered honestly, because
    # the objects on the table are this pass's own -- and a brand-new
    # segmentation's first run would come out as "version 2". Zero means there
    # was no model result here to replace, so this pass produces version 1.
    replaced_version = (
        SegmentationResultVersion.current_version_for(segmentation)
        if SegmentObject.objects.filter(
            segmentation=segmentation,
            superseded_at__isnull=True,
        )
        .exclude(source_model=SOURCE_MODEL_MANUAL)
        .exists()
        else 0
    )

    # Delete existing generated inferred/candidate segments in affected region.
    source_model = resolve_source_model(segmenter, segmentation)
    deleted_count = delete_replaced_candidates(
        segmentation,
        generated_flag=segmenter.generated_flag,
        source_model=source_model,
        roi=roi,
    )
    if deleted_count:
        logger.info(
            "Deleted %d existing %s generated inferred/candidate segments",
            deleted_count,
            segmenter.name,
        )

    # Preview materializes the full thresholded map. Confirmed objects remain
    # untouched and paint above these candidates; resolving their geometry is
    # deferred to the explicit whole-image Confirm action, where the existing
    # 70% union rule can be applied and non-merged candidates can be clipped.
    # Rejections still suppress the same model's repeated proposal here.
    protection = build_protection_index(
        segmentation,
        source_model,
        include_confirmed=False,
    )

    write_result = write_segments(
        segmentation,
        extracted,
        run_identity=run_identity,
        source_model=source_model,
        protection=protection,
        deleted=deleted_count,
        on_status=on_status,
        on_detail=on_detail,
    )

    if on_status is not None:
        on_status("EXTRACTING_CANDIDATES", 99.0)
    if on_detail is not None:
        on_detail(
            f"Candidate save complete: {write_result.written} saved, "
            f"{len(extracted) - write_result.written} filtered"
        )

    if deleted_count > 0 or write_result.written > 0:
        # The candidate set is not what it was, so every judgement recorded
        # against the old one is about objects that are no longer on screen.
        # This is the only event that says so: numbering the new result is what
        # invalidates a stored quality estimate (invariant I-5), and until this
        # call existed nothing in the tree ever advanced the number, so a spot
        # check taken at one threshold kept feeding the headline after a re-run
        # at another. Guarded by the same condition as the overlay rebuild
        # because it is the same question -- did this pass change the objects.
        try:
            SegmentationResultVersion.record_new_result(
                segmentation,
                after_version=replaced_version,
                run_identity=run_identity,
                include_level=include_level,
            )
        except Exception:
            # Bookkeeping beside a completed run. The objects are already
            # written and are the run's real output; losing their version
            # number must not lose them.
            logger.warning(
                "Could not number the new result for segmentation %s",
                segmentation.id,
                exc_info=True,
            )
        try:
            queue_full_overlay_rebuild(segmentation, source_model=source_model)
        except Exception:
            logger.warning(
                "Failed to queue overlay rebuild after extraction for %s",
                segmentation.id,
                exc_info=True,
            )

    return write_result


def extract_and_save_segments(
    segmenter: BaseSegmenter,
    segmentation: ImageSegmentation,
    result: InferenceResult,
    image: np.ndarray,
    roi: ImageROI | None = None,
    min_area: int | None = None,
    on_status: Callable[[str, float], None] | None = None,
    on_detail: Callable[[str], None] | None = None,
    run_identity: dict[str, object] | None = None,
    include_level: float | None = None,
) -> int:
    """Extract segments and save as CANDIDATE SegmentObjects.

    Performs candidate replacement: deletes existing generated inferred/candidate
    segments (filtered by segmenter's generated_flag) and excludes new ones
    that repeat an EXCLUDED decision. CONFIRMED geometry is resolved only when
    the user confirms the preview, not while the full-map preview is generated.

    Args:
        segmenter: The organelle segmenter instance.
        segmentation: The ImageSegmentation instance.
        result: InferenceResult from the segmenter.
        image: Image array used for extraction.
        roi: Optional ROI for coordinate offset.
        min_area: Minimum segment area in native pixels. ``None`` -- the normal
            case -- defers to the segmenter's own per-organelle floor
            (:attr:`~quantem.seg_core.base_segmenter.BaseSegmenter.min_area`).
            Passing a number here overrides that for every organelle at once,
            which is almost never what a caller means.
        on_status: Optional status callback.
        run_identity: The run that produced ``result``, stamped onto every
            object created here. See :mod:`quantem.segmentation.run_identity`.
            ``None`` writes no ``"run"`` key, which reads downstream as "not
            produced by a model" -- so a real inference path must always pass
            one.
        include_level: the dial position this set was extracted at, recorded on
            the numbered result. ``None`` for a model run; see
            :func:`run_extraction`.

    Returns:
        Number of segments created. :func:`run_extraction` returns the same work
        as a :class:`~quantem.seg_core.db.segment_writer.WriteResult` when the
        caller needs more than the count.
    """
    return run_extraction(
        segmenter,
        segmentation,
        result,
        image,
        roi=roi,
        min_area=min_area,
        on_status=on_status,
        on_detail=on_detail,
        run_identity=run_identity,
        include_level=include_level,
    ).written
