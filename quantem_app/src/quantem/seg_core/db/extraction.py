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
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np
import shapely
from shapely.geometry import Polygon as ShapelyPolygon

from quantem.assets.models import ImageROI
from quantem.seg_core.base_segmenter import BaseSegmenter
from quantem.seg_core.types import InferenceResult
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.overlay_ngff import queue_full_overlay_rebuild
from quantem.segmentation.run_identity import RUN_FEATURE_KEY
from quantem.segmentation.source_models import (
    SOURCE_MODEL_MANUAL,
    default_source_model_for_organelle,
    normalize_source_model,
)

logger = logging.getLogger(__name__)

_CONFIRMED_OVERLAP_THRESHOLD = 0.3
_EXCLUDED_OVERLAP_THRESHOLD = 0.8

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


def _to_valid_polygon(coords: list[tuple[float, float]]):
    """Build a shapely polygon from a closed ring, repairing self-intersections.

    Returns None when the ring cannot be turned into a usable polygon.
    """
    try:
        geometry = ShapelyPolygon(coords)
    except (ValueError, TypeError):
        return None
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
        if geometry.is_empty or not geometry.is_valid:
            return None
    if geometry.geom_type == "MultiPolygon":
        # buffer(0) on a bow-tie ring can split it; keep the dominant part.
        geometry = max(geometry.geoms, key=lambda part: part.area)
    if geometry.geom_type != "Polygon":
        return None
    return geometry


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
) -> int:
    """Extract segments and save as CANDIDATE SegmentObjects.

    Performs candidate replacement: deletes existing generated inferred/candidate
    segments (filtered by segmenter's generated_flag), excludes new ones
    that overlap >=30% with CONFIRMED or >=80% with EXCLUDED.

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

    Returns:
        Number of segments created.
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

    # Delete existing generated inferred/candidate segments in affected region.
    generated_flag = segmenter.generated_flag
    source_model = normalize_source_model(getattr(segmenter, "source_model", None))
    if not source_model:
        source_model = default_source_model_for_organelle(
            segmentation.segmentation_type.internal_name
        )
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
    if deleted_count:
        logger.info(
            "Deleted %d existing %s generated inferred/candidate segments",
            deleted_count, segmenter.name,
        )

    # CONFIRMED objects are protected more aggressively than EXCLUDED ones.
    confirmed_geoms = _load_geometries(
        SegmentObject.objects.filter(
            segmentation=segmentation,
            label_state="CONFIRMED",
        )
    )
    excluded_geoms = _load_geometries(
        SegmentObject.objects.filter(
            segmentation=segmentation,
            label_state="EXCLUDED",
            source_model__in=[source_model, SOURCE_MODEL_MANUAL],
        )
    )

    segments_created = 0
    total_extracted = len(extracted)
    save_started_at = time.perf_counter()
    progress_interval = max(1, total_extracted // 100) if total_extracted > 0 else 1
    detail_interval = max(1, total_extracted // 25) if total_extracted > 0 else 1

    if on_detail is not None and total_extracted == 0:
        on_detail("No candidates to save after extraction")

    for idx, seg in enumerate(extracted, start=1):
        geometry = _to_valid_polygon(seg.polygon_coords)
        if geometry is None:
            if on_status is not None and (
                idx % progress_interval == 0 or idx == total_extracted
            ):
                on_status(
                    "EXTRACTING_CANDIDATES",
                    70.0 + (29.0 * idx / max(total_extracted, 1)),
                )
            continue

        # Check overlap with protected labeled segments.
        if _overlaps_with_labeled(
            geometry,
            confirmed_geoms,
            threshold=_CONFIRMED_OVERLAP_THRESHOLD,
        ) or _overlaps_with_labeled(
            geometry,
            excluded_geoms,
            threshold=_EXCLUDED_OVERLAP_THRESHOLD,
        ):
            if on_status is not None and (
                idx % progress_interval == 0 or idx == total_extracted
            ):
                on_status(
                    "EXTRACTING_CANDIDATES",
                    70.0 + (29.0 * idx / max(total_extracted, 1)),
                )
            continue

        min_x, min_y, max_x, max_y = seg.bbox_xyxy
        features = dict(seg.features) if isinstance(seg.features, dict) else {}
        features.setdefault("source_model", source_model)
        if run_identity is not None:
            # Not setdefault: the run that just produced this object is the
            # authority on which settings made it, over anything an extractor
            # happened to leave in its features.
            features[RUN_FEATURE_KEY] = dict(run_identity)
        SegmentObject.objects.create(
            segmentation=segmentation,
            geometry_wkb=shapely.to_wkb(geometry),
            centroid_x=float(seg.centroid_xy[0]),
            centroid_y=float(seg.centroid_xy[1]),
            bbox_minx=float(min_x),
            bbox_miny=float(min_y),
            bbox_maxx=float(max_x),
            bbox_maxy=float(max_y),
            label_state="CANDIDATE",
            source_model=source_model,
            confidence_score=seg.confidence_score,
            features=features,
        )
        segments_created += 1

        if on_status is not None and (
            idx % progress_interval == 0 or idx == total_extracted
        ):
            on_status(
                "EXTRACTING_CANDIDATES",
                70.0 + (29.0 * idx / max(total_extracted, 1)),
            )
        if on_detail is not None and (
            idx % detail_interval == 0 or idx == total_extracted
        ):
            elapsed = time.perf_counter() - save_started_at
            fraction = idx / max(total_extracted, 1)
            if elapsed > 0 and 0.0 < fraction < 1.0:
                eta_seconds = elapsed * (1.0 - fraction) / fraction
                on_detail(
                    f"Saving candidate shapes: {idx}/{total_extracted} ({fraction * 100.0:.0f}%, ETA ~{eta_seconds:.0f}s)"
                )
            else:
                on_detail(
                    f"Saving candidate shapes: {idx}/{total_extracted} ({fraction * 100.0:.0f}%)"
                )

    if on_status is not None:
        on_status("EXTRACTING_CANDIDATES", 99.0)
    if on_detail is not None:
        on_detail(
            f"Candidate save complete: {segments_created} saved, {total_extracted - segments_created} filtered"
        )

    if deleted_count > 0 or segments_created > 0:
        try:
            queue_full_overlay_rebuild(segmentation, source_model=source_model)
        except Exception:
            logger.warning(
                "Failed to queue overlay rebuild after extraction for %s",
                segmentation.id,
                exc_info=True,
            )

    return segments_created


def _load_geometries(queryset) -> list:
    """Materialize shapely geometries for a SegmentObject queryset."""
    geometries = []
    for wkb in queryset.values_list("geometry_wkb", flat=True):
        if not wkb:
            continue
        try:
            geometries.append(shapely.from_wkb(bytes(wkb)))
        except Exception:
            logger.warning("Skipping unreadable segment geometry", exc_info=True)
    return geometries


def _overlaps_with_labeled(
    candidate_geom, labeled_geoms: list, threshold: float = 0.8
) -> bool:
    """Check >=threshold overlap in either direction with any labeled geometry."""
    candidate_bounds = candidate_geom.bounds
    candidate_area = candidate_geom.area
    for labeled in labeled_geoms:
        try:
            # Cheap bbox reject before the exact intersection.
            lb = labeled.bounds
            if (
                lb[2] < candidate_bounds[0]
                or lb[0] > candidate_bounds[2]
                or lb[3] < candidate_bounds[1]
                or lb[1] > candidate_bounds[3]
            ):
                continue
            intersection = candidate_geom.intersection(labeled)
            if intersection.is_empty:
                continue
            inter_area = intersection.area
            if candidate_area > 0 and inter_area / candidate_area >= threshold:
                return True
            if labeled.area > 0 and inter_area / labeled.area >= threshold:
                return True
        except Exception:
            continue
    return False
