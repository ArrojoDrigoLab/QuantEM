"""
Non-overlap greedy selection for inferred segments.

This module provides functions to select a maximal non-overlapping set of
inferred segments based on regression scores, using a simple greedy algorithm.
"""


from django.db.models import Q
from shapely.geometry.base import BaseGeometry

from .models import ImageSegmentation, SegmentObject
from .source_models import source_model_queryset_filter


def bbox_overlap_filter(geometry: BaseGeometry, prefix: str = "bbox") -> Q:
    """Rows whose stored bbox overlaps ``geometry``'s bounds.

    This is the SQLite replacement for ``bbox__intersects=geom``: a pure numeric
    range filter over the indexed bbox columns. It is a *prefilter* -- callers
    that need true intersection must refine the survivors with shapely.
    """
    min_x, min_y, max_x, max_y = geometry.bounds
    return Q(
        **{
            f"{prefix}_maxx__gte": min_x,
            f"{prefix}_minx__lte": max_x,
            f"{prefix}_maxy__gte": min_y,
            f"{prefix}_miny__lte": max_y,
        }
    )


def select_non_overlapping_inferred_segments(
    segmentation: ImageSegmentation,
    threshold: float,
    viewport_geom: BaseGeometry | None = None,
    max_candidates: int = 1000,
    source_model: str | None = None,
) -> list[SegmentObject]:
    """
    Select a maximal non-overlapping set of inferred segments with confidence_score >= threshold.

    Uses a greedy algorithm: segments are sorted by confidence_score (descending),
    and each segment is accepted only if it does not intersect any previously accepted segment.

    Without a spatial database the candidates are fetched once and the overlap
    test runs in-process:
    a bounds-overlap prefilter against the already-accepted segments (what the
    spatial index did) followed by an exact shapely ``intersects``.

    Args:
        segmentation: The ImageSegmentation to select segments from
        threshold: Minimum confidence_score threshold (0.0 to 1.0)
        viewport_geom: Optional viewport geometry to restrict selection to segments
                      whose bbox intersects this viewport
        max_candidates: Maximum number of candidates to consider (default 1000)
    Returns:
        List of SegmentObject instances that are non-overlapping, sorted by
        confidence_score (descending). Empty list if no segments meet criteria.
    """
    # Build initial queryset
    qs = SegmentObject.objects.filter(
        segmentation=segmentation,
        label_state__in=["INFERRED", "CANDIDATE"],
        confidence_score__isnull=False,
        confidence_score__gte=threshold,
    )
    source_filter = source_model_queryset_filter(source_model)
    if source_filter is not None:
        qs = qs.filter(source_filter)
    # Filter by viewport if provided
    if viewport_geom is not None:
        qs = qs.filter(bbox_overlap_filter(viewport_geom))

    # Order by confidence_score descending and limit candidates
    qs = qs.order_by("-confidence_score")[:max_candidates]

    # Greedy selection; convert queryset to a list to avoid multiple database hits.
    candidates = list(qs)

    if not candidates:
        return []

    selected: list[SegmentObject] = []
    accepted: list[tuple[tuple[float, float, float, float], BaseGeometry]] = []

    for candidate in candidates:
        geometry = candidate.geometry
        if geometry is None or geometry.is_empty:
            continue
        min_x, min_y, max_x, max_y = geometry.bounds

        intersects = False
        for (other_minx, other_miny, other_maxx, other_maxy), other in accepted:
            if (
                other_maxx < min_x
                or other_minx > max_x
                or other_maxy < min_y
                or other_miny > max_y
            ):
                continue
            if other.intersects(geometry):
                intersects = True
                break

        if not intersects:
            selected.append(candidate)
            accepted.append(((min_x, min_y, max_x, max_y), geometry))

            # Early exit if we've selected enough segments (5000 max, frontend will cache and filter)
            if len(selected) >= 5000:
                break

    return selected
