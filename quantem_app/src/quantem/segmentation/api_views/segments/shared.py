"""Shared helpers for segment API view modules."""

from __future__ import annotations

import logging
import math
import time

from django.db import OperationalError, transaction
from django.db.models import F
from django.db.models.functions import Abs
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.api_views.shared import completion_lock_response
from quantem.segmentation.bbox_policy import has_narrow_bbox
from quantem.segmentation.features.measure import MeasurementOutcome
from quantem.segmentation.geometry import extract_polygons as _extract_polygons
from quantem.segmentation.geometry.polygons import normalize_polygonal_geometry
from quantem.segmentation.geometry_serialization import (
    GEOMETRY_DETAIL_FULL,
    geometry_coords_from_polygon,
    normalize_geometry_detail,
)
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.overlay_ngff.dirty import (
    full_image_dirty_bbox,
    merge_dirty_bboxes,
)
from quantem.segmentation.overlay_ngff.mutations import (
    register_overlay_mutation,
    register_overlay_mutation_all_bundles,
)
from quantem.segmentation.segment_status import normalize_segment_status
from quantem.segmentation.selection import select_non_overlapping_inferred_segments
from quantem.segmentation.serializers.segments import (
    SegmentObjectLabelUpdateSerializer,
    SegmentObjectSerializer,
    SegmentQueryRegionSerializer,
)
from quantem.segmentation.services.confirm_batch.feature_refresh import (
    _enqueue_segment_feature_refresh,
)
from quantem.segmentation.services.confirm_batch.geometry import (
    filter_supported_confirmed_polygons,
)
from quantem.segmentation.services.confirm_batch.persistence import (
    _parse_optional_sam_score,
    _read_sam_score_from_features,
)
from quantem.segmentation.services.confirm_batch.service import (
    confirm_segment_geometries,
    register_confirmation_overlay_mutation,
)
from quantem.segmentation.services.spatial_lookup import (
    bbox_contains_point_filter,
    bbox_intersects_filter,
    make_bbox,
    make_point,
)
from quantem.segmentation.source_models import normalize_source_model, source_model_queryset_filter

# This module deliberately re-exports names for the sibling view modules in this
# package; `__all__` makes that contract explicit so an F401 autofix cannot strip
# them (it already did once).
__all__ = [n for n in dir() if not n.startswith("__")]

logger = logging.getLogger(__name__)
_MERGE_ELIGIBLE_STATES = ("CANDIDATE", "INFERRED", "CONFIRMED")
_SQLITE_LOCK_RETRY_ATTEMPTS = 3
_SQLITE_LOCK_RETRY_BASE_DELAY_SECONDS = 0.2


def _invalidate_tiles_for_segmentation(segmentation_id: str) -> None:
    del segmentation_id
    return None


def measurement_response_status(outcome: MeasurementOutcome) -> int:
    """``200`` when every edited object was measured, ``207`` when some were not.

    A geometry edit is two things at once: the outline is rewritten, and the
    numbers describing it are re-measured. The first half is committed and
    cannot be taken back without discarding what the user drew, so the response
    is not an error -- but it is not a plain success either. ``POST
    /segments/remove-area/`` used to answer ``200 {"created": 1, "updated": 1}``
    for a cut whose image could not be opened, and the only difference visible
    to the caller was in numbers it had no reason to re-read.

    207 is deliberate: it is in the 2xx range, so a client that checks
    ``response.ok`` still refreshes its overlay off the same body (the edit
    *did* happen), while a client that checks for ``200`` learns that part of
    the operation did not. The ``measurement`` block in the body names the
    objects and says what is missing.
    """
    return status.HTTP_200_OK if outcome.ok else status.HTTP_207_MULTI_STATUS


def _parse_label_states_param(raw_states: str | None) -> list[str]:
    if not raw_states:
        return []
    states_list = [state.strip() for state in raw_states.split(",")]
    valid_states = [choice[0] for choice in SegmentObject.LABEL_STATE_CHOICES]
    return [state for state in states_list if state in valid_states]


def _parse_segment_statuses_param(raw_statuses: str | None) -> list[int]:
    if not raw_statuses:
        return []
    statuses: list[int] = []
    for raw_status in str(raw_statuses).split(","):
        try:
            statuses.append(normalize_segment_status(raw_status.strip()))
        except ValueError:
            continue
    return list(dict.fromkeys(statuses))


def _parse_source_model_param(raw_source_model: str | None) -> str:
    return normalize_source_model(raw_source_model)


def _apply_segment_source_filter(queryset, source_model: str | None):
    filter_q = source_model_queryset_filter(source_model)
    if filter_q is None:
        return queryset
    return queryset.filter(filter_q)


def _is_sqlite_lock_error(exc: OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


def _run_with_sqlite_lock_retry(operation):
    for attempt in range(_SQLITE_LOCK_RETRY_ATTEMPTS):
        try:
            return operation()
        except OperationalError as exc:
            if not _is_sqlite_lock_error(exc) or attempt >= _SQLITE_LOCK_RETRY_ATTEMPTS - 1:
                raise
            delay = _SQLITE_LOCK_RETRY_BASE_DELAY_SECONDS * (attempt + 1)
            logger.warning(
                "SQLite database lock detected in segment endpoint; retrying in %.2fs (attempt %d/%d)",
                delay,
                attempt + 1,
                _SQLITE_LOCK_RETRY_ATTEMPTS,
            )
            time.sleep(delay)


def _filter_supported_polygons(polygons: list[Polygon]) -> list[Polygon]:
    return [polygon for polygon in polygons if not has_narrow_bbox(polygon.envelope)]


def parse_outline_pieces(raw_geometry: object) -> list[Polygon]:
    """Every separate area one drawn outline encloses, largest first.

    A freehand stroke that crosses itself does not enclose one area, it
    encloses several: a figure-of-eight encloses two lobes, and a stroke that
    crosses twice can enclose four. ``make_valid`` splits such a ring into a
    MultiPolygon, and **all** of its parts are real -- the user drew round every
    one of them.

    This used to end ``polygons.sort(...); return polygons[0]``, so
    ``segments/confirm-batch/`` and ``segments/remove-area/`` kept the largest
    lobe and dropped the rest with no mention of it in the response. Measured on
    a 256 px image: a figure-of-eight of two 2500 px lobes stored 2500 px and
    answered ``200 {"created": 1}``; a stroke crossing itself twice over 8750 px
    stored 2500 px; and an erase stroke of the same shape rubbed out half of
    what it was drawn round, also under a 200. Half an object's area is not a
    rounding difference, and it left no trace anywhere for the user to find.

    Returning every piece leaves the decision where it can be made honestly:
    the confirm endpoint stores each area as its own object and says in the
    response that the outline separated (:func:`separated_outlines_payload`),
    and the remove-area endpoint subtracts the union, which is the whole area
    the stroke was drawn around.

    Returns:
        The enclosed polygons, largest first, or ``[]`` when the coordinates do
        not describe a polygon at all.
    """
    if not isinstance(raw_geometry, list) or len(raw_geometry) < 3:
        return []

    coords: list[tuple[float, float]] = []
    for coord in raw_geometry:
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            return []
        try:
            coords.append((float(coord[0]), float(coord[1])))
        except (TypeError, ValueError):
            return []

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    try:
        geometry: BaseGeometry = Polygon(coords)
    except Exception:
        return []

    if not geometry.is_valid:
        # Same repair the single-outline path runs (``parse_drawn_outline``):
        # shapely rejects rings GEOS tolerated, so ``make_valid`` rather than
        # ``buffer(0)``.
        repaired = normalize_polygonal_geometry(geometry)
        if repaired is None:
            return []
        geometry = repaired

    polygons = [
        polygon
        for polygon in _extract_polygons(geometry)
        if polygon.is_valid and not polygon.is_empty and polygon.area > 0.0
    ]
    polygons.sort(key=lambda poly: float(poly.area), reverse=True)
    return polygons


def outline_geometry(pieces: list[Polygon]) -> BaseGeometry:
    """The pieces of one outline as a single geometry the services accept.

    ``confirm_segment_geometries`` runs ``extract_polygons`` over whatever it is
    given and creates one object per polygon, so a MultiPolygon here is what
    makes every lobe survive. Built directly rather than by ``unary_union``:
    lobes of a figure-of-eight meet at a point, and a union would be free to
    return them as one polygon.
    """
    if len(pieces) == 1:
        return pieces[0]
    return MultiPolygon(pieces)


def separated_outlines_payload(
    entries: list[dict[str, int]], *, merged: bool = False
) -> dict | None:
    """The ``outlines`` block: every outline that did not store as one object.

    ``None`` when each outline enclosed exactly one area and that area was
    stored, which is the ordinary case; a client that does not know about this
    block sees nothing new.

    It exists because "you drew one shape and got two objects" -- or none -- is
    a surprise worth stating in the response rather than leaving the caller to
    infer it from a ``created`` count it has no baseline for. Each entry carries
    the index of the outline as it was sent, how many separate areas it turned
    out to enclose, and how many of those were large enough to store.

    Two lists, because a client acts on them differently and an outline can be
    in both:

    ``separated``
        outlines that enclosed more than one area (``areas > 1``). The gesture
        produced more objects than it looked like it would; everything the user
        drew round is still there.
    ``dropped``
        outlines that lost something (``kept < areas``). Part or all of the
        gesture is **not** stored, so a caller that reports this as a plain
        success is reporting something that did not happen.

    ``kept == 0`` with ``areas == 1`` is the case this block was widened for.
    ``filter_supported_confirmed_polygons`` refuses a polygon spanning a pixel
    or less in either dimension, and ``confirm-batch`` used to drop it and
    answer ``200 {"created": 0}`` -- while ``POST .../segments/`` refuses the
    identical shape with a sentence. The endpoint should not be the quieter of
    the two about the same rule.

    Args:
        entries: one per outline that did not store as exactly one object.
        merged: the batch was sent with ``merge_overlaps``, so each outline was
            unioned with whatever it overlapped before the size filter ran. The
            lobes end up inside the confirmed area rather than as objects of
            their own, and no per-lobe "kept" figure is knowable in advance, so
            the wording drops that claim instead of making one up. The caller
            passing ``merged`` is expected to hand over ``kept == areas``.
    """
    if not entries:
        return None

    sentences: list[str] = []
    for entry in entries:
        index = entry["index"]
        areas = entry["areas"]
        kept = entry["kept"]
        if merged:
            sentences.append(
                f"segments[{index}] crosses itself: it encloses {areas} "
                "separate areas rather than one. All of them were merged into "
                "the confirmed area."
            )
            continue
        if areas == 1:
            # One enclosed area, and it was refused. (`areas == 1, kept == 1`
            # is the ordinary case and never reaches here.)
            sentences.append(
                # No literal "--" in a sentence the UI shows: this one reaches a
                # toast verbatim.
                f"segments[{index}] was not stored: the outline spans 1 pixel "
                "or less in one dimension, so there is no area to measure. "
                "Draw it wider, or zoom in before drawing."
            )
            continue

        sentence = (
            f"segments[{index}] crosses itself: it encloses {areas} separate "
            f"areas rather than one."
        )
        if kept == areas:
            sentence += (
                f" All {areas} were kept, each as its own object."
                if kept > 1
                else " It was kept."
            )
        elif kept == 0:
            sentence += (
                " None of them could be stored: every piece spans 1 pixel or "
                "less in one dimension."
            )
        else:
            dropped = areas - kept
            sentence += (
                f" {kept} were kept, each as its own object; {dropped} spanned "
                "1 pixel or less in one dimension and could not be stored."
            )
        sentences.append(sentence)

    return {
        "separated": [entry for entry in entries if entry["areas"] > 1],
        "dropped": [entry for entry in entries if entry["kept"] < entry["areas"]],
        "detail": " ".join(sentences),
    }


_NO_AREA_ERROR = (
    "geometry_coords enclose no area: the points lie on a single line, or the "
    "stroke doubles back exactly on itself"
)


def segmentation_image_size(segmentation: ImageSegmentation) -> tuple[int, int] | None:
    """The target image's pixel dimensions, or None when they are not recorded.

    Read off the ``Asset`` rather than by opening a rendition: this is a
    validation check on a request body, and it must not fail differently
    depending on whether the image file happens to be reachable.
    """
    asset = getattr(segmentation, "asset", None)
    try:
        width = int(getattr(asset, "logical_width", 0) or 0)
        height = int(getattr(asset, "logical_height", 0) or 0)
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def parse_drawn_outline(
    raw_coords: object,
    *,
    image_size: tuple[int, int] | None,
) -> tuple[Polygon | None, str]:
    """One drawn outline as a storable ``Polygon``, or the reason it is not one.

    ``SegmentObject.save`` repairs geometry through
    :func:`~quantem.segmentation.geometry.fields.repair_geometry`, which raises
    when a repair changes the geometry type -- a self-crossing lasso whose
    ``make_valid`` splits into a MultiPolygon is the common case, and it is a
    gesture a person can make with one careless stroke. Nothing caught that
    ``ValueError`` on the create view, so DRF turned an ordinary drawing mistake
    into an HTTP 500 with a Django traceback, while every other bad geometry on
    the same view came back as a 400 with a sentence. This runs the same repair
    first, so the model and the view can never disagree about what is storable.

    The bounds check is the second half. A polygon at ``(1e12, 1e12)``, or one
    entirely at negative coordinates, was stored as a **confirmed** object and
    then could not be measured -- it covers no pixel of the image -- so it
    reached ``objects.csv`` as a row of empty morphometrics. There is no reading
    of an outline outside the image that is a real object, so it is refused
    instead of stored.

    Returns:
        ``(polygon, "")`` when the outline is usable, otherwise
        ``(None, message)`` with a sentence fit to put in a 400.
    """
    if not isinstance(raw_coords, list) or len(raw_coords) < 3:
        return None, "geometry_coords must be a list of at least 3 points"

    coords: list[tuple[float, float]] = []
    for coord in raw_coords:
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            return None, "geometry_coords must be [x, y] pairs"
        try:
            x, y = float(coord[0]), float(coord[1])
        except (TypeError, ValueError):
            return None, "geometry_coords must be [x, y] pairs of numbers"
        if not (math.isfinite(x) and math.isfinite(y)):
            return None, (
                "geometry_coords must be finite numbers: a NaN or infinite "
                "coordinate has no position in the image"
            )
        coords.append((x, y))

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    try:
        polygon = Polygon(coords)
    except Exception:
        return None, "geometry_coords do not describe a polygon"

    if not polygon.is_valid:
        repaired = normalize_polygonal_geometry(polygon)
        pieces = _extract_polygons(repaired) if repaired is not None else []
        if not pieces:
            # The repair produced no polygon at all -- a collinear ring becomes
            # a LineString. Nothing crossed; there was never an area.
            return None, _NO_AREA_ERROR
        if len(pieces) > 1:
            return None, (
                f"This outline crosses itself and separates into {len(pieces)} "
                "pieces, which cannot be stored as one object. Redraw it "
                "without crossing the path, or send it to "
                "segments/confirm-batch/, which stores each of the "
                f"{len(pieces)} enclosed areas as its own object."
            )
        polygon = pieces[0]

    if polygon.is_empty or polygon.area <= 0.0:
        return None, _NO_AREA_ERROR

    if image_size is not None:
        width, height = image_size
        if not polygon.intersects(make_bbox(0.0, 0.0, float(width), float(height))):
            min_x, min_y, max_x, max_y = polygon.bounds
            return None, (
                f"This outline lies entirely outside the image: it spans "
                f"x {min_x:g}..{max_x:g}, y {min_y:g}..{max_y:g}, and the image "
                f"is {width}x{height} pixels. Nothing there can be measured."
            )

    return polygon, ""


def _geometries_overlap(left: BaseGeometry, right: BaseGeometry) -> bool:
    try:
        return bool(left.intersects(right))
    except Exception:
        return False


def _geometry_coords_from_polygon(geometry: BaseGeometry | None) -> list[list[float]]:
    return geometry_coords_from_polygon(geometry)
