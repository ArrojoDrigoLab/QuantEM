"""Dense-label allocation and render-time colour/state LUT.

The on-disk ``labels`` raster stores a *dense label* per pixel. This module
owns everything about what a label *means*:

* :func:`resolve_object_style` -- maps a live ``SegmentObject`` to its
  ``(priority, state, colour)`` (used both for build paint-order and for LUT
  colours).
* :func:`bundle_queryset` -- the set of objects a bundle renders.
* :func:`replace_bundle_labels` -- dense-label
  bookkeeping in :class:`SegmentationOverlayLabel`.
* :func:`build_label_lut_binary` / :func:`build_label_lut_json` -- the
  render-time LUT served to the client. Because the label -> object mapping is
  stable across state changes, confirming/recolouring an object only changes
  what these functions emit. Contested overlaps also need a priority repaint.
"""

from __future__ import annotations

from typing import Any, NamedTuple
from uuid import UUID

import numpy as np
from django.db.models import QuerySet

from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    SegmentationOverlayLabel,
    SegmentationOverlayState,
    SegmentObject,
)
from quantem.segmentation.services.spatial_lookup import (
    centroid_in_bbox_filter,
    union_geometries,
)
from quantem.segmentation.source_models import overlay_bundle_source_filter

from .constants import (
    COLOR_CANDIDATE,
    COLOR_CONFIRMED,
    COLOR_EXCLUDED,
    PRIORITY_CANDIDATE,
    PRIORITY_CONFIRMED,
    PRIORITY_EXCLUDED,
    PRIORITY_MANUAL,
    PRIORITY_REFINED,
    STATE_CANDIDATE,
    STATE_CONFIRMED,
    STATE_DEFAULT_VISIBLE,
    STATE_EXCLUDED,
)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _base_style(label_state: object) -> tuple[int, str, str]:
    if label_state == "CONFIRMED":
        return (PRIORITY_CONFIRMED, STATE_CONFIRMED, COLOR_CONFIRMED)
    if label_state == "EXCLUDED":
        return (PRIORITY_EXCLUDED, STATE_EXCLUDED, COLOR_EXCLUDED)
    # CANDIDATE / INFERRED / unknown -> candidate
    return (PRIORITY_CANDIDATE, STATE_CANDIDATE, COLOR_CANDIDATE)


def resolve_object_style(obj: SegmentObject) -> tuple[int, str, str]:
    """Return ``(priority, state_key, colour_hex)`` for a live object.

    ``priority`` arbitrates which object wins a contested pixel (higher wins).
    ``state_key`` drives default visibility. ``colour_hex`` is the fill colour;
    the border colour is derived from it on the GPU.
    """
    priority, state_key, color = _base_style(getattr(obj, "label_state", None))
    refined = str(getattr(obj, "refined", "UNREFINED"))
    if refined == "MANUAL":
        return (PRIORITY_MANUAL, state_key, color)
    if refined == "AUTOMATIC":
        return (max(priority, PRIORITY_REFINED), state_key, color)
    return (priority, state_key, color)


def bundle_queryset(
    segmentation: ImageSegmentation,
    source_model: str | None,
) -> QuerySet:
    """Return the queryset of objects this bundle rasterises.

    The source-less bundle contains every object and is rendered with a
    confirmed-only LUT.  A named bundle contains that model's objects plus the
    hand-drawn ones, and is rendered with the candidate LUT.  Both retain
    objects across a state flip, making ordinary Confirm a LUT-only operation.

    This uses ``overlay_bundle_source_filter``, not the wider
    ``source_model_queryset_filter`` that the query endpoints use: bundle
    membership excludes *another model's* confirmed objects, which already have
    a home in the source-less confirmed display and must not be duplicated into
    every model comparison layer.  Hand-drawn objects are the exception and are
    kept, because the candidate layer reads this bundle and it is the only
    layer that draws an unconfirmed outline at all -- see
    :func:`~quantem.segmentation.source_models.overlay_bundle_source_filter`
    for the R13 failure that rule exists to prevent.
    """
    queryset = SegmentObject.objects.filter(segmentation=segmentation, superseded_at__isnull=True)
    source_filter = overlay_bundle_source_filter(source_model)
    if source_filter is not None:
        queryset = queryset.filter(source_filter)
    return queryset


# ---------------------------------------------------------------------------
# Dense-label bookkeeping
# ---------------------------------------------------------------------------
def replace_bundle_labels(
    state: SegmentationOverlayState,
    *,
    assignments: list[tuple[int, Any]],
) -> None:
    """Replace all label rows for a bundle (published with each snapshot).

    ``assignments`` is ``[(label, object_uuid), ...]``.
    """
    SegmentationOverlayLabel.objects.filter(overlay_state=state).delete()
    SegmentationOverlayLabel.objects.bulk_create(
        [
            SegmentationOverlayLabel(
                overlay_state=state,
                label=label,
                object_uuid=object_uuid,
            )
            for label, object_uuid in assignments
        ],
        batch_size=2000,
    )


def existing_label_map(state: SegmentationOverlayState, *, current_only=False) -> dict[Any, int]:
    # Do not express this as an IN subquery. Immediately after a bulk import,
    # PostgreSQL can estimate one row on both sides and choose a quadratic
    # nested loop (tens of seconds for 20k objects). Two indexed scans and a
    # Python hash intersection remain linear even before ANALYZE runs.
    rows = SegmentationOverlayLabel.objects.filter(overlay_state=state).values_list(
        "object_uuid", "label"
    )
    labels = dict(rows.iterator(chunk_size=2000))
    if not current_only or not labels:
        return labels
    return {
        object_id: labels[object_id]
        for object_id in bundle_queryset(state.segmentation, state.candidate_source_model)
        .values_list("id", flat=True)
        .iterator(chunk_size=2000)
        if object_id in labels
    }


def remove_obsolete_labels(state: SegmentationOverlayState, assignments) -> None:
    """Reclaim obsolete identities without another statistics-sensitive join."""
    live = {object_id for _, object_id in assignments}
    obsolete = []
    for row_id, object_id in (
        SegmentationOverlayLabel.objects.filter(overlay_state=state)
        .values_list("id", "object_uuid")
        .iterator(chunk_size=2000)
    ):
        if object_id not in live:
            obsolete.append(row_id)
    for start in range(0, len(obsolete), 1000):
        SegmentationOverlayLabel.objects.filter(pk__in=obsolete[start : start + 1000]).delete()


# ---------------------------------------------------------------------------
# Render-time LUT
# ---------------------------------------------------------------------------
class _LabelIdentity(NamedTuple):
    label: int
    object_uuid: UUID


class _LiveStyle(NamedTuple):
    id: UUID
    label_state: str
    refined: str


def _resolve_live_objects(
    state: SegmentationOverlayState,
    assignments=None,
) -> tuple[list[_LabelIdentity], dict[Any, _LiveStyle]]:
    rows = (
        [
            _LabelIdentity(label=int(label), object_uuid=UUID(str(object_id)))
            for label, object_id in assignments
        ]
        if assignments is not None
        else [
            _LabelIdentity(*row)
            for row in SegmentationOverlayLabel.objects.filter(overlay_state=state)
            .values_list("label", "object_uuid")
            .iterator(chunk_size=2000)
        ]
    )
    object_ids = tuple({row.object_uuid for row in rows})
    objects: dict[Any, _LiveStyle] = {}
    if object_ids:
        # Bound SQL parameters without scanning every object in a large
        # segmentation when this bundle contains only a small subset.
        for start in range(0, len(object_ids), 1000):
            for obj in (
                SegmentObject.objects.filter(
                    id__in=object_ids[start : start + 1000],
                    segmentation_id=state.segmentation_id,
                    superseded_at__isnull=True,
                )
                .values_list("id", "label_state", "refined")
                .iterator(chunk_size=1000)
            ):
                objects[obj[0]] = _LiveStyle(*obj)
    return rows, objects


def candidate_ids_inside_completed_rois(state: SegmentationOverlayState) -> set[Any]:
    """Ids of non-confirmed segments whose centroid falls in a CompletedROI.

    Inside an area the user has marked complete their labels are authoritative,
    so model candidates there are hidden -- otherwise predicted candidates draw
    over/next to the confirmed objects in a region the user already finished.
    Centroid (not geometry) membership is used so an object is judged by where it
    sits, rather than hiding candidates that merely graze the ROI border.

    Only ever *hides*; the raster and label->object mapping are untouched, so this
    re-evaluates on a new LUT revision instead of rebuilding pixels. HTTP
    palette caching shares this calculation across viewers and response formats.

    The GeoDjango ``Union`` aggregate + ``centroid__intersects`` lookup is done in
    Python here: a shapely union of the fetched ROI polygons, then a centroid
    bbox prefilter narrowed by an exact ``contains`` test.
    """
    segmentation_id = state.segmentation_id
    roi_rows = list(
        CompletedROI.objects.filter(segmentation_id=segmentation_id).only("geometry_wkb")
    )
    if not roi_rows:
        return set()
    roi_union = union_geometries(row.geometry for row in roi_rows)
    if roi_union is None:
        return set()

    prefilter = centroid_in_bbox_filter(roi_union)
    if prefilter is None:
        return set()
    candidates = (
        SegmentObject.objects.filter(segmentation_id=segmentation_id, superseded_at__isnull=True)
        .filter(prefilter)
        .exclude(label_state="CONFIRMED")
        .only("id", "centroid_x", "centroid_y")
    )
    suppressed: set[Any] = set()
    for candidate in candidates.iterator():
        centroid = candidate.centroid
        if centroid is None:
            continue
        if roi_union.contains(centroid):
            suppressed.add(candidate.id)
    return suppressed


def build_label_lut_binary(
    state: SegmentationOverlayState,
    *,
    hidden_states: frozenset[str] = frozenset(),
    assignments=None,
) -> tuple[bytes, int]:
    """Return ``(rgba_bytes, max_label)``.

    ``rgba_bytes`` is a flat ``(max_label + 1, 4)`` uint8 array: per dense label,
    the fill colour + alpha (alpha 0 = hidden). Label 0 is background
    (transparent). The client uploads this directly as the LUT texture.

    ``hidden_states`` forces alpha 0 for any label in those states (on top of the
    default per-state visibility) -- used to serve a confirmed-only LUT to the
    right (review) panel without a second on-disk artifact.
    """
    return binary_from_json(
        build_label_lut_json(state, assignments=assignments), hidden_states=hidden_states
    )


def build_label_lut_json(state: SegmentationOverlayState, *, assignments=None) -> dict[str, Any]:
    """Return the label -> object map for picking and client-side toggles."""
    if state.segmentation.segmentation_type.measurement_mode == "global":
        color = str(state.segmentation.segmentation_type.default_color or "#33CC66")
        return {
            "lut_revision": int(state.lut_revision),
            "bundle_version": int(state.bundle_version),
            "max_label": 1,
            "overlay_kind": "binary_mask",
            "pickable": False,
            "color": color,
            "objects": [],
        }
    rows, objects = _resolve_live_objects(state, assignments)
    suppressed = candidate_ids_inside_completed_rois(state)
    entries: list[dict[str, Any]] = []
    for row in rows:
        obj = objects.get(row.object_uuid)
        if obj is None:
            continue
        _, state_key, color_hex = resolve_object_style(obj)
        entry = {
            "label": int(row.label),
            "uuid": str(row.object_uuid),
            "state": state_key,
            "color": color_hex,
        }
        if row.object_uuid in suppressed:
            # Hidden by a CompletedROI: keep the mapping (so picking/debug still
            # resolve) but tell the client not to draw it.
            entry["hidden_by_completed_roi"] = True
        entries.append(entry)
    return {
        "lut_revision": int(state.lut_revision),
        "bundle_version": int(state.bundle_version),
        "max_label": max((row.label for row in rows), default=0),
        "objects": entries,
    }


def binary_from_json(payload, *, hidden_states=frozenset()):
    """Derive the binary palette from the same cached visibility/picking rows."""
    maximum = int(payload["max_label"])
    buffer = np.zeros((maximum + 1, 4), dtype=np.uint8)
    if payload.get("overlay_kind") == "binary_mask":
        try:
            color = _hex_to_rgb(payload["color"].lstrip("#"))
        except (ValueError, IndexError):
            color = _hex_to_rgb(COLOR_CONFIRMED)
        buffer[1] = (*color, 0 if STATE_CONFIRMED in hidden_states else 255)
    else:
        for row in payload["objects"]:
            visible = (
                STATE_DEFAULT_VISIBLE.get(row["state"], True)
                and row["state"] not in hidden_states
                and not row.get("hidden_by_completed_roi", False)
            )
            buffer[row["label"]] = (*_hex_to_rgb(row["color"].lstrip("#")), 255 if visible else 0)
    return buffer.tobytes(), maximum
