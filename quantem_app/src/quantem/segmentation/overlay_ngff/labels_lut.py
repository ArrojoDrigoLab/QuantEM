"""Dense-label allocation and render-time colour/state LUT.

The on-disk ``labels`` raster stores a *dense label* per pixel. This module
owns everything about what a label *means*:

* :func:`resolve_object_style` -- maps a live ``SegmentObject`` to its
  ``(priority, state, colour)`` (used both for build paint-order and for LUT
  colours).
* :func:`bundle_queryset` -- the set of objects a bundle renders.
* :func:`allocate_labels` / :func:`replace_bundle_labels` -- dense-label
  bookkeeping in :class:`SegmentationOverlayLabel`.
* :func:`build_label_lut_binary` / :func:`build_label_lut_json` -- the
  render-time LUT served to the client. Because the label -> object mapping is
  stable across state changes, confirming/recolouring an object only changes
  what these functions emit -- never the raster.
"""

from __future__ import annotations

from typing import Any

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
from quantem.segmentation.source_models import source_model_queryset_filter

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
    """Return the queryset of objects this bundle renders.

    Membership rule: ``CONFIRMED OR manual OR <this source model>``.
    """
    queryset = SegmentObject.objects.filter(segmentation=segmentation)
    source_filter = source_model_queryset_filter(source_model)
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
    """Replace all label rows for a bundle (used on full rebuild).

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


def existing_label_map(state: SegmentationOverlayState) -> dict[Any, int]:
    return {
        row.object_uuid: row.label
        for row in SegmentationOverlayLabel.objects.filter(overlay_state=state).only(
            "object_uuid", "label"
        )
    }


def allocate_labels(
    state: SegmentationOverlayState,
    *,
    new_objects: list[Any],
) -> dict[Any, int]:
    """Assign dense labels to newly created objects, reusing freed gaps.

    Returns a ``{object_uuid: label}`` map for the supplied objects. Existing
    objects keep their labels (full rebuild is the only renumbering event).
    """
    if not new_objects:
        return {}
    used = set(
        SegmentationOverlayLabel.objects.filter(overlay_state=state).values_list("label", flat=True)
    )
    assigned: dict[Any, int] = {}
    rows: list[SegmentationOverlayLabel] = []
    candidate = 1
    for object_uuid in new_objects:
        while candidate in used:
            candidate += 1
        used.add(candidate)
        assigned[object_uuid] = candidate
        rows.append(
            SegmentationOverlayLabel(
                overlay_state=state,
                label=candidate,
                object_uuid=object_uuid,
            )
        )
    SegmentationOverlayLabel.objects.bulk_create(rows, batch_size=2000)
    return assigned


def remove_labels_for_objects(
    state: SegmentationOverlayState,
    *,
    object_uuids: list[Any],
) -> None:
    if not object_uuids:
        return
    SegmentationOverlayLabel.objects.filter(
        overlay_state=state, object_uuid__in=object_uuids
    ).delete()


# ---------------------------------------------------------------------------
# Render-time LUT
# ---------------------------------------------------------------------------
def _resolve_live_objects(
    state: SegmentationOverlayState,
) -> tuple[list[SegmentationOverlayLabel], dict[Any, SegmentObject]]:
    rows = list(
        SegmentationOverlayLabel.objects.filter(overlay_state=state).only("label", "object_uuid")
    )
    object_ids = [row.object_uuid for row in rows]
    objects: dict[Any, SegmentObject] = {}
    if object_ids:
        for obj in SegmentObject.objects.filter(id__in=object_ids).only(
            "id", "label_state", "refined", "status", "source_model"
        ):
            objects[obj.id] = obj
    return rows, objects


def candidate_ids_inside_completed_rois(state: SegmentationOverlayState) -> set[Any]:
    """Ids of non-confirmed segments whose centroid falls in a CompletedROI.

    Inside an area the user has marked complete their labels are authoritative,
    so model candidates there are hidden -- otherwise predicted candidates draw
    over/next to the confirmed objects in a region the user already finished.
    Centroid (not geometry) membership is used so an object is judged by where it
    sits, rather than hiding candidates that merely graze the ROI border.

    Only ever *hides*; the raster and label->object mapping are untouched, so this
    re-evaluates for free whenever ROIs change (a LUT revision, not a rebuild).

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
        SegmentObject.objects.filter(segmentation_id=segmentation_id)
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
) -> tuple[bytes, int]:
    """Return ``(rgba_bytes, max_label)``.

    ``rgba_bytes`` is a flat ``(max_label + 1, 4)`` uint8 array: per dense label,
    the fill colour + alpha (alpha 0 = hidden). Label 0 is background
    (transparent). The client uploads this directly as the LUT texture.

    ``hidden_states`` forces alpha 0 for any label in those states (on top of the
    default per-state visibility) -- used to serve a confirmed-only LUT to the
    right (review) panel without a second on-disk artifact.
    """
    if state.segmentation.segmentation_type.measurement_mode == "global":
        color = str(state.segmentation.segmentation_type.default_color or "33CC66").lstrip("#")
        try:
            red, green, blue = _hex_to_rgb(color)
        except (ValueError, IndexError):
            red, green, blue = _hex_to_rgb(COLOR_CONFIRMED)
        visible = STATE_CONFIRMED not in hidden_states
        buffer = np.zeros((2, 4), dtype=np.uint8)
        buffer[1] = (red, green, blue, 255 if visible else 0)
        return buffer.tobytes(), 1

    rows, objects = _resolve_live_objects(state)
    suppressed = candidate_ids_inside_completed_rois(state)
    max_label = max((row.label for row in rows), default=0)
    buffer = np.zeros((max_label + 1, 4), dtype=np.uint8)
    for row in rows:
        obj = objects.get(row.object_uuid)
        if obj is None:
            continue
        _, state_key, color_hex = resolve_object_style(obj)
        red, green, blue = _hex_to_rgb(color_hex)
        visible = STATE_DEFAULT_VISIBLE.get(state_key, True) and state_key not in hidden_states
        if row.object_uuid in suppressed:
            visible = False
        buffer[row.label] = (red, green, blue, 255 if visible else 0)
    return buffer.tobytes(), int(max_label)


def build_label_lut_json(state: SegmentationOverlayState) -> dict[str, Any]:
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
    rows, objects = _resolve_live_objects(state)
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
