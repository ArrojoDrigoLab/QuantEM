"""Named-object editing API for image-specific analysis masks."""

from __future__ import annotations

import re
import secrets

from django.db import transaction
from django.db.models import Max
from django.shortcuts import get_object_or_404
from django.urls import path
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.api_views.segments.shared import (
    outline_geometry,
    parse_outline_pieces,
)
from quantem.segmentation.geometry.polygons import normalize_polygonal_geometry
from quantem.segmentation.geometry_serialization import geometry_payload
from quantem.segmentation.global_masks import save_global_mask_from_geometries
from quantem.segmentation.models import AnalysisMaskObject, ImageSegmentation
from quantem.segmentation.overlay_ngff.dirty import full_image_dirty_bbox
from quantem.segmentation.overlay_ngff.mutations import (
    register_overlay_mutation_all_bundles,
)
from quantem.segmentation.services.confirm_batch.geometry import (
    safe_difference,
    safe_union,
)
from quantem.segmentation.type_definitions import ANALYSIS_MASK

_OBJECT_NAME = re.compile(r"^Object\s+(\d+)$", re.IGNORECASE)
_OBJECT_COLORS = (
    "#38bdf8",
    "#22c55e",
    "#f97316",
    "#a855f7",
    "#f43f5e",
    "#eab308",
    "#14b8a6",
    "#6366f1",
    "#ec4899",
    "#84cc16",
)


def _analysis_segmentation(seg_id) -> ImageSegmentation:
    segmentation = get_object_or_404(
        ImageSegmentation.objects.select_related("asset", "segmentation_type"),
        id=seg_id,
    )
    if segmentation.segmentation_type.internal_name != ANALYSIS_MASK.internal_name:
        raise ValueError("Named analysis-mask objects belong only to an Analysis Mask.")
    return segmentation


def _object_payload(obj: AnalysisMaskObject) -> dict:
    return {
        "id": str(obj.id),
        "segmentation": str(obj.segmentation_id),
        "name": obj.name,
        "color": obj.color,
        "sort_order": int(obj.sort_order),
        "geometry": geometry_payload(obj.geometry),
        "created_at": obj.created_at,
        "updated_at": obj.updated_at,
    }


def _next_object_name(segmentation: ImageSegmentation) -> str:
    largest = 0
    for name in AnalysisMaskObject.objects.filter(segmentation=segmentation).values_list(
        "name", flat=True
    ):
        match = _OBJECT_NAME.fullmatch(str(name).strip())
        if match:
            largest = max(largest, int(match.group(1)))
    return f"Object {largest + 1}"


def _next_color(segmentation: ImageSegmentation) -> str:
    used = set(
        AnalysisMaskObject.objects.filter(segmentation=segmentation).values_list("color", flat=True)
    )
    available = [color for color in _OBJECT_COLORS if color not in used]
    return secrets.choice(available or list(_OBJECT_COLORS))


def _parse_shapes(segmentation: ImageSegmentation, raw_shapes: object) -> BaseGeometry | None:
    if not isinstance(raw_shapes, list) or not raw_shapes:
        return None

    merged: BaseGeometry | None = None
    for index, raw_shape in enumerate(raw_shapes):
        if not isinstance(raw_shape, dict):
            raise ValueError(f"shapes[{index}] must be an object.")
        raw_rings = raw_shape.get("rings")
        if not isinstance(raw_rings, list) or not raw_rings:
            raise ValueError(f"shapes[{index}].rings must contain an exterior ring.")

        exterior_pieces = parse_outline_pieces(raw_rings[0])
        geometry = outline_geometry(exterior_pieces) if exterior_pieces else None
        if geometry is None:
            raise ValueError(f"shapes[{index}] encloses no usable area.")
        for ring_index, raw_hole in enumerate(raw_rings[1:], start=1):
            hole_pieces = parse_outline_pieces(raw_hole)
            if not hole_pieces:
                raise ValueError(f"shapes[{index}].rings[{ring_index}] encloses no usable area.")
            geometry = safe_difference(geometry, outline_geometry(hole_pieces))
            if geometry is None:
                break
        merged = safe_union(merged, geometry)

    if merged is None:
        return None
    width = int(segmentation.asset.logical_width or 0)
    height = int(segmentation.asset.logical_height or 0)
    if width <= 0 or height <= 0:
        raise ValueError("The image dimensions are unavailable.")
    try:
        return normalize_polygonal_geometry(merged.intersection(box(0, 0, width, height)))
    except Exception as exc:
        raise ValueError("The drawn shape could not be clipped to the image.") from exc


def _rebuild(segmentation: ImageSegmentation):
    geometries = []
    for obj in AnalysisMaskObject.objects.filter(segmentation=segmentation).iterator():
        geometry = obj.geometry
        if geometry is not None and not geometry.is_empty:
            geometries.append(geometry)
    return save_global_mask_from_geometries(
        segmentation,
        geometries,
        source="manual",
        metadata={"editable_analysis_objects": True},
    )


def _foreground_pixels(segmentation: ImageSegmentation) -> int:
    try:
        return int(segmentation.global_mask.foreground_pixels)
    except Exception:
        return 0


def _register_overlay(segmentation: ImageSegmentation):
    return register_overlay_mutation_all_bundles(
        segmentation,
        dirty_bbox=full_image_dirty_bbox(segmentation),
        force_full_rebuild=True,
    )


class AnalysisMaskObjectListCreateView(APIView):
    """List objects, or atomically apply one Include/Exclude edit."""

    def get(self, request, seg_id):
        del request
        try:
            segmentation = _analysis_segmentation(seg_id)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        objects = AnalysisMaskObject.objects.filter(segmentation=segmentation)
        return Response(
            {
                "objects": [_object_payload(obj) for obj in objects],
                "foreground_pixels": _foreground_pixels(segmentation),
            },
            status=status.HTTP_200_OK,
        )

    def post(self, request, seg_id):
        try:
            segmentation = _analysis_segmentation(seg_id)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        operation = str(request.data.get("operation") or "include").strip().lower()
        if operation not in {"include", "exclude"}:
            return Response(
                {"error": "operation must be include or exclude."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        object_id = request.data.get("object_id")
        if not object_id and operation == "exclude":
            return Response(
                {"error": "A new object starts in Include mode; there is nothing to exclude yet."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            incoming = _parse_shapes(segmentation, request.data.get("shapes"))
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        if incoming is None or incoming.is_empty:
            return Response(
                {"error": "The drawn shape encloses no area inside the image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        with transaction.atomic():
            if object_id:
                obj = get_object_or_404(
                    AnalysisMaskObject.objects.select_for_update(),
                    id=object_id,
                    segmentation=segmentation,
                )
            else:
                last_order = (
                    AnalysisMaskObject.objects.filter(segmentation=segmentation).aggregate(
                        value=Max("sort_order")
                    )["value"]
                    or 0
                )
                obj = AnalysisMaskObject(
                    segmentation=segmentation,
                    name=_next_object_name(segmentation),
                    color=_next_color(segmentation),
                    sort_order=int(last_order) + 1,
                )

            current = obj.geometry
            if operation == "include":
                result = safe_union(current, incoming)
            else:
                result = safe_difference(current, incoming) if current is not None else None
            obj.geometry = result
            obj.save()

            # Editing a mask is an explicit unlock. The dedicated page never
            # marks masks complete, but this makes masks saved by an older UI
            # editable instead of silently refusing every operation.
            update_fields = []
            if segmentation.status_stage == "COMPLETED":
                segmentation.status_stage = "CANDIDATES_READY"
                update_fields.append("status_stage")
            if segmentation.final_result_provenance:
                segmentation.final_result_provenance = {}
                update_fields.append("final_result_provenance")
            if update_fields:
                segmentation.save(update_fields=update_fields)
        # WKB is the editing representation. Do not rewrite a full-image PNG
        # or rebuild its pyramid for every brush stroke: that was the long
        # pause after R. Object Save (and the page-level Save Analysis Masks)
        # materializes the union once through the save endpoint below.
        return Response(
            {
                "object": _object_payload(obj),
                "foreground_pixels": _foreground_pixels(segmentation),
                "overlay": None,
            },
            status=status.HTTP_201_CREATED if not object_id else status.HTTP_200_OK,
        )


class AnalysisMaskObjectDetailView(APIView):
    """Rename or delete one analysis-mask object."""

    def _get(self, seg_id, object_id):
        segmentation = _analysis_segmentation(seg_id)
        obj = get_object_or_404(
            AnalysisMaskObject,
            id=object_id,
            segmentation=segmentation,
        )
        return segmentation, obj

    def patch(self, request, seg_id, object_id):
        try:
            _segmentation, obj = self._get(seg_id, object_id)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        name = str(request.data.get("name") or "").strip()
        if not name:
            return Response(
                {"error": "Object name cannot be empty."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if len(name) > 100:
            return Response(
                {"error": "Object name must be 100 characters or fewer."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        obj.name = name
        obj.save(update_fields=["name", "updated_at"])
        return Response(_object_payload(obj), status=status.HTTP_200_OK)

    def delete(self, request, seg_id, object_id):
        del request
        try:
            segmentation, obj = self._get(seg_id, object_id)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        deleted_id = str(obj.id)
        obj.delete()
        return Response(
            {
                "deleted_id": deleted_id,
                "foreground_pixels": _foreground_pixels(segmentation),
                "overlay": None,
            },
            status=status.HTTP_200_OK,
        )


class AnalysisMaskObjectSaveView(APIView):
    """Materialize every named object into the one analysis binary mask."""

    def post(self, request, seg_id):
        del request
        try:
            segmentation = _analysis_segmentation(seg_id)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        with transaction.atomic():
            record = _rebuild(segmentation)
            if segmentation.status_stage != "CANDIDATES_READY":
                segmentation.status_stage = "CANDIDATES_READY"
                segmentation.save(update_fields=["status_stage"])
        overlay = _register_overlay(segmentation)
        return Response(
            {
                "objects": [
                    _object_payload(obj)
                    for obj in AnalysisMaskObject.objects.filter(segmentation=segmentation)
                ],
                "foreground_pixels": int(record.foreground_pixels),
                "overlay": overlay,
            },
            status=status.HTTP_200_OK,
        )


urlpatterns = [
    path(
        "segmentations/<uuid:seg_id>/analysis-mask-objects/",
        AnalysisMaskObjectListCreateView.as_view(),
        name="analysis-mask-object-list-create",
    ),
    path(
        "segmentations/<uuid:seg_id>/analysis-mask-objects/save/",
        AnalysisMaskObjectSaveView.as_view(),
        name="analysis-mask-object-save",
    ),
    path(
        "segmentations/<uuid:seg_id>/analysis-mask-objects/<uuid:object_id>/",
        AnalysisMaskObjectDetailView.as_view(),
        name="analysis-mask-object-detail",
    ),
]
