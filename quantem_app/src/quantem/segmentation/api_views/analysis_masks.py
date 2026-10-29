"""Named-object editing API for image-specific analysis masks."""

from __future__ import annotations

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
from quantem.segmentation.area_instances import (
    AREA_INSTANCE_COLORS,
    default_instance_name,
    materialize_legacy_area_instances,
)
from quantem.segmentation.geometry.polygons import normalize_polygonal_geometry
from quantem.segmentation.geometry_serialization import geometry_payload
from quantem.segmentation.global_masks import save_global_mask_from_geometries
from quantem.segmentation.models import (
    AnalysisMaskObject,
    AnalysisMaskObjectAnnotation,
    GlobalMask,
    ImageSegmentation,
)
from quantem.segmentation.mutations import (
    record_change,
    register_overlay_mutation_all_bundles,
)
from quantem.segmentation.overlay_ngff.dirty import full_image_dirty_bbox
from quantem.segmentation.services.confirm_batch.geometry import (
    safe_difference,
    safe_union,
)

CELL_INTERNAL_NAME = "quantem_internal_cells"
MAX_CELL_TAGS = 25
MAX_TAG_NAME_LENGTH = 64
MAX_TAG_VALUE_LENGTH = 200


def _stored_tags(raw_tags: object) -> list[dict[str, str]]:
    """Return the key/value tag shape while accepting legacy string tags."""
    if not isinstance(raw_tags, list):
        return []
    tags: list[dict[str, str]] = []
    for raw_tag in raw_tags:
        if isinstance(raw_tag, str):
            name = raw_tag.strip()
            value = ""
        elif isinstance(raw_tag, dict):
            name = str(raw_tag.get("name") or "").strip()
            value = str(raw_tag.get("value") or "").strip()
        else:
            continue
        if name:
            tags.append({"name": name, "value": value})
    return tags


def _object_tags(obj: AnalysisMaskObject) -> list[dict[str, str]]:
    try:
        return _stored_tags(obj.annotation.tags)
    except AnalysisMaskObjectAnnotation.DoesNotExist:
        return []


def _cell_tag_suggestions() -> list[dict[str, object]]:
    canonical_names: dict[str, str] = {}
    canonical_values: dict[str, dict[str, str]] = {}
    annotations = (
        AnalysisMaskObjectAnnotation.objects.filter(
            object__segmentation__segmentation_type__internal_name=CELL_INTERNAL_NAME
        )
        .order_by("created_at", "id")
        .values_list("tags", flat=True)
    )
    for raw_tags in annotations.iterator():
        for tag in _stored_tags(raw_tags):
            name_key = tag["name"].casefold()
            canonical_names.setdefault(name_key, tag["name"])
            values = canonical_values.setdefault(name_key, {})
            if tag["value"]:
                values.setdefault(tag["value"].casefold(), tag["value"])
    return [
        {
            "name": name,
            "values": list(canonical_values.get(name_key, {}).values()),
        }
        for name_key, name in canonical_names.items()
    ]


def _validated_cell_tags(raw_tags: object) -> list[dict[str, str]]:
    if not isinstance(raw_tags, list):
        raise ValueError("tags must be a list.")
    if len(raw_tags) > MAX_CELL_TAGS:
        raise ValueError(f"A cell can have at most {MAX_CELL_TAGS} tags.")

    suggestions = _cell_tag_suggestions()
    canonical_names = {str(row["name"]).casefold(): str(row["name"]) for row in suggestions}
    canonical_values = {
        str(row["name"]).casefold(): {str(value).casefold(): str(value) for value in row["values"]}
        for row in suggestions
    }
    tags: list[dict[str, str]] = []
    used_names: set[str] = set()
    for index, raw_tag in enumerate(raw_tags):
        if not isinstance(raw_tag, dict):
            raise ValueError(f"tags[{index}] must contain a name and value.")
        name = str(raw_tag.get("name") or "").strip()
        value = str(raw_tag.get("value") or "").strip()
        if not name and not value:
            continue
        if not name:
            raise ValueError(f"tags[{index}] needs a tag name.")
        if len(name) > MAX_TAG_NAME_LENGTH:
            raise ValueError(f"Tag names must be {MAX_TAG_NAME_LENGTH} characters or fewer.")
        if len(value) > MAX_TAG_VALUE_LENGTH:
            raise ValueError(f"Tag values must be {MAX_TAG_VALUE_LENGTH} characters or fewer.")
        name_key = name.casefold()
        if name_key in used_names:
            raise ValueError("A cell cannot contain the same tag name twice.")
        used_names.add(name_key)
        canonical_name = canonical_names.get(name_key, name)
        canonical_value = canonical_values.get(name_key, {}).get(value.casefold(), value)
        tags.append({"name": canonical_name, "value": canonical_value})
    return tags


def _analysis_segmentation(seg_id) -> ImageSegmentation:
    segmentation = get_object_or_404(
        ImageSegmentation.objects.select_related("asset", "segmentation_type"),
        id=seg_id,
    )
    if segmentation.segmentation_type.category != segmentation.segmentation_type.CATEGORY_AREA:
        raise ValueError("Area objects belong only to an area segmentation.")
    return segmentation


def _object_payload(obj: AnalysisMaskObject) -> dict:
    return {
        "id": str(obj.id),
        "segmentation": str(obj.segmentation_id),
        "name": obj.name,
        "color": obj.color,
        "sort_order": int(obj.sort_order),
        "tags": _object_tags(obj),
        "geometry": geometry_payload(obj.geometry),
        "created_at": obj.created_at,
        "updated_at": obj.updated_at,
    }


def _materialize_legacy_objects_if_needed(segmentation: ImageSegmentation) -> None:
    """Run the expensive raster-to-vector bridge only for actual legacy data.

    A newly created Cells layer has neither a global raster nor superseded
    source rasters. Taking a transaction lock and entering the legacy loader
    on every initial GET and every polygon POST made that ordinary empty state
    pay migration overhead before the user could draw anything.
    """
    has_legacy_raster = (
        GlobalMask.objects.filter(segmentation_id=segmentation.id).exists()
        or GlobalMask.objects.filter(segmentation__superseded_by_id=segmentation.id).exists()
    )
    if has_legacy_raster:
        materialize_legacy_area_instances(segmentation)


def _next_color(segmentation: ImageSegmentation) -> str:
    used = set(
        AnalysisMaskObject.objects.filter(segmentation=segmentation).values_list("color", flat=True)
    )
    available = [color for color in AREA_INSTANCE_COLORS if color not in used]
    return secrets.choice(available or list(AREA_INSTANCE_COLORS))


def _overlaps_another_instance(
    segmentation: ImageSegmentation,
    geometry: BaseGeometry | None,
    *,
    exclude_id=None,
) -> bool:
    if geometry is None or geometry.is_empty:
        return False
    if (
        segmentation.segmentation_type.topology_policy
        != segmentation.segmentation_type.TOPOLOGY_NON_OVERLAPPING
    ):
        return False
    objects = AnalysisMaskObject.objects.filter(segmentation=segmentation)
    if exclude_id:
        objects = objects.exclude(id=exclude_id)
    for other in objects.iterator():
        other_geometry = other.geometry
        if other_geometry is not None and geometry.intersection(other_geometry).area > 0:
            return True
    return False


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
    from quantem.segmentation.area_pixels import count_geometry_pixels

    # These endpoints materialize legacy masks before editing/listing. Count
    # current vectors directly, including the intentionally empty collection.
    return count_geometry_pixels(
        (
            obj.geometry
            for obj in AnalysisMaskObject.objects.filter(segmentation=segmentation).iterator()
        ),
        height=int(segmentation.asset.logical_height or 0),
        width=int(segmentation.asset.logical_width or 0),
    )


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
        _materialize_legacy_objects_if_needed(segmentation)
        objects = AnalysisMaskObject.objects.select_related("annotation").filter(
            segmentation=segmentation
        )
        return Response(
            {
                "objects": [_object_payload(obj) for obj in objects],
                "foreground_pixels": _foreground_pixels(segmentation),
                "tag_suggestions": _cell_tag_suggestions()
                if segmentation.segmentation_type.internal_name == CELL_INTERNAL_NAME
                else [],
            },
            status=status.HTTP_200_OK,
        )

    @transaction.atomic
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
            # New Cells do not enter the legacy materializer anymore, so take
            # the parent lock explicitly to serialize Cell names/sort orders.
            segmentation = (
                ImageSegmentation.objects.select_for_update(of=("self",))
                .select_related("asset", "segmentation_type")
                .get(pk=segmentation.pk)
            )
            _materialize_legacy_objects_if_needed(segmentation)
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
                    name=default_instance_name(segmentation),
                    color=_next_color(segmentation),
                    sort_order=int(last_order) + 1,
                )

            current = obj.geometry
            if operation == "include":
                result = safe_union(current, incoming)
            else:
                result = safe_difference(current, incoming) if current is not None else None
            if _overlaps_another_instance(
                segmentation,
                result,
                exclude_id=obj.id if obj.pk else None,
            ):
                return Response(
                    {
                        "error": (
                            "This definition uses non-overlapping instances. "
                            "Adjust the shape so it does not cover another instance."
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
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
            # Vector rows are now the durable Cell/Sample Area source. The old
            # raster save used to invalidate dependent analyses as a side
            # effect; keep that integrity guarantee when autosaving vectors.
            record_change(segmentation, geometry=True, area_vectors=True)
        # WKB is the durable editing representation. Do not rewrite a
        # full-image PNG or rebuild its pyramid for every brush stroke: that
        # was the long pause after R. Compatibility consumers can materialize
        # the union through the endpoint below; Cells do not need a second
        # manual save after every already-persisted vector edit.
        return Response(
            {
                "object": _object_payload(obj),
                "foreground_pixels": _foreground_pixels(segmentation),
                "overlay": None,
            },
            status=status.HTTP_201_CREATED if not object_id else status.HTTP_200_OK,
        )


class AnalysisMaskObjectDetailView(APIView):
    """Update metadata on or delete one analysis-mask object."""

    def _get(self, seg_id, object_id):
        segmentation = _analysis_segmentation(seg_id)
        ImageSegmentation.objects.select_for_update().get(pk=segmentation.pk)
        obj = get_object_or_404(
            AnalysisMaskObject,
            id=object_id,
            segmentation=segmentation,
        )
        return segmentation, obj

    @transaction.atomic
    def patch(self, request, seg_id, object_id):
        try:
            segmentation, obj = self._get(seg_id, object_id)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        name = str(request.data.get("name") or "").strip() if "name" in request.data else obj.name
        if not name and (segmentation.segmentation_type.internal_name == CELL_INTERNAL_NAME):
            return Response(
                {"error": "Cell name cannot be empty."}, status=status.HTTP_400_BAD_REQUEST
            )
        if len(name) > 100:
            return Response(
                {"error": "Object name must be 100 characters or fewer."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        tags = None
        if "tags" in request.data:
            if segmentation.segmentation_type.internal_name != CELL_INTERNAL_NAME:
                return Response(
                    {"error": "Tags can be added only to Cells."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            try:
                tags = _validated_cell_tags(request.data.get("tags"))
            except ValueError as exc:
                return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        with transaction.atomic():
            obj = AnalysisMaskObject.objects.select_for_update().get(pk=obj.pk)
            obj.name = name
            obj.save(update_fields=["name", "updated_at"])
            if tags is not None:
                annotation, _created = AnalysisMaskObjectAnnotation.objects.update_or_create(
                    object=obj,
                    defaults={"tags": tags},
                )
                obj.annotation = annotation
            record_change(segmentation)
        return Response(_object_payload(obj), status=status.HTTP_200_OK)

    @transaction.atomic
    def delete(self, request, seg_id, object_id):
        del request
        try:
            segmentation, obj = self._get(seg_id, object_id)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        deleted_id = str(obj.id)
        with transaction.atomic():
            obj.delete()
            record_change(segmentation, geometry=True, area_vectors=True)
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

    @transaction.atomic
    def post(self, request, seg_id):
        del request
        try:
            segmentation = _analysis_segmentation(seg_id)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        with transaction.atomic():
            ImageSegmentation.objects.select_for_update().get(pk=segmentation.pk)
            _materialize_legacy_objects_if_needed(segmentation)
            record = _rebuild(segmentation)
            if segmentation.status_stage != "CANDIDATES_READY":
                segmentation.status_stage = "CANDIDATES_READY"
                segmentation.save(update_fields=["status_stage"])
        overlay = _register_overlay(segmentation)
        return Response(
            {
                "objects": [
                    _object_payload(obj)
                    for obj in AnalysisMaskObject.objects.select_related("annotation").filter(
                        segmentation=segmentation
                    )
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
