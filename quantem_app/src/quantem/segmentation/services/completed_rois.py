from __future__ import annotations

from django.db import transaction
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.geometry import extract_polygons, normalize_polygonal_geometry
from quantem.segmentation.models import CompletedROI, ImageSegmentation
from quantem.segmentation.services.spatial_lookup import (
    bbox_intersects_filter,
)


def _normalize_single_polygon_geometry(
    geometry: BaseGeometry | None,
    *,
    field_name: str,
) -> Polygon:
    if geometry is None or geometry.is_empty:
        raise ValueError(f"{field_name} must not be empty.")

    normalized = normalize_polygonal_geometry(geometry)
    if normalized is None or normalized.is_empty:
        raise ValueError(f"{field_name} could not be repaired.")

    polygons = extract_polygons(normalized)
    if len(polygons) != 1:
        raise ValueError(f"{field_name} must resolve to exactly one polygon.")

    polygon = polygons[0]
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError(f"{field_name} must be valid polygon geometry.")
    return polygon


def _parse_polygon_coords(raw_coords: object) -> Polygon:
    if not isinstance(raw_coords, list) or len(raw_coords) < 3:
        raise ValueError("polygon_coords must include at least 3 points.")

    coords: list[tuple[float, float]] = []
    for coord in raw_coords:
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            raise ValueError("polygon_coords must be [x, y] pairs.")
        try:
            coords.append((float(coord[0]), float(coord[1])))
        except (TypeError, ValueError) as exc:
            raise ValueError("polygon_coords must be numeric.") from exc

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    try:
        geometry: BaseGeometry = Polygon(coords)
    except Exception as exc:
        raise ValueError("polygon_coords must define valid polygon geometry.") from exc
    return _normalize_single_polygon_geometry(
        geometry,
        field_name="polygon_coords",
    )


def _image_bounds_polygon(segmentation: ImageSegmentation) -> Polygon:
    asset = segmentation.asset
    if asset is None:
        raise ValueError("Segmentation has no target asset.")
    width = int(asset.logical_width or 0)
    height = int(asset.logical_height or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Segmentation asset is missing dimensions.")
    return box(0.0, 0.0, float(width), float(height))


def _validate_polygon_within_image(
    *,
    segmentation: ImageSegmentation,
    polygon: Polygon,
) -> None:
    try:
        if not _image_bounds_polygon(segmentation).covers(polygon):
            raise ValueError("polygon_coords must remain inside the image bounds.")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("polygon_coords could not be validated against the image bounds.") from exc


def list_completed_rois(segmentation: ImageSegmentation):
    return CompletedROI.objects.filter(segmentation=segmentation).order_by("created_at", "id")


@transaction.atomic
def save_completed_roi(
    *,
    segmentation: ImageSegmentation,
    polygon_coords: object,
) -> tuple[CompletedROI, bool]:
    draft = _parse_polygon_coords(polygon_coords)
    _validate_polygon_within_image(segmentation=segmentation, polygon=draft)

    merged_records: list[CompletedROI] = []
    merged_ids: set[str] = set()
    retained_record: CompletedROI | None = None

    while True:
        candidate_qs = (
            CompletedROI.objects.select_for_update()
            .filter(segmentation=segmentation)
            .exclude(id__in=merged_ids)
        )
        bounds_filter = bbox_intersects_filter(draft.envelope)
        if bounds_filter is not None:
            candidate_qs = candidate_qs.filter(bounds_filter)
        candidates = list(candidate_qs.order_by("created_at", "id"))
        merged_this_round = False

        for existing in candidates:
            existing_geometry = existing.geometry
            if existing_geometry is None:
                continue
            try:
                intersects = bool(existing_geometry.intersects(draft))
            except Exception:
                intersects = False
            if not intersects:
                continue

            try:
                merged_geometry = _normalize_single_polygon_geometry(
                    existing_geometry.union(draft),
                    field_name="merged completed ROI geometry",
                )
            except ValueError:
                continue

            _validate_polygon_within_image(
                segmentation=segmentation,
                polygon=merged_geometry,
            )
            draft = merged_geometry
            merged_records.append(existing)
            merged_ids.add(str(existing.id))
            if retained_record is None:
                retained_record = existing
            merged_this_round = True
            break

        if not merged_this_round:
            break

    if retained_record is None:
        return (
            CompletedROI.objects.create(
                segmentation=segmentation,
                geometry=draft,
            ),
            True,
        )

    retained_record.geometry = draft
    retained_record.save(update_fields=["geometry", "bbox", "updated_at"])

    absorbed_ids = [record.id for record in merged_records if record.id != retained_record.id]
    if absorbed_ids:
        CompletedROI.objects.filter(id__in=absorbed_ids).delete()

    return retained_record, False


@transaction.atomic
def subtract_completed_roi(
    *,
    segmentation: ImageSegmentation,
    polygon_coords: object,
) -> dict[str, int]:
    """Remove a freehand polygon area from the confirmed-area layer.

    The cut polygon is subtracted from every existing completed ROI it overlaps.
    A subtraction can shrink a polygon, punch a hole in it (interior ring), split
    it into several pieces, or remove it entirely; the resulting pieces are
    re-normalized back into one CompletedROI row per disjoint polygon so the
    layer always stays a clean set of valid polygons.
    """
    cut = _parse_polygon_coords(polygon_coords)

    rows_qs = CompletedROI.objects.select_for_update().filter(segmentation=segmentation)
    bounds_filter = bbox_intersects_filter(cut.envelope)
    if bounds_filter is not None:
        rows_qs = rows_qs.filter(bounds_filter)
    rows = list(rows_qs.order_by("created_at", "id"))

    updated = 0
    deleted = 0
    created = 0

    for row in rows:
        row_geometry = row.geometry
        if row_geometry is None:
            continue
        try:
            if not row_geometry.intersects(cut):
                continue
            remainder = row_geometry.difference(cut)
        except Exception:
            continue

        pieces: list[Polygon] = []
        if remainder is not None and not remainder.is_empty:
            normalized = normalize_polygonal_geometry(remainder)
            if normalized is not None and not normalized.is_empty:
                pieces = [
                    polygon
                    for polygon in extract_polygons(normalized)
                    if polygon is not None and not polygon.is_empty and polygon.is_valid
                ]

        if not pieces:
            row.delete()
            deleted += 1
            continue

        row.geometry = pieces[0]
        row.save(update_fields=["geometry", "bbox", "updated_at"])
        updated += 1

        for extra in pieces[1:]:
            CompletedROI.objects.create(
                segmentation=segmentation,
                geometry=extra,
            )
            created += 1

    return {"updated": updated, "deleted": deleted, "created": created}
