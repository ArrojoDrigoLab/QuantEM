"""Persistent instance geometry for Cells and reusable instance-area definitions."""

from __future__ import annotations

import numpy as np
from django.db import transaction
from scipy import ndimage
from shapely.affinity import translate
from shapely.geometry import Polygon
from skimage import measure

from quantem.segmentation.geometry.polygons import normalize_polygonal_geometry
from quantem.segmentation.global_masks import load_global_mask
from quantem.segmentation.models import AnalysisMaskObject, ImageSegmentation
from quantem.segmentation.type_definitions import CELLS

AREA_INSTANCE_COLORS = (
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


def is_instance_area(segmentation: ImageSegmentation) -> bool:
    definition = segmentation.segmentation_type
    return (
        definition.category == definition.CATEGORY_AREA
        and definition.representation == definition.REPRESENTATION_INSTANCES
    )


def next_cell_name(segmentation: ImageSegmentation) -> str:
    """Return ``Cell N`` above the highest current automatic cell number."""
    largest = 0
    for name in AnalysisMaskObject.objects.filter(segmentation=segmentation).values_list(
        "name", flat=True
    ):
        value = str(name or "").strip()
        prefix, separator, suffix = value.rpartition(" ")
        if separator and prefix.casefold() == "cell" and suffix.isdigit():
            largest = max(largest, int(suffix))
    return f"Cell {largest + 1}"


def default_instance_name(segmentation: ImageSegmentation) -> str:
    if segmentation.segmentation_type.internal_name == CELLS.internal_name:
        return next_cell_name(segmentation)
    return ""


def _component_geometry(component: np.ndarray):
    """Trace one component, retaining interior holes from the saved raster."""
    padded = np.pad(component.astype(bool, copy=False), 1, constant_values=False)
    contours = measure.find_contours(padded, level=0.5)
    rings = [
        [(float(col - 1), float(row - 1)) for row, col in contour]
        for contour in contours
        if len(contour) >= 4
    ]
    if not rings:
        return None
    ring_polygons = [Polygon(ring) for ring in rings]
    exterior_index = max(range(len(rings)), key=lambda index: abs(float(ring_polygons[index].area)))
    exterior = ring_polygons[exterior_index]
    holes = [
        ring
        for index, ring in enumerate(rings)
        if index != exterior_index
        and exterior.contains(ring_polygons[index].representative_point())
    ]
    return normalize_polygonal_geometry(Polygon(rings[exterior_index], holes))


@transaction.atomic
def materialize_legacy_area_instances(
    segmentation: ImageSegmentation,
) -> list[AnalysisMaskObject]:
    """Persist connected components when a legacy instance area has only a raster.

    This is intentionally lazy. Older installations can keep mask files outside
    the database migration environment; the first edit or analysis is the first
    time both the database and its configured data directory are guaranteed to
    be available. Once written, UUIDs and Cell names are stable.
    """
    # Lock the stable parent row, not just its children. A SELECT FOR UPDATE on
    # an empty child queryset locks nothing; two PostgreSQL requests could then
    # both materialize the same legacy raster, choose the same Cell name and
    # sort order, and commit duplicate objects. SQLite's IMMEDIATE transaction
    # used to hide that race by serializing every writer.
    segmentation = (
        ImageSegmentation.objects.select_for_update()
        .select_related("segmentation_type")
        .get(pk=segmentation.pk)
    )
    if segmentation.segmentation_type.category != segmentation.segmentation_type.CATEGORY_AREA:
        return list(AnalysisMaskObject.objects.filter(segmentation=segmentation))

    existing = list(
        AnalysisMaskObject.objects.select_for_update().filter(segmentation=segmentation)
    )
    created: list[AnalysisMaskObject] = []
    sources = list(segmentation.superseded_segmentations.select_related("segmentation_type").all())
    # The canonical layer's own raster is materialized only when it has no
    # vector objects. Superseded source rasters are processed independently so
    # two legacy masks merged into Cells do not silently lose the second one.
    if not existing:
        sources.insert(0, segmentation)

    next_number = 1
    for name in [obj.name for obj in existing]:
        prefix, separator, suffix = str(name or "").strip().rpartition(" ")
        if separator and prefix.casefold() == "cell" and suffix.isdigit():
            next_number = max(next_number, int(suffix) + 1)

    for source in sources:
        try:
            global_record = source.global_mask
        except Exception:
            global_record = None
        metadata = dict(getattr(global_record, "metadata", {}) or {})
        if metadata.get("instances_migrated_to") == str(segmentation.id):
            continue
        try:
            mask = load_global_mask(source).astype(bool, copy=False)
        except Exception:
            continue
        if not mask.any():
            if global_record is not None:
                metadata["instances_migrated_to"] = str(segmentation.id)
                global_record.metadata = metadata
                global_record.save(update_fields=["metadata", "updated_at"])
            continue

        labels, _count = ndimage.label(mask)
        slices = ndimage.find_objects(labels)
        for label_index, bounds in enumerate(slices, start=1):
            if bounds is None:
                continue
            y_slice, x_slice = bounds
            component = labels[y_slice, x_slice] == label_index
            if not component.any():
                continue
            polygon = _component_geometry(component)
            if polygon is None:
                continue
            geometry = translate(polygon, xoff=float(x_slice.start), yoff=float(y_slice.start))
            name = ""
            if segmentation.segmentation_type.internal_name == CELLS.internal_name:
                name = f"Cell {next_number}"
                next_number += 1
            created.append(
                AnalysisMaskObject(
                    segmentation=segmentation,
                    name=name,
                    color=AREA_INSTANCE_COLORS[
                        (len(existing) + len(created)) % len(AREA_INSTANCE_COLORS)
                    ],
                    sort_order=len(existing) + len(created) + 1,
                    geometry_wkb=geometry.wkb,
                )
            )
        if global_record is not None:
            metadata["instances_migrated_to"] = str(segmentation.id)
            global_record.metadata = metadata
            global_record.save(update_fields=["metadata", "updated_at"])
    if created:
        AnalysisMaskObject.objects.bulk_create(created)
    return list(AnalysisMaskObject.objects.filter(segmentation=segmentation))
