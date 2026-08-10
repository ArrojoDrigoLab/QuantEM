"""Shared geometry helpers used across segmentation runtime modules."""

from .fields import (
    POLYGONAL_TYPES,
    bbox_field_names,
    bbox_property,
    expand_update_fields,
    geometry_from_wkb,
    geometry_to_wkb,
    point_property,
    repair_geometry,
    wkb_geometry_property,
)
from .polygons import (
    extract_polygons,
    iter_polygons,
    normalize_polygonal_geometry,
    polygon_coords,
)

__all__ = [
    "POLYGONAL_TYPES",
    "bbox_field_names",
    "bbox_property",
    "expand_update_fields",
    "extract_polygons",
    "geometry_from_wkb",
    "geometry_to_wkb",
    "iter_polygons",
    "normalize_polygonal_geometry",
    "point_property",
    "polygon_coords",
    "repair_geometry",
    "wkb_geometry_property",
]
