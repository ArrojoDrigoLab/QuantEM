"""Shapely/WKB storage helpers for the geometry columns.

QuantEM ships plain SQLite with no spatial extension, so pixel-space geometry
lives in ordinary columns:

* ``<name>_wkb`` -- a ``BinaryField`` holding shapely WKB (image pixel space),
* ``bbox_minx`` / ``bbox_miny`` / ``bbox_maxx`` / ``bbox_maxy`` -- indexed floats,
* ``centroid_x`` / ``centroid_y`` -- indexed floats.

The properties built here keep call sites reading and writing shapely geometries
under the original logical names (``segment.geometry``, ``segment.bbox``,
``segment.centroid``), which is what the ``GEOSGeometry`` attributes used to give
them.

They are real ``property`` objects on purpose: ``django.db.models.Model.__init__``
forwards leftover keyword arguments only to attributes ``inspect.getattr_static``
reports as ``property``, and every existing call site does
``SegmentObject.objects.create(geometry=..., centroid=..., bbox=...)``.
"""

from __future__ import annotations

from shapely import wkb as shapely_wkb
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

POLYGONAL_TYPES = ("Polygon", "MultiPolygon")


def geometry_to_wkb(geometry: BaseGeometry) -> bytes:
    """Serialize a shapely geometry to WKB bytes for a ``BinaryField``."""
    return shapely_wkb.dumps(geometry)


def geometry_from_wkb(payload: bytes | memoryview | None) -> BaseGeometry | None:
    """Deserialize WKB read back from a ``BinaryField``; ``None`` stays ``None``."""
    if payload is None:
        return None
    data = bytes(payload)
    if not data:
        return None
    return shapely_wkb.loads(data)


def repair_geometry(
    geometry: object,
    *,
    subject: str,
    allowed_types: tuple[str, ...] = ("Polygon",),
) -> BaseGeometry:
    """Return ``geometry`` as a valid geometry whose type is in ``allowed_types``.

    GEOS silently tolerated some self-touching rings that shapely reports as
    invalid, so ingest runs ``shapely.make_valid``. A repair that changes the
    geometry type (a bowtie becoming a MultiPolygon, say) is an error rather than
    a silent truncation.
    """
    if not isinstance(geometry, BaseGeometry):
        raise ValueError(f"{subject} must be a shapely geometry.")
    if geometry.is_empty:
        raise ValueError(f"{subject} must not be empty.")

    repaired = geometry
    if not repaired.is_valid:
        try:
            repaired = make_valid(repaired)
        except Exception as exc:
            raise ValueError(f"{subject} could not be repaired.") from exc

    if repaired.is_empty or not repaired.is_valid or repaired.geom_type not in allowed_types:
        raise ValueError(
            f"{subject} must be a valid {'/'.join(allowed_types)}, got {repaired.geom_type}."
        )
    return repaired


def bbox_field_names(prefix: str = "bbox") -> tuple[str, str, str, str]:
    """Concrete column names backing the ``prefix`` bbox property."""
    return (f"{prefix}_minx", f"{prefix}_miny", f"{prefix}_maxx", f"{prefix}_maxy")


def wkb_geometry_property(wkb_attr: str, *, doc: str | None = None) -> property:
    """Build a shapely-geometry property backed by the WKB column ``wkb_attr``."""
    cache_attr = f"_{wkb_attr}_shapely_cache"

    def _get(self):
        raw = getattr(self, wkb_attr)
        cached = self.__dict__.get(cache_attr)
        if cached is not None and cached[0] is raw:
            return cached[1]
        geometry = geometry_from_wkb(raw)
        self.__dict__[cache_attr] = (raw, geometry)
        return geometry

    def _set(self, value):
        if value is None:
            setattr(self, wkb_attr, None)
            self.__dict__.pop(cache_attr, None)
            return
        if not isinstance(value, BaseGeometry):
            raise TypeError(
                f"{wkb_attr[:-4]} must be a shapely geometry, got {type(value).__name__}."
            )
        raw = geometry_to_wkb(value)
        setattr(self, wkb_attr, raw)
        self.__dict__[cache_attr] = (raw, value)

    return property(_get, _set, doc=doc)


def bbox_property(prefix: str = "bbox", *, doc: str | None = None) -> property:
    """Build an axis-aligned-rectangle property over the ``prefix`` float columns.

    Reading returns a shapely ``box``; assigning any non-empty geometry stores its
    bounds, so ``obj.bbox = polygon`` and ``obj.bbox = polygon.envelope`` agree.
    """
    minx_attr, miny_attr, maxx_attr, maxy_attr = bbox_field_names(prefix)

    def _get(self):
        values = [
            getattr(self, name, None) for name in (minx_attr, miny_attr, maxx_attr, maxy_attr)
        ]
        if any(value is None for value in values):
            return None
        return box(*(float(value) for value in values))

    def _set(self, value):
        if value is None:
            for name in (minx_attr, miny_attr, maxx_attr, maxy_attr):
                setattr(self, name, None)
            return
        if not isinstance(value, BaseGeometry):
            raise TypeError(f"{prefix} must be a shapely geometry, got {type(value).__name__}.")
        if value.is_empty:
            raise ValueError(f"{prefix} must not be empty.")
        min_x, min_y, max_x, max_y = value.bounds
        setattr(self, minx_attr, float(min_x))
        setattr(self, miny_attr, float(min_y))
        setattr(self, maxx_attr, float(max_x))
        setattr(self, maxy_attr, float(max_y))

    return property(_get, _set, doc=doc)


def point_property(
    x_attr: str,
    y_attr: str,
    *,
    name: str = "point",
    doc: str | None = None,
) -> property:
    """Build a shapely ``Point`` property over two float columns."""

    def _get(self):
        x_value = getattr(self, x_attr, None)
        y_value = getattr(self, y_attr, None)
        if x_value is None or y_value is None:
            return None
        return Point(float(x_value), float(y_value))

    def _set(self, value):
        if value is None:
            setattr(self, x_attr, None)
            setattr(self, y_attr, None)
            return
        if isinstance(value, BaseGeometry):
            if value.is_empty or value.geom_type != "Point":
                raise ValueError(f"{name} must be a non-empty Point, got {value.geom_type}.")
            x_value, y_value = value.x, value.y
        else:
            try:
                x_value, y_value = value
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a shapely Point or an (x, y) pair.") from exc
        setattr(self, x_attr, float(x_value))
        setattr(self, y_attr, float(y_value))

    return property(_get, _set, doc=doc)


def expand_update_fields(
    update_fields,
    mapping: dict[str, tuple[str, ...]],
) -> list[str] | None:
    """Translate logical geometry names in ``update_fields`` to real columns.

    ``segment.save(update_fields=["geometry", "centroid", "bbox"])`` was valid
    against the GeoDjango columns and stays valid here; this rewrites those names
    to ``geometry_wkb``, ``centroid_x``/``centroid_y``, ``bbox_minx``/... so the
    call sites do not have to know about the storage layout.
    """
    if update_fields is None:
        return None
    expanded: list[str] = []
    for name in update_fields:
        expanded.extend(mapping.get(name, (name,)))
    return list(dict.fromkeys(expanded))
