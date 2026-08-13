from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.models import SegmentObject

from .geometry import merge_polygons

MERGE_ELIGIBLE_STATES = ("CANDIDATE", "INFERRED", "CONFIRMED")
MANUAL_DELETE_ELIGIBLE_STATES = ("CANDIDATE", "INFERRED")
MANUAL_CANDIDATE_OVERLAP_THRESHOLD = 0.30
MIN_OVERLAP_AREA = 1e-6


@dataclass
class _ConfirmedFamily:
    segment: SegmentObject | None
    polygons: list[Polygon]
    features: dict[str, object]
    confidence_score: float | None = None
    is_manual_new: bool = False
    dirty: bool = False

    def union_geometry(self) -> BaseGeometry | None:
        return merge_polygons(self.polygons)
