from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.models import SegmentObject

from .geometry import merge_polygons

MERGE_ELIGIBLE_STATES = ("CANDIDATE", "INFERRED", "CONFIRMED")
MANUAL_DELETE_ELIGIBLE_STATES = ("CANDIDATE", "INFERRED")
MANUAL_CANDIDATE_OVERLAP_THRESHOLD = 0.70
MIN_OVERLAP_AREA = 1e-6


@dataclass
class CandidateCleanupResult:
    """Candidate rows changed while manual ground truth takes precedence."""

    deleted_ids: list[str] = field(default_factory=list)
    updated_ids: list[str] = field(default_factory=list)
    created_ids: list[str] = field(default_factory=list)
    affected_geometries: list[BaseGeometry] = field(default_factory=list)

    def as_payload(self) -> dict[str, int]:
        return {
            "deleted": len(self.deleted_ids),
            "updated": len(self.updated_ids),
            "created": len(self.created_ids),
        }


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
