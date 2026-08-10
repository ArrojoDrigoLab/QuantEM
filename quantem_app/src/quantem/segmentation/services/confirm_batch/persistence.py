from __future__ import annotations

from quantem.segmentation.models import ImageSegmentation, SegmentObject

from .geometry import filter_supported_confirmed_polygons
from .types import _ConfirmedFamily


def _parse_optional_sam_score(raw_score: object) -> float | None:
    if raw_score is None:
        return None
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return None


def _read_sam_score_from_features(features: object) -> float | None:
    if not isinstance(features, dict):
        return None
    return _parse_optional_sam_score(features.get("sam_score"))


def _persist_confirmed_family(
    *,
    segmentation: ImageSegmentation,
    family: _ConfirmedFamily,
) -> dict[str, list[str]]:
    polygons = filter_supported_confirmed_polygons(list(family.polygons))
    polygons.sort(key=lambda polygon: float(polygon.area), reverse=True)
    features = dict(family.features) if isinstance(family.features, dict) else {}
    confidence_score = (
        float(family.confidence_score)
        if isinstance(family.confidence_score, (float, int))
        else None
    )
    result = {
        "created_ids": [],
        "updated_ids": [],
        "deleted_ids": [],
        "refresh_ids": [],
    }

    if family.segment is None:
        primary_segment: SegmentObject | None = None
        for polygon in polygons:
            created_segment = SegmentObject.objects.create(
                segmentation=segmentation,
                label_state="CONFIRMED",
                confidence_score=confidence_score,
                features=dict(features),
                base_segment=primary_segment,
                geometry=polygon,
                centroid=polygon.centroid,
                bbox=polygon.envelope,
            )
            if primary_segment is None:
                primary_segment = created_segment
            result["created_ids"].append(str(created_segment.id))
            result["refresh_ids"].append(str(created_segment.id))
        return result

    if not polygons:
        segment_id = str(family.segment.id)
        family.segment.delete()
        result["deleted_ids"].append(segment_id)
        return result

    primary_polygon = polygons[0]
    family.segment.geometry = primary_polygon
    family.segment.centroid = primary_polygon.centroid
    family.segment.bbox = primary_polygon.envelope
    family.segment.label_state = "CONFIRMED"
    family.segment.confidence_score = confidence_score
    family.segment.features = dict(features)
    family.segment.save(
        update_fields=[
            "geometry",
            "centroid",
            "bbox",
            "label_state",
            "confidence_score",
            "features",
        ]
    )
    result["updated_ids"].append(str(family.segment.id))
    result["refresh_ids"].append(str(family.segment.id))

    base_segment = family.segment.resolve_base_segment_or_self()
    for polygon in polygons[1:]:
        created_segment = SegmentObject.objects.create(
            segmentation=segmentation,
            label_state="CONFIRMED",
            confidence_score=confidence_score,
            features=dict(features),
            base_segment=base_segment,
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
        )
        result["created_ids"].append(str(created_segment.id))
        result["refresh_ids"].append(str(created_segment.id))
    return result


__all__ = [
    "_parse_optional_sam_score",
    "_persist_confirmed_family",
    "_read_sam_score_from_features",
]
