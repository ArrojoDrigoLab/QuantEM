from __future__ import annotations

import logging
from typing import Any

from django.db import transaction
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.features.measure import (
    UNMEASURED_DETAIL,
    MeasurementOutcome,
    measure_segments,
)
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.overlay_ngff import (
    DirtyBBox,
    merge_dirty_bboxes,
    register_overlay_mutation_all_bundles,
)
from quantem.segmentation.services.spatial_lookup import (
    bbox_intersects_filter,
)
from quantem.segmentation.source_models import SOURCE_MODEL_MANUAL

from .feature_refresh import _enqueue_segment_feature_refresh
from .geometry import (
    extract_polygons,
    filter_supported_confirmed_polygons,
    geometries_overlap,
    merge_polygons,
)
from .overlap import delete_manual_overlap_candidates, resolve_overlap_between_families
from .persistence import (
    _parse_optional_sam_score,
    _persist_confirmed_family,
    _read_sam_score_from_features,
)
from .types import MERGE_ELIGIBLE_STATES, _ConfirmedFamily

logger = logging.getLogger(__name__)

_CONFIRMED_ONLY_FIELDS = (
    "id",
    "segmentation_id",
    "geometry_wkb",
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
    "centroid_x",
    "centroid_y",
    "features",
    "base_segment_id",
    "label_state",
    "confidence_score",
)


def _measure_changed_segments(
    segmentation: ImageSegmentation, segment_ids: list[str]
) -> MeasurementOutcome:
    """Re-measure every object this batch drew or reshaped.

    ``objects.csv`` only reports ``CONFIRMED`` objects, and both branches below
    create confirmed objects from polygons with an empty ``features`` dict or
    reshape an existing one without touching its stored ``area``. Either way the
    morphometrics would be blank or describe the previous outline. Measuring is
    synchronous because the queued refresh is opt-in
    (``QUANTEM_ENABLE_SEGMENT_FEATURE_REFRESH_TRIGGERS`` defaults to ``0``) and a
    paper's table cannot depend on an environment variable.

    ``geometry_changed=True``: every id here is an outline this batch has just
    written. A piece split off an existing confirmed object inherits its
    ``features`` and its ``confidence_score``
    (``confirm_batch/persistence.py``), so without this the model probability of
    the shape *before* the merge would be reported for each piece after it.
    """
    if not segment_ids:
        return MeasurementOutcome()
    segments = list(
        SegmentObject.objects.filter(
            segmentation=segmentation, id__in=list(dict.fromkeys(segment_ids))
        )
    )
    try:
        return measure_segments(segmentation, segments, geometry_changed=True)
    except Exception:  # pragma: no cover - measurement must never lose an object
        logger.warning(
            "Failed to measure confirmed segments for %s",
            segmentation.id,
            exc_info=True,
        )
        return MeasurementOutcome(
            unmeasured=tuple(str(segment.id) for segment in segments),
            reason=UNMEASURED_DETAIL,
        )


def _confirm_manual_segments(
    *,
    segmentation: ImageSegmentation,
    incoming: list[dict[str, object]],
    enqueue_feature_refresh: bool,
) -> dict[str, object]:
    created = 0
    updated = 0
    deleted = 0
    confirmed_ids: list[str] = []
    feature_refresh_ids: list[str] = []
    affected_geometries: list[BaseGeometry] = []
    prepared_manual_families: list[_ConfirmedFamily] = []

    for item in incoming:
        geometry = item.get("geometry")
        normalized_geometry = geometry if isinstance(geometry, BaseGeometry) else None
        polygons = filter_supported_confirmed_polygons(extract_polygons(normalized_geometry))
        if not polygons:
            continue

        sam_score = _parse_optional_sam_score(item.get("sam_score"))
        incoming_features = item.get("features")
        features = (
            dict(incoming_features)
            if isinstance(incoming_features, dict)
            else ({"sam_score": float(sam_score)} if sam_score is not None else {})
        )
        if sam_score is not None and "sam_score" not in features:
            features["sam_score"] = float(sam_score)
        confidence_score = item.get("confidence_score")
        prepared_manual_families.append(
            _ConfirmedFamily(
                segment=None,
                polygons=polygons,
                features=features,
                confidence_score=(
                    float(confidence_score) if isinstance(confidence_score, (float, int)) else None
                ),
                is_manual_new=True,
                dirty=True,
            )
        )

    manual_bounds = merge_polygons(
        [
            union_geometry.envelope
            for family in prepared_manual_families
            for union_geometry in [family.union_geometry()]
            if union_geometry is not None
        ]
    )

    with transaction.atomic():
        confirmed_qs = SegmentObject.objects.filter(
            segmentation=segmentation,
            label_state="CONFIRMED",
        )
        bounds_filter = bbox_intersects_filter(manual_bounds)
        if bounds_filter is not None:
            confirmed_qs = confirmed_qs.filter(bounds_filter)
        existing_confirmed = list(confirmed_qs.only(*_CONFIRMED_ONLY_FIELDS))
        confirmed_families = [
            _ConfirmedFamily(
                segment=segment,
                polygons=extract_polygons(segment.geometry),
                features=dict(segment.features) if isinstance(segment.features, dict) else {},
                confidence_score=segment.confidence_score,
            )
            for segment in existing_confirmed
        ]
        manual_families: list[_ConfirmedFamily] = []

        for manual_family in prepared_manual_families:
            affected_geometries.extend(manual_family.polygons)

            for existing_family in confirmed_families:
                existing_geometry = existing_family.union_geometry()
                manual_geometry = manual_family.union_geometry()
                if existing_geometry is None or manual_geometry is None:
                    continue
                if not geometries_overlap(existing_geometry, manual_geometry):
                    continue
                affected_geometries.append(existing_geometry)
                if resolve_overlap_between_families(manual_family, existing_family):
                    affected_geometries.extend(manual_family.polygons)
                    affected_geometries.extend(existing_family.polygons)

            confirmed_families.append(manual_family)
            manual_families.append(manual_family)

        deleted_candidates, deleted_candidate_geometries = delete_manual_overlap_candidates(
            segmentation=segmentation,
            manual_families=manual_families,
        )
        deleted += deleted_candidates
        affected_geometries.extend(deleted_candidate_geometries)

        for family in confirmed_families:
            if family.segment is not None and not family.dirty:
                continue
            persist_result = _persist_confirmed_family(
                segmentation=segmentation,
                family=family,
            )
            created += len(persist_result["created_ids"])
            updated += len(persist_result["updated_ids"])
            deleted += len(persist_result["deleted_ids"])
            feature_refresh_ids.extend(persist_result["refresh_ids"])
            if family.is_manual_new:
                confirmed_ids.extend(persist_result["created_ids"])
            affected_geometries.extend(family.polygons)

    measurement = _measure_changed_segments(segmentation, feature_refresh_ids)

    if enqueue_feature_refresh:
        try:
            _enqueue_segment_feature_refresh(
                segmentation_id=str(segmentation.id),
                segment_ids=feature_refresh_ids,
                recompute_features=(created > 0 or updated > 0 or deleted > 0),
            )
        except Exception as exc:
            logger.warning(
                "Failed to queue feature refresh after manual confirm-batch for %s: %s",
                segmentation.id,
                exc,
                exc_info=True,
            )

    dirty_bbox = merge_dirty_bboxes(segmentation, affected_geometries)

    return {
        "created": created,
        "updated": updated,
        "deleted": deleted,
        "confirmed_ids": confirmed_ids,
        "dirty_bbox": dirty_bbox.as_dict() if dirty_bbox is not None else None,
        "measurement": measurement,
    }


def register_confirmation_overlay_mutation(
    *,
    segmentation: ImageSegmentation,
    result: dict[str, object],
    fallback_geometries: list[BaseGeometry | None],
) -> dict[str, Any] | None:
    """Apply the standard overlay mutation flow after a confirmation batch."""
    created = int(result.get("created", 0) or 0)
    updated = int(result.get("updated", 0) or 0)
    deleted = int(result.get("deleted", 0) or 0)
    if created <= 0 and updated <= 0 and deleted <= 0:
        return None

    dirty_bbox = None
    dirty_bbox_payload = result.get("dirty_bbox")
    if isinstance(dirty_bbox_payload, dict):
        try:
            dirty_bbox = DirtyBBox(
                x_min=int(dirty_bbox_payload["x_min"]),
                y_min=int(dirty_bbox_payload["y_min"]),
                x_max=int(dirty_bbox_payload["x_max"]),
                y_max=int(dirty_bbox_payload["y_max"]),
            )
        except Exception:
            dirty_bbox = None

    return register_overlay_mutation_all_bundles(
        segmentation,
        dirty_bbox=dirty_bbox or merge_dirty_bboxes(segmentation, fallback_geometries),
    )


def _strip_generated_flags(features: dict) -> dict:
    """Drop model "*_generated" markers so a manual object isn't re-inferred back
    to a model source_model by SegmentObject.save()."""
    return {key: value for key, value in features.items() if not str(key).endswith("_generated")}


def confirm_segment_geometries(
    *,
    segmentation: ImageSegmentation,
    incoming: list[dict[str, object]],
    merge_overlaps: bool,
    manual_creation: bool = False,
    enqueue_feature_refresh: bool = True,
) -> dict[str, object]:
    """Confirm polygons and optionally queue post-confirm feature refresh."""
    if manual_creation and not merge_overlaps:
        # Manual draw without merge: split overlaps with existing confirmed
        # objects.
        return _confirm_manual_segments(
            segmentation=segmentation,
            incoming=incoming,
            enqueue_feature_refresh=enqueue_feature_refresh,
        )

    # Manual draw WITH merge (ER "Confirm Drawn Area") falls through to the union
    # branch below but forces source_model="manual" on the resulting objects so a
    # drawn correction that absorbs model candidates is recorded as manual GT.
    forced_source_model = SOURCE_MODEL_MANUAL if manual_creation else None

    created = 0
    updated = 0
    deleted = 0
    confirmed_ids: list[str] = []
    feature_refresh_ids: list[str] = []
    affected_geometries: list[BaseGeometry] = []

    with transaction.atomic():
        if not merge_overlaps:
            for item in incoming:
                geometry = item["geometry"]
                if not isinstance(geometry, BaseGeometry):
                    continue
                polygons = filter_supported_confirmed_polygons(extract_polygons(geometry))
                if not polygons:
                    continue
                sam_score = item.get("sam_score")
                features = {"sam_score": float(sam_score)} if isinstance(sam_score, float) else {}
                for polygon in polygons:
                    affected_geometries.append(polygon)
                    segment = SegmentObject.objects.create(
                        segmentation=segmentation,
                        label_state="CONFIRMED",
                        confidence_score=None,
                        features=features,
                        geometry=polygon,
                        centroid=polygon.centroid,
                        bbox=polygon.envelope,
                    )
                    created += 1
                    confirmed_ids.append(str(segment.id))
                    feature_refresh_ids.append(str(segment.id))
        else:
            eligible_segments = list(
                SegmentObject.objects.filter(
                    segmentation=segmentation,
                    label_state__in=MERGE_ELIGIBLE_STATES,
                )
            )
            eligible_geometries = {
                str(segment.id): segment.geometry for segment in eligible_segments
            }
            remaining = [
                {
                    "geometry": item["geometry"],
                    "sam_score": item.get("sam_score"),
                }
                for item in incoming
            ]

            while remaining:
                seed = remaining.pop(0)
                group_geometry = seed["geometry"]
                if not isinstance(group_geometry, BaseGeometry):
                    continue

                group_scores: list[float] = []
                seed_score = _parse_optional_sam_score(seed.get("sam_score"))
                if seed_score is not None:
                    group_scores.append(seed_score)
                overlapping_ids: set[str] = set()

                changed = True
                while changed:
                    changed = False

                    still_remaining: list[dict[str, object]] = []
                    for candidate in remaining:
                        candidate_geometry = candidate.get("geometry")
                        if not isinstance(candidate_geometry, BaseGeometry):
                            continue
                        if geometries_overlap(group_geometry, candidate_geometry):
                            affected_geometries.append(candidate_geometry)
                            group_geometry = group_geometry.union(candidate_geometry)
                            candidate_score = _parse_optional_sam_score(candidate.get("sam_score"))
                            if candidate_score is not None:
                                group_scores.append(candidate_score)
                            changed = True
                        else:
                            still_remaining.append(candidate)
                    remaining = still_remaining

                    for existing in eligible_segments:
                        existing_id = str(existing.id)
                        if existing_id in overlapping_ids:
                            continue
                        existing_geometry = eligible_geometries.get(existing_id)
                        if existing_geometry is None:
                            continue
                        if geometries_overlap(group_geometry, existing_geometry):
                            overlapping_ids.add(existing_id)
                            affected_geometries.append(existing_geometry)
                            group_geometry = group_geometry.union(existing_geometry)
                            existing_score = _read_sam_score_from_features(existing.features)
                            if existing_score is not None:
                                group_scores.append(existing_score)
                            changed = True

                merged_polygons = filter_supported_confirmed_polygons(
                    extract_polygons(group_geometry)
                )
                if not merged_polygons:
                    continue

                merged_polygons.sort(key=lambda poly: float(poly.area), reverse=True)

                overlapping_segments = [
                    segment for segment in eligible_segments if str(segment.id) in overlapping_ids
                ]
                confirmed_existing = [
                    segment
                    for segment in overlapping_segments
                    if segment.label_state == "CONFIRMED"
                ]
                primary_segment = (
                    confirmed_existing[0]
                    if confirmed_existing
                    else (overlapping_segments[0] if overlapping_segments else None)
                )
                used_existing_ids: set[str] = set()
                merged_score = max(group_scores) if group_scores else None

                for poly_index, polygon in enumerate(merged_polygons):
                    use_primary = poly_index == 0 and primary_segment is not None
                    affected_geometries.append(polygon)

                    if use_primary:
                        features = (
                            dict(primary_segment.features)
                            if isinstance(primary_segment.features, dict)
                            else {}
                        )
                        if merged_score is not None:
                            features["sam_score"] = float(merged_score)
                        primary_update_fields = [
                            "geometry",
                            "centroid",
                            "bbox",
                            "label_state",
                            "confidence_score",
                            "features",
                        ]
                        if forced_source_model is not None:
                            features = _strip_generated_flags(features)
                            primary_segment.source_model = forced_source_model
                            primary_update_fields.append("source_model")
                        primary_segment.geometry = polygon
                        primary_segment.centroid = polygon.centroid
                        primary_segment.bbox = polygon.envelope
                        primary_segment.label_state = "CONFIRMED"
                        primary_segment.confidence_score = None
                        primary_segment.features = features
                        primary_segment.save(update_fields=primary_update_fields)
                        eligible_geometries[str(primary_segment.id)] = polygon
                        updated += 1
                        confirmed_ids.append(str(primary_segment.id))
                        feature_refresh_ids.append(str(primary_segment.id))
                        used_existing_ids.add(str(primary_segment.id))
                        continue

                    features = (
                        {"sam_score": float(merged_score)} if merged_score is not None else {}
                    )
                    create_kwargs: dict[str, Any] = {}
                    if forced_source_model is not None:
                        create_kwargs["source_model"] = forced_source_model
                    created_segment = SegmentObject.objects.create(
                        segmentation=segmentation,
                        label_state="CONFIRMED",
                        confidence_score=None,
                        features=features,
                        geometry=polygon,
                        centroid=polygon.centroid,
                        bbox=polygon.envelope,
                        **create_kwargs,
                    )
                    created += 1
                    confirmed_ids.append(str(created_segment.id))
                    feature_refresh_ids.append(str(created_segment.id))
                    eligible_segments.append(created_segment)
                    eligible_geometries[str(created_segment.id)] = polygon

                delete_ids = [
                    str(segment.id)
                    for segment in overlapping_segments
                    if str(segment.id) not in used_existing_ids
                ]
                if delete_ids:
                    SegmentObject.objects.filter(
                        segmentation=segmentation,
                        id__in=delete_ids,
                    ).delete()
                    deleted += len(delete_ids)
                    eligible_segments = [
                        segment
                        for segment in eligible_segments
                        if str(segment.id) not in delete_ids
                    ]

    measurement = _measure_changed_segments(segmentation, feature_refresh_ids)

    if enqueue_feature_refresh:
        try:
            _enqueue_segment_feature_refresh(
                segmentation_id=str(segmentation.id),
                segment_ids=feature_refresh_ids,
                recompute_features=(created > 0 or updated > 0 or deleted > 0),
            )
        except Exception as exc:
            logger.warning(
                "Failed to queue feature refresh after confirm-batch for %s: %s",
                segmentation.id,
                exc,
                exc_info=True,
            )

    dirty_bbox = merge_dirty_bboxes(segmentation, affected_geometries)
    return {
        "created": created,
        "updated": updated,
        "deleted": deleted,
        "confirmed_ids": confirmed_ids,
        "dirty_bbox": dirty_bbox.as_dict() if dirty_bbox is not None else None,
        "measurement": measurement,
    }
