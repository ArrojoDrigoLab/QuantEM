"""The include-level dial: read where it is, and ask for it to be moved.

The threshold is the foreground cutoff (:mod:`quantem.segmentation.run_identity`).
Moving it does **not** run the model or change candidate objects: the browser
recolors the probability map that the run already stored. Pressing Preview is
the separate materialization step; it re-thresholds that stored map, replaces
only the unconfirmed candidates, and leaves confirmed/manual annotations intact. The
backend of that commit is
:func:`~quantem.seg_core.db.inference.replay_stored_probability_map`; the worker
is :mod:`quantem.jobs.handlers.rethreshold`.

Why the refusals happen *here*, before anything is queued
---------------------------------------------------------
Every reason a dial move cannot work is knowable at the moment it is asked for,
and all of them are cheap to check: no map stored, a map from an older build, a
model with more than one output, a model not installed on this machine, a run
already holding the image. Queuing anyway would put a task on screen that is
certain to go red, and hand the user the reason a minute after the moment they
could have acted on it. So ``POST`` answers 409 with the sentence, and ``GET``
answers the same question in advance so the control can be greyed out with the
reason beside it rather than failing under the user's hand.

The two unavailable-map cases say different things, and that difference is
carried all the way to the client. Both end in "run the model again"; only one
of them will keep happening until the stored result is replaced. See
:data:`~quantem.segmentation.prob_maps.persistence.NO_STORED_MAP_MESSAGE` and
:data:`~quantem.segmentation.prob_maps.persistence.LEGACY_MAP_MESSAGE`.

``urlpatterns`` at the bottom is spliced into
:mod:`quantem.segmentation.urls`, which is why the routes are defined here and
not there: four packages are adding routes this release and none of them opens
that file.
"""

from __future__ import annotations

import logging
from urllib.parse import urlencode

from django.db import transaction
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from django.urls import path, reverse
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView
from shapely import STRtree
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from quantem.assets.models import ImageROI
from quantem.core.error_codes import ERROR_CODE_FIELD, ErrorCode
from quantem.jobs.constants import (
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
    QUEUE_P1_INTERACTIVE,
)
from quantem.jobs.models import Job
from quantem.seg_core.db.extraction import resolve_min_area
from quantem.seg_core.db.prob_maps import get_prob_map_file_path
from quantem.seg_core.registry import get_segmenter_or_none
from quantem.segmentation.models import (
    CompletedROI,
    ImageSegmentation,
    RoiSegmentationStatus,
    SegmentationResultVersion,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff.dirty import full_image_dirty_bbox
from quantem.segmentation.overlay_ngff.mutations import (
    register_overlay_mutation_all_bundles,
)
from quantem.segmentation.prob_maps.persistence import stored_map_readiness
from quantem.segmentation.prob_maps.preview import (
    ensure_probability_preview,
    probability_map_size,
)
from quantem.segmentation.segment_status import status_for_segment_lifecycle
from quantem.segmentation.services.confirm_batch.feature_refresh import (
    _enqueue_segment_feature_refresh,
)
from quantem.segmentation.services.confirm_batch.geometry import (
    extract_polygons,
    filter_supported_confirmed_polygons,
    geometries_overlap,
    geometry_area,
    safe_difference,
    safe_intersection,
    safe_union,
)
from quantem.segmentation.services.confirm_batch.overlap import (
    overlap_qualifies_for_union,
)
from quantem.segmentation.source_models import (
    SOURCE_MODEL_MANUAL,
    default_source_model_for_organelle,
    get_source_model_definition,
    normalize_source_model,
    resolve_segmenter_internal_name,
)
from quantem.segmentation.status_reconcile import reconcile_segmentation_status

from .shared import (
    _ORGANELLE_ACTION_JOB_TYPES,
    active_segmentation_job,
    blocking_job_response_payload,
    completion_lock_response,
)

logger = logging.getLogger(__name__)

#: The dial's range. It is a probability, so this is not a product choice.
INCLUDE_LEVEL_MIN = 0.0
INCLUDE_LEVEL_MAX = 1.0

#: Said when the model that produced the stored map cannot be loaded here.
#: Re-thresholding needs that model's own extraction settings -- its area floor,
#: its closing radius -- so without it this is not the same operation the run
#: performed, and doing it with another model's settings would silently produce
#: a different candidate set under the same name.
MODEL_UNAVAILABLE_MESSAGE = (
    "The model that found these objects is not available on this computer, so "
    "they cannot be redone at a different include level. Install it on the "
    "Models screen."
)

#: Said for a model whose foreground comes from several outputs combined. The
#: combination is not what gets stored, so replaying would have to redo it --
#: a second decision procedure, which is the thing the stored-map ordering
#: exists to avoid.
MULTI_OUTPUT_MESSAGE = (
    "The include level can only be moved for models that produce a single "
    "confidence map, and this one combines several."
)


class IncludeLevelSerializer(serializers.Serializer):
    """One requested dial position.

    Validated here rather than at the model layer, which does not constrain the
    field: ``ImageSegmentation.include_level`` is a plain ``FloatField``, and a
    level outside 0-1 would be accepted, stored, and handed to the segmenter's
    threshold setter, where it produces either every pixel or none of them and
    reports success.
    """

    include_level = serializers.FloatField(
        min_value=INCLUDE_LEVEL_MIN,
        max_value=INCLUDE_LEVEL_MAX,
        error_messages={
            "required": "Choose an include level between 0 and 1.",
            "invalid": "The include level has to be a number between 0 and 1.",
            "min_value": "The include level has to be between 0 and 1.",
            "max_value": "The include level has to be between 0 and 1.",
        },
    )
    source_model = serializers.CharField(required=False, allow_blank=True)
    roi_id = serializers.UUIDField(required=False, allow_null=True)


class ConfirmModelOutputSerializer(serializers.Serializer):
    """The model whose current candidates should become analysis-ready."""

    source_model = serializers.CharField(required=True, allow_blank=False)


def _effective_source_model(
    segmentation: ImageSegmentation,
    source_model: str | None,
) -> str:
    normalized = normalize_source_model(source_model)
    if normalized:
        return normalized
    return default_source_model_for_organelle(segmentation.segmentation_type.internal_name)


def _manual_review_geometries(segmentation: ImageSegmentation) -> list:
    """Areas whose dense manual labels must win over a whole-image accept.

    QuantEM has two representations for a manually finished area. The freeform
    ``CompletedROI`` is used by the Confirmed area tool; the rectangular
    ``RoiSegmentationStatus`` is used by the per-organelle ROI workflow. A
    candidate touching either is deliberately left as a candidate. Confirming
    it automatically would turn the model back into ground truth inside the
    exact area the user marked as exhaustively reviewed.
    """

    geometries = [
        completed.geometry
        for completed in CompletedROI.objects.filter(segmentation=segmentation).only("geometry_wkb")
    ]
    reviewed_windows = (
        RoiSegmentationStatus.objects.filter(
            segmentation=segmentation,
            is_complete=True,
        )
        .select_related("image_roi")
        .only(
            "image_roi__x",
            "image_roi__y",
            "image_roi__width",
            "image_roi__height",
        )
    )
    for reviewed in reviewed_windows:
        roi = reviewed.image_roi
        geometries.append(
            box(
                float(roi.x),
                float(roi.y),
                float(roi.x + roi.width),
                float(roi.y + roi.height),
            )
        )
    return geometries


def _partition_model_candidates(
    segmentation: ImageSegmentation,
    source_model: str,
) -> tuple[list[SegmentObject], list[SegmentObject], int]:
    """Return confirmable candidates, candidates on manual ground, and ROI count."""

    candidates = list(
        SegmentObject.objects.filter(
            segmentation=segmentation,
            source_model=source_model,
            label_state="CANDIDATE",
            superseded_at__isnull=True,
        ).only(
            "id",
            "geometry_wkb",
            "label_state",
            "refined",
            "status",
            "source_model",
            "confidence_score",
            "features",
            "run_version",
        )
    )
    manual_geometries = _manual_review_geometries(segmentation)
    if not manual_geometries:
        return candidates, [], 0

    reviewed_area = prep(unary_union(manual_geometries))
    confirmable: list[SegmentObject] = []
    protected: list[SegmentObject] = []
    for candidate in candidates:
        if reviewed_area.intersects(candidate.geometry):
            protected.append(candidate)
        else:
            confirmable.append(candidate)
    return confirmable, protected, len(manual_geometries)


def _model_output_counts(
    segmentation: ImageSegmentation,
    source_model: str,
) -> dict[str, int]:
    confirmable, protected, manual_roi_count = _partition_model_candidates(
        segmentation,
        source_model,
    )
    confirmed = SegmentObject.objects.filter(
        segmentation=segmentation,
        source_model=source_model,
        label_state="CONFIRMED",
        superseded_at__isnull=True,
    ).count()
    return {
        "candidate_count": len(confirmable) + len(protected),
        "confirmable_candidate_count": len(confirmable),
        "manual_roi_candidate_count": len(protected),
        "manual_roi_count": manual_roi_count,
        "confirmed_model_count": int(confirmed),
    }


def _confirmed_primary_key(segment: SegmentObject) -> tuple[int, int, str]:
    """Keep hand-drawn provenance when a candidate joins an existing object."""

    return (
        0 if segment.source_model == SOURCE_MODEL_MANUAL or segment.refined == "MANUAL" else 1,
        0 if segment.refined == "MANUAL" else 1,
        str(segment.id),
    )


def _candidate_remainder(
    geometry: BaseGeometry,
    blockers: list[BaseGeometry],
) -> BaseGeometry | None:
    """Remove confirmed pixels without changing any confirmed boundary."""

    if not blockers:
        return geometry
    try:
        blocked = unary_union(blockers)
    except Exception:
        blocked = None
        for blocker in blockers:
            blocked = safe_union(blocked, blocker)
    if blocked is None:
        return geometry
    return safe_difference(geometry, blocked)


def _post_exclusion_polygons(
    geometry: BaseGeometry | None,
    *,
    min_area: int,
) -> list:
    """Usable connected pieces that still meet the organelle's native-pixel floor."""

    polygons = filter_supported_confirmed_polygons(extract_polygons(geometry))
    return [polygon for polygon in polygons if geometry_area(polygon) >= min_area]


def _confirm_model_candidates(
    *,
    segmentation: ImageSegmentation,
    candidates: list[SegmentObject],
    min_area: int,
) -> dict[str, int]:
    """Accept model candidates without ever seam-splitting a confirmed object.

    Preview candidates retain the full post-processed model geometry.  At this
    explicit confirmation boundary, a candidate and an existing confirmed
    object are unioned only when the established 70% either-direction rule
    qualifies.  Otherwise the confirmed union is subtracted from the model
    candidate, every confirmed outline stays byte-for-byte unchanged, and the
    minimum-area floor is applied to the remaining connected pieces.
    """

    confirmed = list(
        SegmentObject.objects.filter(
            segmentation=segmentation,
            label_state="CONFIRMED",
            superseded_at__isnull=True,
        ).only(
            "id",
            "geometry_wkb",
            "label_state",
            "refined",
            "status",
            "source_model",
            "features",
        )
    )
    confirmed_geometries = [segment.geometry for segment in confirmed]
    tree = STRtree(confirmed_geometries) if confirmed_geometries else None
    current_geometry = {
        str(segment.id): geometry
        for segment, geometry in zip(confirmed, confirmed_geometries, strict=True)
    }
    deleted_confirmed_ids: set[str] = set()

    plain_confirmations: list[SegmentObject] = []
    accepted_candidates = 0
    merged_candidates = 0
    clipped_candidates = 0
    filtered_candidates = 0
    created_fragments = 0
    deleted_confirmed = 0

    for candidate in candidates:
        candidate_geometry = candidate.geometry
        neighbours: list[tuple[SegmentObject, BaseGeometry]] = []
        if tree is not None:
            for position in tree.query(candidate_geometry).tolist():
                existing = confirmed[int(position)]
                existing_id = str(existing.id)
                if existing_id in deleted_confirmed_ids:
                    continue
                geometry = current_geometry.get(existing_id)
                if geometry is not None and geometries_overlap(candidate_geometry, geometry):
                    neighbours.append((existing, geometry))

        if not neighbours:
            candidate.label_state = "CONFIRMED"
            candidate.refined = "UNREFINED"
            candidate.confidence_score = None
            candidate.status = status_for_segment_lifecycle(
                label_state=candidate.label_state,
                refined=candidate.refined,
            )
            plain_confirmations.append(candidate)
            accepted_candidates += 1
            continue

        qualifying = [
            (segment, geometry)
            for segment, geometry in neighbours
            if overlap_qualifies_for_union(candidate_geometry, geometry)
        ]
        if qualifying:
            qualifying_ids = {str(segment.id) for segment, _geometry in qualifying}
            # Pixels belonging to a distinct, non-qualifying confirmed object
            # are never pulled into the union through the candidate.
            candidate_piece = _candidate_remainder(
                candidate_geometry,
                [
                    geometry
                    for segment, geometry in neighbours
                    if str(segment.id) not in qualifying_ids
                ],
            )
            merged_geometry: BaseGeometry | None = candidate_piece
            for _segment, geometry in qualifying:
                merged_geometry = safe_union(merged_geometry, geometry)
            merged_polygons = _post_exclusion_polygons(
                merged_geometry,
                min_area=1,
            )
            if not merged_polygons:
                # Defensive fallback: every qualifying existing object is real
                # geometry, so a failed union must preserve them, not delete them.
                filtered_candidates += 1
                candidate.delete()
                continue
            # Usually this is one connected union. If clipping against a
            # distinct confirmed object disconnects it, keep every component
            # and assign each old confirmed row to the component containing
            # most of its old area. No defensive repair may discard a person's
            # already-confirmed geometry.
            members_by_polygon: list[list[SegmentObject]] = [[] for _polygon in merged_polygons]
            for segment, geometry in qualifying:
                best_index = max(
                    range(len(merged_polygons)),
                    key=lambda index: geometry_area(
                        safe_intersection(merged_polygons[index], geometry)
                    ),
                )
                members_by_polygon[best_index].append(segment)

            for polygon, members in zip(
                merged_polygons,
                members_by_polygon,
                strict=True,
            ):
                if not members:
                    SegmentObject.objects.create(
                        segmentation=segmentation,
                        geometry=polygon,
                        centroid=polygon.centroid,
                        bbox=polygon.envelope,
                        label_state="CONFIRMED",
                        refined="UNREFINED",
                        source_model=candidate.source_model,
                        confidence_score=None,
                        features=dict(candidate.features or {}),
                        run_version=candidate.run_version,
                    )
                    created_fragments += 1
                    continue

                primary = min(members, key=_confirmed_primary_key)
                primary.geometry = polygon
                primary.centroid = polygon.centroid
                primary.bbox = polygon.envelope
                primary.save(update_fields=["geometry", "centroid", "bbox"])
                current_geometry[str(primary.id)] = polygon
                redundant_ids = [str(segment.id) for segment in members if segment.id != primary.id]
                if redundant_ids:
                    SegmentObject.objects.filter(
                        segmentation=segmentation,
                        id__in=redundant_ids,
                    ).delete()
                    deleted_confirmed_ids.update(redundant_ids)
                    deleted_confirmed += len(redundant_ids)
            candidate.delete()
            accepted_candidates += 1
            merged_candidates += 1
            continue

        remainder = _candidate_remainder(
            candidate_geometry,
            [geometry for _segment, geometry in neighbours],
        )
        pieces = _post_exclusion_polygons(remainder, min_area=min_area)
        if not pieces:
            candidate.delete()
            filtered_candidates += 1
            continue

        pieces.sort(key=geometry_area, reverse=True)
        primary_piece = pieces[0]
        candidate.geometry = primary_piece
        candidate.centroid = primary_piece.centroid
        candidate.bbox = primary_piece.envelope
        candidate.label_state = "CONFIRMED"
        candidate.refined = "UNREFINED"
        candidate.confidence_score = None
        candidate.status = status_for_segment_lifecycle(
            label_state=candidate.label_state,
            refined=candidate.refined,
        )
        candidate.save(
            update_fields=[
                "geometry",
                "centroid",
                "bbox",
                "label_state",
                "refined",
                "confidence_score",
                "status",
            ]
        )
        for piece in pieces[1:]:
            SegmentObject.objects.create(
                segmentation=segmentation,
                geometry=piece,
                centroid=piece.centroid,
                bbox=piece.envelope,
                label_state="CONFIRMED",
                refined="UNREFINED",
                source_model=candidate.source_model,
                confidence_score=None,
                features=dict(candidate.features or {}),
                run_version=candidate.run_version,
            )
        accepted_candidates += 1
        clipped_candidates += 1
        created_fragments += max(0, len(pieces) - 1)

    if plain_confirmations:
        SegmentObject.objects.bulk_update(
            plain_confirmations,
            ["label_state", "refined", "confidence_score", "status"],
            batch_size=500,
        )

    return {
        "confirmed_count": accepted_candidates,
        "merged_candidate_count": merged_candidates,
        "clipped_candidate_count": clipped_candidates,
        "filtered_after_overlap_count": filtered_candidates,
        "created_fragment_count": created_fragments,
        "deleted_confirmed_count": deleted_confirmed,
    }


def _refusal(detail: str, *, code: ErrorCode | None = None) -> Response:
    """A 409 the client can both read and act on.

    409 rather than 400: nothing about the request is malformed. It conflicts
    with the state of the stored result, which is a state the user can change --
    by running the model once -- and the body says so.
    """
    payload: dict[str, object] = {"detail": detail}
    if code is not None:
        payload[ERROR_CODE_FIELD] = str(code)
    return Response(payload, status=status.HTTP_409_CONFLICT)


def _resolve_segmenter(segmentation: ImageSegmentation, source_model: str):
    """``(segmenter, model_name)``, or ``(None, refusal)`` when the dial cannot move."""
    segmenter_internal_name = resolve_segmenter_internal_name(
        segmentation_type_internal_name=segmentation.segmentation_type.internal_name,
        source_model=source_model,
    )
    # QuantEM and OmniEM share one segmenter class per organelle.  The selected
    # pack is therefore constructor state, not something the registry key can
    # express.  Dropping it here silently turns every OmniEM replay into a
    # QuantEM replay while the route continues to count OmniEM candidates.
    segmenter = get_segmenter_or_none(
        segmenter_internal_name,
        source_model=source_model,
    )
    if segmenter is None:
        return None, MODEL_UNAVAILABLE_MESSAGE

    model_names = list(segmenter.get_dl_model_names())
    if len(model_names) != 1:
        return None, MULTI_OUTPUT_MESSAGE
    return segmenter, model_names[0]


def _dial_state(
    segmentation: ImageSegmentation,
    source_model: str,
    *,
    roi: ImageROI | None = None,
) -> dict:
    """Everything the control needs to render itself, including why it cannot move.

    One payload for both verbs, so the sentence a greyed-out dial shows and the
    sentence a refused move returns are the same sentence from the same check.
    Two derivations of "can this move" is how a control comes to look available
    and then fail when it is used.
    """
    segmenter, resolved = _resolve_segmenter(segmentation, source_model)
    run_version = SegmentationResultVersion.current_version_for(segmentation)
    effective_source_model = _effective_source_model(segmentation, source_model)
    state: dict[str, object] = {
        "include_level": segmentation.include_level,
        "default_include_level": None,
        "minimum": INCLUDE_LEVEL_MIN,
        "maximum": INCLUDE_LEVEL_MAX,
        "run_version": run_version,
        "measurement_mode": segmentation.segmentation_type.measurement_mode,
        "object_count": SegmentObject.objects.filter(
            segmentation=segmentation,
            superseded_at__isnull=True,
        )
        .exclude(label_state="EXCLUDED")
        .count(),
        "can_move": False,
        "detail": "",
    }
    state.update(_model_output_counts(segmentation, effective_source_model))

    if segmenter is None:
        state["detail"] = resolved
        return state

    state["default_include_level"] = getattr(segmenter, "fg_threshold", None)

    readiness = stored_map_readiness(
        segmentation=segmentation,
        segmenter=segmenter,
        model_name=resolved,
        roi=roi,
    )
    if not readiness.ready:
        state["detail"] = readiness.detail
        state[ERROR_CODE_FIELD] = str(ErrorCode.PROBABILITY_MAP_MISSING)
        return state

    state["can_move"] = True
    query_params: dict[str, str] = {}
    if source_model:
        query_params["source_model"] = source_model
    if roi is not None:
        query_params["roi_id"] = str(roi.id)
        state["preview_bounds"] = [
            int(roi.x),
            int(roi.y),
            int(roi.width),
            int(roi.height),
        ]
    else:
        full_map_path = get_prob_map_file_path(
            segmentation,
            resolved,
            str(getattr(segmenter, "prob_map_prefix", "") or ""),
        )
        width, height = probability_map_size(full_map_path)
        state["preview_bounds"] = [
            0,
            0,
            width,
            height,
        ]
    query = urlencode(query_params)
    preview_url = reverse("segmentation-include-level-map", args=[segmentation.id])
    state["preview_url"] = f"{preview_url}{'?' + query if query else ''}"
    return state


class SegmentationIncludeLevelMapView(APIView):
    """Serve the saved grayscale result used by the live threshold preview."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        source_model = normalize_source_model(request.query_params.get("source_model"))
        roi_id = request.query_params.get("roi_id")
        roi = None
        if roi_id:
            roi = ImageROI.objects.filter(asset=segmentation.asset, id=roi_id).first()
            if roi is None:
                return _refusal("That region is no longer on this image.")
        segmenter, resolved = _resolve_segmenter(segmentation, source_model)
        if segmenter is None:
            return _refusal(resolved)

        readiness = stored_map_readiness(
            segmentation=segmentation,
            segmenter=segmenter,
            model_name=resolved,
            roi=roi,
        )
        if not readiness.ready:
            return _refusal(readiness.detail, code=ErrorCode.PROBABILITY_MAP_MISSING)

        file_path = get_prob_map_file_path(
            segmentation,
            resolved,
            str(getattr(segmenter, "prob_map_prefix", "") or ""),
            str(roi.id) if roi is not None else None,
        )
        preview_path = ensure_probability_preview(file_path)
        response = FileResponse(preview_path.open("rb"), content_type="image/png")
        response["Cache-Control"] = "no-store"
        return response


class SegmentationIncludeLevelView(APIView):
    """Read the dial, or ask for it to be moved.

    ``GET`` is free: two filesystem stats and two indexed queries, no map
    decoded. It is meant to be called whenever the panel opens.

    ``POST`` queues one re-extract. It is the explicit Preview action, never a
    slider event: one request replaces the prior candidate set once.
    """

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        source_model = normalize_source_model(request.query_params.get("source_model"))
        roi_id = request.query_params.get("roi_id")
        roi = None
        if roi_id:
            roi = ImageROI.objects.filter(asset=segmentation.asset, id=roi_id).first()
            if roi is None:
                return _refusal("That region is no longer on this image.")
        return Response(
            _dial_state(segmentation, source_model, roi=roi),
            status=status.HTTP_200_OK,
        )

    def post(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )

        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked

        serializer = IncludeLevelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        include_level = float(serializer.validated_data["include_level"])
        source_model = normalize_source_model(serializer.validated_data.get("source_model"))
        roi_id = serializer.validated_data.get("roi_id")

        segmenter, resolved = _resolve_segmenter(segmentation, source_model)
        if segmenter is None:
            return _refusal(resolved)

        roi = None
        if roi_id is not None:
            roi = ImageROI.objects.filter(asset=segmentation.asset, id=roi_id).first()
            if roi is None:
                return _refusal("That region is no longer on this image.")

        readiness = stored_map_readiness(
            segmentation=segmentation,
            segmenter=segmenter,
            model_name=resolved,
            roi=roi,
        )
        if not readiness.ready:
            return _refusal(readiness.detail, code=ErrorCode.PROBABILITY_MAP_MISSING)

        # The user is telling us nothing is running. If the stage says otherwise
        # because a worker died mid-run, correct it now rather than refusing a
        # dial move on the strength of a phantom.
        reconcile_segmentation_status(segmentation)
        blocking_job = active_segmentation_job(
            segmentation,
            job_types=_ORGANELLE_ACTION_JOB_TYPES,
        )
        if blocking_job is not None:
            return Response(
                blocking_job_response_payload(blocking_job),
                status=status.HTTP_409_CONFLICT,
            )

        payload: dict[str, object] = {
            # Required, and not by convention: this job type is in
            # ACTIVE_SEGMENTATION_JOB_TYPES, whose failure reconcilers read this
            # exact key to release an image whose worker died.
            "segmentation_id": str(segmentation.id),
            "segmentation_type": segmentation.segmentation_type.internal_name,
            "include_level": include_level,
        }
        if source_model:
            payload["source_model"] = source_model
        if roi is not None:
            payload["roi_id"] = str(roi.id)

        job = Job.enqueue(
            job_type=JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
            payload=payload,
            priority="high",
            resource_class="cpu",
            queue_name=QUEUE_P1_INTERACTIVE,
            # Deterministic failures only: a missing or unreadable stored map
            # does not improve on a second attempt, and the user is holding the
            # dial waiting for an answer.
            max_attempts=1,
            tags=[f"segmentation:{seg_id}"],
        )
        logger.info(
            "Queued a re-extract of segmentation %s at include level %s",
            segmentation.id,
            include_level,
        )
        return Response(
            {"job_id": str(job.id), "include_level": include_level},
            status=status.HTTP_202_ACCEPTED,
        )


class SegmentationConfirmModelOutputView(APIView):
    """Confirm this model's whole-image candidates outside manually reviewed ROIs.

    This is intentionally server-side. A large image's left panel may hold only
    the objects in or near the viewport, so turning the currently rendered IDs
    into a frontend batch would silently confirm part of an image while the
    button says it confirmed the whole result.
    """

    def post(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked

        serializer = ConfirmModelOutputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        source_model = _effective_source_model(
            segmentation,
            serializer.validated_data["source_model"],
        )
        definition = get_source_model_definition(source_model)
        if (
            definition is None
            or definition.organelle_internal_name != segmentation.segmentation_type.internal_name
        ):
            return Response(
                {"detail": "Choose a model that belongs to this segmentation."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if segmentation.segmentation_type.measurement_mode != "objects":
            return Response(
                {
                    "detail": (
                        "This segmentation is a single foreground mask, not a set "
                        "of candidate objects. Its preview is already ready for analysis."
                    )
                },
                status=status.HTTP_409_CONFLICT,
            )

        segmenter, resolved = _resolve_segmenter(segmentation, source_model)
        if segmenter is None:
            return _refusal(resolved)
        min_area = resolve_min_area(segmenter, None)

        reconcile_segmentation_status(segmentation)
        blocking_job = active_segmentation_job(
            segmentation,
            job_types=_ORGANELLE_ACTION_JOB_TYPES,
        )
        if blocking_job is not None:
            return Response(
                blocking_job_response_payload(blocking_job),
                status=status.HTTP_409_CONFLICT,
            )

        with transaction.atomic():
            confirmable, protected, manual_roi_count = _partition_model_candidates(
                segmentation,
                source_model,
            )
            confirmation = _confirm_model_candidates(
                segmentation=segmentation,
                candidates=confirmable,
                min_area=min_area,
            )
            changed = bool(
                confirmation["confirmed_count"]
                or confirmation["filtered_after_overlap_count"]
                or confirmation["deleted_confirmed_count"]
            )
            if changed:
                overlay = register_overlay_mutation_all_bundles(
                    segmentation,
                    dirty_bbox=full_image_dirty_bbox(segmentation),
                    source_model=source_model,
                    allow_sync_partial=False,
                )
            else:
                overlay = None

        if changed:
            _enqueue_segment_feature_refresh(
                segmentation_id=str(segmentation.id),
                segment_ids=[],
                recompute_features=True,
            )

        return Response(
            {
                "segmentation_id": str(segmentation.id),
                "source_model": source_model,
                **confirmation,
                "min_area": min_area,
                "skipped_manual_roi_count": len(protected),
                "manual_roi_count": manual_roi_count,
                "remaining_candidate_count": len(protected),
                "overlay": overlay,
            },
            status=status.HTTP_200_OK,
        )


urlpatterns = [
    path(
        "segmentations/<uuid:seg_id>/include-level/map",
        SegmentationIncludeLevelMapView.as_view(),
        name="segmentation-include-level-map",
    ),
    path(
        "segmentations/<uuid:seg_id>/include-level",
        SegmentationIncludeLevelView.as_view(),
        name="segmentation-include-level",
    ),
    path(
        "segmentations/<uuid:seg_id>/confirm-model-output",
        SegmentationConfirmModelOutputView.as_view(),
        name="segmentation-confirm-model-output",
    ),
]
