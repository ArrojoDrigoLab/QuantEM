"""Segmentation-level list/create/status/delete endpoints."""

from __future__ import annotations

import logging
import shutil

from django.apps import apps
from django.db import transaction
from django.db.models import Count, Prefetch
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.asset_resolver import get_active_asset
from quantem.assets.roi_state import (
    get_active_roi_for_segmentation,
    initialize_roi_from_template,
)
from quantem.jobs.constants import JOB_TYPE_RUN_SEGMENTATION_ROI, QUEUE_P3_ROI
from quantem.jobs.models import Job
from quantem.seg_core.registry import get_segmenter_or_none
from quantem.segmentation.completion import (
    completion_preview,
    is_locked,
    locked_payload,
    restore_last_archive,
)
from quantem.segmentation.models import (
    AnalysisMaskObject,
    ImageSegmentation,
    ProbabilityMap,
    SegmentationConfig,
    SegmentationType,
)
from quantem.segmentation.serializers import (
    ImageSegmentationCreateSerializer,
    ImageSegmentationSerializer,
    ProbabilityMapSerializer,
)
from quantem.segmentation.source_models import (
    default_source_model_for_organelle,
    normalize_source_model,
    resolve_create_segmentation_request,
    resolve_segmenter_internal_name,
)
from quantem.segmentation.status_reconcile import (
    reconcile_segmentation_status,
    reconcile_segmentation_statuses,
)
from quantem.segmentation.type_definitions import (
    ANALYSIS_MASK,
    ORGANELLE_INTERNAL_NAMES,
    find_builtin_segmentation_type,
)
from quantem.segmentation.type_service import (
    ensure_segmentation_type,
    resolve_or_create_segmentation_type,
)

logger = logging.getLogger(__name__)


def _resolve_public_asset_image(asset_id):
    return get_active_asset(asset_id)


def _parse_bool(raw: object) -> bool | None:
    """Strict-ish bool parse. None means "the caller did not say"."""
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


class AssetSegmentationListCreateView(APIView):
    """List and create segmentations for a canonical asset."""

    def get(self, request, asset_id):
        asset = _resolve_public_asset_image(asset_id)
        segmentations = ImageSegmentation.objects.filter(asset=asset, superseded_by__isnull=True)
        # ``asset`` is joined for ``objects_pixel_size``, which compares the
        # image's pixel size now with the one its objects were produced under.
        # Without it that is one extra query per row of a list a screen polls.
        segmentations = segmentations.select_related(
            "segmentation_type", "config", "asset"
        ).prefetch_related(
            Prefetch(
                "analysis_mask_objects",
                queryset=AnalysisMaskObject.objects.filter(
                    segmentation__segmentation_type__representation=(
                        SegmentationType.REPRESENTATION_SEMANTIC
                    )
                ).only("id", "segmentation_id", "name"),
                to_attr="semantic_display_objects",
            )
        )
        # A run whose worker died leaves its stage behind, and this is the read
        # the labeling screen makes: without the repair the screen shows a
        # "Running... 40%" pill for a run with no process behind it until the
        # job heartbeat goes stale minutes later.
        segmentations = reconcile_segmentation_statuses(segmentations)
        serializer = ImageSegmentationSerializer(segmentations, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, asset_id):
        asset = _resolve_public_asset_image(asset_id)
        return _create_segmentation_for_asset(request, asset=asset)


def _create_segmentation_for_asset(request, *, asset):
    create_serializer = ImageSegmentationCreateSerializer(data=request.data)
    if not create_serializer.is_valid():
        return Response(
            create_serializer.errors,
            status=status.HTTP_400_BAD_REQUEST,
        )

    validated_data = create_serializer.validated_data
    run_inference = bool(validated_data.get("run_inference", True))
    segmentation_type = None
    source_model = normalize_source_model(validated_data.get("source_model"))
    measurement_mode = validated_data.get("measurement_mode")
    if validated_data.get("segmentation_type_id"):
        segmentation_type = get_object_or_404(
            SegmentationType,
            id=validated_data["segmentation_type_id"],
        )
    elif validated_data.get("segmentation_type_name"):
        requested_name = validated_data["segmentation_type_name"]
        builtin_definition = find_builtin_segmentation_type(requested_name)
        if builtin_definition is not None:
            canonical_definition, resolved_source_model = resolve_create_segmentation_request(
                builtin_definition,
                source_model,
            )
            segmentation_type = ensure_segmentation_type(canonical_definition)
            source_model = resolved_source_model
        else:
            segmentation_type = resolve_or_create_segmentation_type(
                requested_name,
                measurement_mode=(measurement_mode or SegmentationType.MEASUREMENT_MODE_OBJECTS),
            )
            source_model = source_model or default_source_model_for_organelle(
                segmentation_type.internal_name
            )
    is_analysis_mask = segmentation_type.internal_name == ANALYSIS_MASK.internal_name
    analysis_name = (validated_data.get("analysis_name") or "").strip()
    if is_analysis_mask and not analysis_name:
        return Response(
            {"analysis_name": ["Give this analysis mask a name."]},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if not is_analysis_mask and analysis_name:
        return Response(
            {
                "analysis_name": [
                    "An analysis name can only be used with Analysis Segmentation Mask."
                ]
            },
            status=status.HTTP_400_BAD_REQUEST,
        )
    if measurement_mode and (
        validated_data.get("segmentation_type_id")
        or find_builtin_segmentation_type(segmentation_type.internal_name)
    ):
        return Response(
            {
                "measurement_mode": [
                    "Choose a measurement mode only when creating a new custom segmentation."
                ]
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Only the four released organelles are model-backed. Area definitions and
    # reusable custom organelles open directly into manual labeling workflows.
    manual_only_workflow = segmentation_type.internal_name not in ORGANELLE_INTERNAL_NAMES
    queue_inference = not manual_only_workflow and run_inference
    if queue_inference:
        source_model = source_model or default_source_model_for_organelle(
            segmentation_type.internal_name
        )
    else:
        source_model = None
    image_segmentation, created = ImageSegmentation.objects.get_or_create(
        asset=asset,
        segmentation_type=segmentation_type,
        display_name=analysis_name if is_analysis_mask else "",
        superseded_by__isnull=True,
    )
    # This endpoint is get_or_create, and it queues a run for what it returns.
    # For an organelle already marked done that would start a run over a locked
    # segmentation -- the same refusal the explicit run endpoints make.
    if not created and is_locked(image_segmentation):
        return Response(
            locked_payload(image_segmentation),
            status=status.HTTP_409_CONFLICT,
        )
    update_fields = []
    if image_segmentation.asset_id is None:
        image_segmentation.asset = asset
        update_fields.append("asset")
    if update_fields:
        image_segmentation.save(update_fields=[*update_fields, "updated_at"])

    if not queue_inference:
        updates: list[str] = []
        if created and image_segmentation.status_stage != "CANDIDATES_READY":
            image_segmentation.status_stage = "CANDIDATES_READY"
            updates.append("status_stage")
        if created and image_segmentation.status_progress != 100.0:
            image_segmentation.status_progress = 100.0
            updates.append("status_progress")
        if updates:
            image_segmentation.save(update_fields=updates)
    if not manual_only_workflow:
        initialize_roi_from_template(image_segmentation)

        SegmentationConfig.objects.get_or_create(segmentation=image_segmentation)

    segmenter_internal_name = None
    if queue_inference:
        segmenter_internal_name = resolve_segmenter_internal_name(
            segmentation_type_internal_name=segmentation_type.internal_name,
            source_model=source_model,
        )
    segmenter = (
        None
        if not queue_inference
        else get_segmenter_or_none(
            segmenter_internal_name or segmentation_type.internal_name,
            source_model=source_model,
        )
    )
    if segmenter is not None:
        tags = [
            f"asset:{asset.id}",
            f"segmentation:{image_segmentation.id}",
            f"segmentation_type:{segmentation_type.internal_name}",
            f"source_model:{source_model}",
        ]
        roi_image = get_active_roi_for_segmentation(image_segmentation)
        roi_id = str(roi_image.id) if roi_image else None
        Job.enqueue(
            job_type=JOB_TYPE_RUN_SEGMENTATION_ROI,
            payload={
                "segmentation_id": str(image_segmentation.id),
                "segmentation_type": segmentation_type.internal_name,
                "roi_id": roi_id,
                "source_model": source_model,
                "asset_id": str(asset.id),
            },
            priority="high",
            resource_class=segmenter.job_resource_class,
            queue_name=QUEUE_P3_ROI,
            tags=tags,
        )

    serializer = ImageSegmentationSerializer(image_segmentation)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


def _segmentation_delete_preview(segmentation: ImageSegmentation) -> dict:
    """Live counts of what deleting this segmentation destroys, and what stays.

    Read fresh whenever the confirm dialog opens, to the Mark-Done standard:
    the dialog quotes these numbers, ``DELETE`` compares its acknowledged
    object count against a fresh read, and a stale dialog gets a 409 with the
    current numbers instead of silently deleting objects nobody was shown.
    """
    from quantem.segmentation.models import SegmentObject  # noqa: PLC0415

    label_counts = {
        row["label_state"]: row["n"]
        for row in SegmentObject.objects.filter(segmentation=segmentation)
        .values("label_state")
        .annotate(n=Count("id"))
        .order_by()
    }
    object_count = sum(label_counts.values())

    Adapter = apps.get_model("finetune", "Adapter")
    AnalysisRun = apps.get_model("analysis", "AnalysisRun")

    return {
        "segmentation_id": str(segmentation.id),
        "segmentation_type": segmentation.display_name or segmentation.segmentation_type.long_name,
        "object_count": int(object_count),
        "objects_by_label_state": {
            state: int(label_counts.get(state, 0))
            for state, _label in SegmentObject.LABEL_STATE_CHOICES
        },
        "probability_map_count": ProbabilityMap.objects.filter(segmentation=segmentation).count(),
        "overlay_count": segmentation.overlay_states.count(),
        "adapter_count": Adapter.objects.filter(segmentation=segmentation).count(),
        # Kept, not deleted: a run's numbers and its export bundle are the
        # record of an analysis that happened. They survive with
        # segmentation_deleted: true in their payloads.
        "analysis_run_count": AnalysisRun.objects.filter(segmentation=segmentation).count(),
        "locked": is_locked(segmentation),
    }


def _remove_segmentation_files(segmentation_id: str) -> None:
    """Best-effort removal of this segmentation's on-disk artifacts.

    Called *after* the database rows are gone: a leftover directory with no row
    pointing at it is an orphan nothing will ever read, while a row whose files
    were deleted first would claim artifacts that do not exist. Failures are
    logged and swallowed for the same reason -- the deletion the user asked for
    has already happened.
    """
    from quantem.core.config import (  # noqa: PLC0415
        DATA_DIR,
        GLOBAL_MASKS_DIR,
        PROB_MAPS_DIR,
        TMP_DIR,
    )
    from quantem.segmentation.overlay_ngff.paths import get_overlay_root  # noqa: PLC0415

    for path in (
        get_overlay_root(segmentation_id),
        GLOBAL_MASKS_DIR / segmentation_id,
        PROB_MAPS_DIR / segmentation_id,
        TMP_DIR / "prob_maps" / segmentation_id,
        DATA_DIR / "correction_revisions" / segmentation_id,
    ):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:  # pragma: no cover - rmtree(ignore_errors) rarely raises
            logger.warning(
                "Could not remove %s while deleting segmentation %s.",
                path,
                segmentation_id,
                exc_info=True,
            )


class SegmentationDetailView(APIView):
    """One segmentation: read it, or delete it and everything it owns.

    ``GET``
        The serialized segmentation plus ``delete_preview`` -- the live counts a
        deletion confirm dialog must quote (objects by label state, probability
        maps, overlays, adapters, and the analysis runs that would be kept).

    ``DELETE``
        Deletes the segmentation and everything that belongs to it: its objects
        (confirmed, rejected and unreviewed alike), its overlay rasters, its
        probability maps, correction revisions,
        its completed-ROI record and its feedback. Saved models and weights are kept. There
        is no undo short of running the model again. Refused with a 409 while a
        job is queued, running or retrying on the segmentation (the body names
        the task the way Tasks & Queues names it, and carries the job's id and
        type as fields for the client) and while the segmentation is locked by
        Mark Image Done (unlock first -- the lock exists so "done" stays
        final, and deletion is the strongest possible mutation).

        Analysis runs that reference the segmentation are deliberately **not**
        deleted: the run and its export bundle are the record of an analysis
        that happened. They survive with ``segmentation`` set to null and
        ``segmentation_deleted: true`` in their payloads.

        The optional ``acknowledged_object_count`` (body or query parameter)
        is the object count the user was shown; a mismatch -- usually a run
        that finished while the dialog was open -- is refused with a 409
        carrying the fresh preview, exactly like Mark Image Done's discard.
    """

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        reconcile_segmentation_status(segmentation)
        payload = dict(ImageSegmentationSerializer(segmentation).data)
        payload["delete_preview"] = _segmentation_delete_preview(segmentation)
        return Response(payload, status=status.HTTP_200_OK)

    def delete(self, request, seg_id):
        from quantem.segmentation.api_views.shared import (  # noqa: PLC0415
            active_segmentation_job,
            delete_blocked_response_payload,
        )

        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )

        # Any live job whose payload names this segmentation blocks the delete:
        # a full/ROI segmentation run, an analysis, an adapter training. Pulling
        # the rows out from under a worker mid-write is a crash with a
        # half-deleted segmentation behind it; the refusal names the job so the
        # user can cancel it and try again.
        job = active_segmentation_job(segmentation, job_types=None)
        if job is not None:
            # ``detail`` is rendered verbatim in the confirm dialog, so it says
            # what is happening and names the screen with the control on it.
            # See delete_blocked_response_payload for what it used to say.
            return Response(
                delete_blocked_response_payload(job),
                status=status.HTTP_409_CONFLICT,
            )

        # The completion lock refuses every mutation, and deletion is the
        # strongest one. locked_payload names the reason and the way out.
        if is_locked(segmentation):
            return Response(
                locked_payload(segmentation),
                status=status.HTTP_409_CONFLICT,
            )

        preview = _segmentation_delete_preview(segmentation)

        # Same contract as Mark Image Done's discard: the dialog quoted a
        # count, and if the count has moved since -- a run finished while it
        # was open -- nothing is deleted and the fresh numbers come back.
        acknowledged_raw = None
        if isinstance(request.data, dict) and "acknowledged_object_count" in request.data:
            acknowledged_raw = request.data.get("acknowledged_object_count")
        elif "acknowledged_object_count" in request.query_params:
            acknowledged_raw = request.query_params.get("acknowledged_object_count")
        if acknowledged_raw is not None:
            try:
                acknowledged = int(str(acknowledged_raw).strip())
            except (TypeError, ValueError):
                return Response(
                    {
                        # The field name belonged in this sentence when only a
                        # client could reach it; ``detail`` is rendered in the
                        # dialog, so it reads as English (I-12, internal-name).
                        "detail": (
                            "The object count sent with this delete could not "
                            "be read as a number. Nothing was deleted; reopen "
                            "the delete dialog and confirm again."
                        ),
                        "delete_preview": preview,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if acknowledged != preview["object_count"]:
                return Response(
                    {
                        "detail": (
                            f"This segmentation now holds "
                            f"{preview['object_count']} object(s), not "
                            f"{acknowledged}. Nothing was deleted. Re-confirm "
                            "against the current count."
                        ),
                        "delete_preview": preview,
                    },
                    status=status.HTTP_409_CONFLICT,
                )

        segmentation_id = str(segmentation.id)

        with transaction.atomic():
            # CASCADE takes the objects, overlays (states + labels), probability
            # map rows, completed ROIs, feedback, config and revision records.
            # AnalysisRun and Adapter use SET_NULL: saved measurements, model
            # identities and trained weights survive their source segmentation.
            segmentation.delete()

        _remove_segmentation_files(segmentation_id)

        return Response(
            {
                "deleted": preview,
                "analysis_runs_kept": preview["analysis_run_count"],
                "adapters_kept": preview["adapter_count"],
            },
            status=status.HTTP_200_OK,
        )


class ProbabilityMapListCreateView(APIView):
    """List probability maps for a segmentation."""

    def get(self, request, segmentation_id):
        segmentation = get_object_or_404(ImageSegmentation, id=segmentation_id)
        reconcile_segmentation_status(segmentation)
        prob_maps = ProbabilityMap.objects.filter(segmentation=segmentation)
        map_serializer = ProbabilityMapSerializer(prob_maps, many=True)

        return Response(
            {
                "segmentation": {
                    "id": str(segmentation.id),
                    "status_stage": segmentation.status_stage,
                    "status_progress": segmentation.status_progress,
                    "status_error": segmentation.status_error,
                },
                "probability_maps": map_serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class SegmentationCompleteView(APIView):
    """Confirm whole-image manual coverage; model runs supply coverage automatically."""

    def get(self, request, seg_id):
        from quantem.segmentation.coverage import coverage_state

        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        return Response(
            {**completion_preview(segmentation), "coverage": coverage_state(segmentation)}
        )

    def post(self, request, seg_id):
        from quantem.jobs.constants import ACTIVE_SEGMENTATION_JOB_TYPES
        from quantem.segmentation.api_views.shared import active_segmentation_job
        from quantem.segmentation.coverage import coverage_state

        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        job = active_segmentation_job(segmentation, job_types=ACTIVE_SEGMENTATION_JOB_TYPES)
        if job:
            from quantem.segmentation.api_views.shared import blocking_job_response_payload

            return Response(blocking_job_response_payload(job), status=409)
        if coverage_state(segmentation)["ready"]:
            return Response(ImageSegmentationSerializer(segmentation).data)
        if not (
            request.data.get("acknowledge_full_manual") is True
            and request.data.get("acknowledge_no_model") is True
        ):
            return Response(
                {
                    "detail": "Confirm that you have manually segmented the entire image and that no model outputs are included."
                },
                status=400,
            )
        from quantem.segmentation.models import SegmentObject

        generated = SegmentObject.objects.filter(
            segmentation=segmentation, superseded_at=None
        ).exclude(source_model="manual")
        from quantem.segmentation.models import CompletedROI, GlobalMask

        generated_mask = (
            GlobalMask.objects.filter(segmentation=segmentation)
            .exclude(source__startswith="manual")
            .exists()
        )
        if generated.exists() or generated_mask:
            return Response(
                {
                    "detail": "This image still contains model-derived objects. Mark completed ROIs or apply a full-image model result."
                },
                status=400,
            )
        segmentation.manual_full_image = True
        segmentation.status_stage = "CANDIDATES_READY"
        segmentation.status_error = ""
        from django.utils import timezone

        from quantem import __version__

        segmentation.final_result_provenance = {
            "quantem_version": __version__,
            "finalized_at": timezone.now().isoformat(),
            "model_identifier": "manual",
            "final_level_kind": "manual",
        }
        from shapely.geometry import box

        from quantem.analysis.superseded import mark_superseded

        with transaction.atomic():
            CompletedROI.objects.create(
                segmentation=segmentation,
                geometry=box(
                    0, 0, segmentation.asset.logical_width, segmentation.asset.logical_height
                ),
            )
            mark_superseded(segmentation)
            segmentation.save(
                update_fields=[
                    "manual_full_image",
                    "status_stage",
                    "status_error",
                    "final_result_provenance",
                    "updated_at",
                ]
            )
        return Response(ImageSegmentationSerializer(segmentation).data)

    def delete(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        # Compatibility for older installations: recover any objects archived by
        # the former completion action, without reclaiming prediction files.
        restored = restore_last_archive(segmentation)
        from quantem.segmentation.overlay_ngff import (
            queue_full_overlay_rebuild,
            register_state_mutation,
        )

        if restored:
            queue_full_overlay_rebuild(segmentation)
        else:
            register_state_mutation(segmentation)
        segmentation.manual_full_image = False
        if segmentation.status_stage == "COMPLETED":
            segmentation.status_stage = "CANDIDATES_READY"
        segmentation.save(update_fields=["manual_full_image", "status_stage", "updated_at"])
        return Response({**ImageSegmentationSerializer(segmentation).data, "restored": restored})
