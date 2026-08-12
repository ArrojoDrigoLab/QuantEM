"""Segmentation-level list/create/status/delete endpoints."""

from __future__ import annotations

import logging
import shutil

from django.apps import apps
from django.db import transaction
from django.db.models import Count
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.asset_resolver import get_active_asset
from quantem.assets.roi_state import get_active_roi_for_asset
from quantem.jobs.constants import JOB_TYPE_RUN_SEGMENTATION_ROI, QUEUE_P3_ROI
from quantem.jobs.models import Job
from quantem.seg_core.db.prob_maps import delete_probability_maps_for_segmentation
from quantem.seg_core.registry import get_segmenter_or_none
from quantem.segmentation.completion import (
    archive_and_discard,
    completion_preview,
    is_locked,
    locked_payload,
    restore_last_archive,
)
from quantem.segmentation.models import (
    ImageSegmentation,
    ProbabilityMap,
    SegmentationConfig,
    SegmentationType,
)
from quantem.segmentation.overlay_ngff import (
    full_image_dirty_bbox,
    register_overlay_mutation_all_bundles,
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
        segmentations = ImageSegmentation.objects.filter(asset=asset)
        # ``asset`` is joined for ``objects_pixel_size``, which compares the
        # image's pixel size now with the one its objects were produced under.
        # Without it that is one extra query per row of a list a screen polls.
        segmentations = segmentations.select_related("segmentation_type", "config", "asset")
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

    # Only the four released organelles are model-backed. Tissue, analysis,
    # and reusable custom segmentations open directly into equivalent manual
    # labeling workflows and never queue a model run.
    manual_only_workflow = segmentation_type.internal_name not in ORGANELLE_INTERNAL_NAMES
    if not manual_only_workflow:
        source_model = source_model or default_source_model_for_organelle(
            segmentation_type.internal_name
        )
    else:
        source_model = None
    image_segmentation, created = ImageSegmentation.objects.get_or_create(
        asset=asset,
        segmentation_type=segmentation_type,
        display_name=analysis_name if is_analysis_mask else "",
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

    if manual_only_workflow:
        updates: list[str] = []
        if created and image_segmentation.status_stage != "CANDIDATES_READY":
            image_segmentation.status_stage = "CANDIDATES_READY"
            updates.append("status_stage")
        if created and image_segmentation.status_progress != 100.0:
            image_segmentation.status_progress = 100.0
            updates.append("status_progress")
        if updates:
            image_segmentation.save(update_fields=updates)
    else:
        existing_roi = get_active_roi_for_asset(asset)
        if existing_roi:
            existing_roi.segmentations.add(image_segmentation)

        SegmentationConfig.objects.get_or_create(segmentation=image_segmentation)

    segmenter_internal_name = None
    if not manual_only_workflow:
        segmenter_internal_name = resolve_segmenter_internal_name(
            segmentation_type_internal_name=segmentation_type.internal_name,
            source_model=source_model,
        )
    segmenter = (
        None
        if manual_only_workflow
        else get_segmenter_or_none(segmenter_internal_name or segmentation_type.internal_name)
    )
    if segmenter is not None:
        tags = [
            f"asset:{asset.id}",
            f"segmentation:{image_segmentation.id}",
            f"segmentation_type:{segmentation_type.internal_name}",
            f"source_model:{source_model}",
        ]
        roi_image = get_active_roi_for_asset(asset)
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
    from quantem.core.config import PROB_MAPS_DIR, TMP_DIR  # noqa: PLC0415
    from quantem.segmentation.overlay_ngff.paths import get_overlay_root  # noqa: PLC0415

    for path in (
        get_overlay_root(segmentation_id),
        PROB_MAPS_DIR / segmentation_id,
        TMP_DIR / "prob_maps" / segmentation_id,
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
        probability maps, its adapters (including any trained head weights),
        its completed-ROI record and its feedback. Nothing is archived; there
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

        # Adapter head weights live on disk under MODELS_DIR and the row only
        # names the path, so collect the files before the CASCADE removes the
        # rows that know about them.
        Adapter = apps.get_model("finetune", "Adapter")
        head_files = [
            head_file
            for adapter in Adapter.objects.filter(segmentation=segmentation)
            if (head_file := adapter.head_file) is not None
        ]

        with transaction.atomic():
            # CASCADE takes the objects, overlays (states + labels), probability
            # map rows, completed ROIs, feedback, config, archives and adapters.
            # AnalysisRun.segmentation is SET_NULL: the runs and their export
            # bundles stay, marked segmentation_deleted in their payloads.
            segmentation.delete()

        for head_file in head_files:
            try:
                head_file.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not remove adapter head %s while deleting segmentation %s.",
                    head_file,
                    segmentation_id,
                    exc_info=True,
                )
        _remove_segmentation_files(segmentation_id)

        return Response(
            {
                "deleted": preview,
                "analysis_runs_kept": preview["analysis_run_count"],
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
    """Lock a segmentation, and optionally prune what nobody confirmed.

    ``GET``
        The read-only preview: how many objects a discard would destroy, broken
        down by label state and source model, and whether it could be undone.
        Changes nothing. This is what a confirmation dialog needs in order to
        name a number, and there was previously no way to ask.

    ``POST``
        Marks the segmentation ``COMPLETED``. **Keeps every object unless the
        request explicitly asks otherwise**: ``discard_unconfirmed`` defaults to
        false, so no client can destroy a run's output without meaning to. When
        it is true, ``acknowledged_discard_count`` must equal what is actually
        there -- a client whose count is stale gets a 409 carrying the fresh
        preview rather than silently deleting objects the user was never shown.
        A discard is archived first and can be undone with ``DELETE``.

    ``DELETE``
        Unlocks the segmentation and restores the objects the last completion
        discarded. It used to only flip ``status_stage`` back and restore
        nothing, which made "unlock" a word for "the button is available again"
        rather than an undo.
    """

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        return Response(completion_preview(segmentation), status=status.HTTP_200_OK)

    def post(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        payload = request.data if isinstance(request.data, dict) else {}

        # ``POST`` marks done; it has no "and also undo it" mode. A body saying
        # ``is_complete: false`` was read by nobody, so the request locked the
        # segmentation and answered 200 -- the opposite of what it asked for,
        # reported as success. Refuse it and name the verb that unlocks.
        raw_is_complete = payload.get("is_complete")
        if raw_is_complete is not None and _parse_bool(raw_is_complete) is not True:
            return Response(
                {
                    # HTTP verbs are for the ``unlock`` block below, which is
                    # what a client reads; a sentence naming them is I-12's
                    # http-verb class and tells a reader nothing they can act
                    # on. "Unlock segmentation" is the control's own caption.
                    "detail": (
                        "Marking this segmentation done cannot undo it. Use "
                        "Unlock segmentation on the labeling screen to unlock "
                        "it and restore whatever the completion discarded."
                    ),
                    "unlock": {
                        "method": "DELETE",
                        "path": f"/api/segmentations/{segmentation.id}/complete",
                    },
                    "preview": completion_preview(segmentation),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        raw_discard = payload.get("discard_unconfirmed")
        discard = _parse_bool(raw_discard)
        if raw_discard is not None and discard is None:
            return Response(
                {
                    "detail": (
                        "discard_unconfirmed must be a boolean. Omit it to keep every object."
                    ),
                    "preview": completion_preview(segmentation),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        # The safe default. An older client, a retry, a hand-rolled curl -- none
        # of them can delete a run's output by leaving a field out.
        discard = bool(discard)

        outcome = {
            "discarded_count": 0,
            "restorable": False,
            "archive_id": None,
        }
        if discard:
            preview = completion_preview(segmentation)
            acknowledged = payload.get("acknowledged_discard_count")
            if not isinstance(acknowledged, int) or isinstance(acknowledged, bool):
                try:
                    acknowledged = int(str(acknowledged).strip())
                except (TypeError, ValueError):
                    return Response(
                        {
                            "detail": (
                                "discard_unconfirmed requires "
                                "acknowledged_discard_count: the number of objects "
                                "the user was shown before agreeing to delete them."
                            ),
                            "preview": preview,
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )
            if acknowledged != preview["discard_count"]:
                # Usually an inference run finished while the dialog was open.
                # Refusing costs a re-confirm; not refusing destroys objects
                # nobody was told about.
                return Response(
                    {
                        "detail": (
                            f"This segmentation now holds {preview['discard_count']} "
                            f"unconfirmed object(s), not {acknowledged}. Nothing was "
                            "deleted. Re-confirm against the current count."
                        ),
                        "preview": preview,
                    },
                    status=status.HTTP_409_CONFLICT,
                )

        with transaction.atomic():
            if discard:
                outcome = archive_and_discard(segmentation)
            segmentation.status_stage = "COMPLETED"
            segmentation.save(update_fields=["status_stage"])
            register_overlay_mutation_all_bundles(
                segmentation,
                dirty_bbox=full_image_dirty_bbox(segmentation),
                force_full_rebuild=True,
            )

        removed_probability_maps = delete_probability_maps_for_segmentation(segmentation)
        logger.info(
            "Reclaimed %d probability map(s) for completed segmentation %s.",
            removed_probability_maps,
            segmentation.id,
        )

        serializer = ImageSegmentationSerializer(segmentation)
        response_payload = dict(serializer.data)
        response_payload["completion"] = {
            "discarded_unconfirmed": discard,
            "discarded_count": int(outcome["discarded_count"]),
            "restorable": bool(outcome["restorable"]),
            "archive_id": outcome["archive_id"],
        }
        return Response(response_payload, status=status.HTTP_200_OK)

    def delete(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        restored = restore_last_archive(segmentation)
        if segmentation.status_stage == "COMPLETED":
            segmentation.status_stage = "CANDIDATES_READY"
            segmentation.save(update_fields=["status_stage"])
        if restored["restored_count"]:
            register_overlay_mutation_all_bundles(
                segmentation,
                dirty_bbox=full_image_dirty_bbox(segmentation),
                force_full_rebuild=True,
            )
        serializer = ImageSegmentationSerializer(segmentation)
        response_payload = dict(serializer.data)
        response_payload["restored"] = {
            "restored_count": int(restored["restored_count"]),
            "archived_count": int(restored["archived_count"]),
            # False with a non-zero archived_count means the discard was too
            # large to archive: those objects are gone and the UI must not
            # promise otherwise.
            "restorable": bool(restored["restorable"]),
        }
        return Response(response_payload, status=status.HTTP_200_OK)
