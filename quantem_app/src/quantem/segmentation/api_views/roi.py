"""ROI placement, initialization and completion for editable segmentations."""

from __future__ import annotations

import logging
import os

from django.db import transaction
from django.shortcuts import get_object_or_404
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.roi_state import (
    activate_roi,
    get_active_roi_for_segmentation,
    rois_for_segmentation,
)
from quantem.assets.utils import create_roi_image_from_image
from quantem.core.config import ROIS_DIR
from quantem.seg_core.db.prob_maps import delete_probability_maps_for_roi
from quantem.segmentation.models import ImageSegmentation, RoiSegmentationStatus, SegmentObject
from quantem.segmentation.mutations import (
    register_overlay_mutation_all_bundles,
)
from quantem.segmentation.overlay_ngff import (
    merge_dirty_bboxes,
)
from quantem.segmentation.roi_selection import select_roi_for_image
from quantem.segmentation.serializers import (
    CompletedRoiCreateSerializer,
    CompletedRoiSerializer,
    SegmentationRoiSerializer,
    SegmentObjectSerializer,
)
from quantem.segmentation.services.completed_rois import (
    list_completed_rois,
    save_completed_roi,
    subtract_completed_roi,
)
from quantem.segmentation.services.confirm_batch.overlap import (
    delete_candidates_intersecting_geometry,
)
from quantem.segmentation.services.spatial_lookup import bbox_intersects_filter, make_bbox
from quantem.segmentation.source_models import normalize_source_model, source_model_queryset_filter

from .shared import (
    completion_lock_response,
    get_or_create_roi_image,
    get_segmentation_target_image,
)


class SegmentationRoiActivateSerializer(serializers.Serializer):
    roi_id = serializers.UUIDField()


def _delete_candidates_for_completed_roi(segmentation, roi_image):
    """Apply the ROI Done any-overlap rule."""
    roi_geometry = make_bbox(
        roi_image.x,
        roi_image.y,
        roi_image.x + roi_image.width,
        roi_image.y + roi_image.height,
    )
    return delete_candidates_intersecting_geometry(
        segmentation=segmentation,
        region=roi_geometry,
    )


def _register_candidate_cleanup_overlay(segmentation, cleanup):
    """Queue the candidate raster update after the deletion transaction commits."""
    if cleanup.deleted_ids:
        return register_overlay_mutation_all_bundles(
            segmentation,
            dirty_bbox=merge_dirty_bboxes(segmentation, cleanup.affected_geometries),
            allow_sync_partial=False,
        )
    return None


class CompletedRoiListCreateView(APIView):
    """List or create user-marked completed ROIs for a segmentation."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        serializer = CompletedRoiSerializer(
            list_completed_rois(segmentation),
            many=True,
        )
        return Response(serializer.data, status=status.HTTP_200_OK)

    @transaction.atomic
    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        # A completed area is proofreading state that belongs to this
        # segmentation, so it follows this segmentation's lock.
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        serializer = CompletedRoiCreateSerializer(data=request.data or {})
        serializer.is_valid(raise_exception=True)

        try:
            completed_roi, created = save_completed_roi(
                segmentation=segmentation,
                polygon_coords=serializer.validated_data["polygon_coords"],
            )
        except ValueError as exc:
            return Response(
                {"error": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        response_serializer = CompletedRoiSerializer(completed_roi)
        return Response(
            response_serializer.data,
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


class CompletedRoiSubtractView(APIView):
    """Subtract a freehand polygon from the confirmed-area layer."""

    @transaction.atomic
    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        serializer = CompletedRoiCreateSerializer(data=request.data or {})
        serializer.is_valid(raise_exception=True)

        try:
            summary = subtract_completed_roi(
                segmentation=segmentation,
                polygon_coords=serializer.validated_data["polygon_coords"],
            )
        except ValueError as exc:
            return Response(
                {"error": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        completed_rois = CompletedRoiSerializer(
            list_completed_rois(segmentation),
            many=True,
        ).data
        return Response(
            {**summary, "completed_rois": completed_rois},
            status=status.HTTP_200_OK,
        )


class SegmentationRoiListCreateView(APIView):
    """List or create ROIs for a segmentation."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        target_image = get_segmentation_target_image(segmentation)

        rois = list(
            _roi_queryset_for_segmentation(segmentation).order_by(
                "created_at",
                "id",
            )
        )
        if not rois:
            roi_image = get_or_create_roi_image(target_image, segmentation)
            rois = [roi_image] if roi_image else []

        serializer = SegmentationRoiSerializer(
            rois,
            many=True,
            context={"segmentation": segmentation},
        )
        return Response(serializer.data, status=status.HTTP_200_OK)

    @transaction.atomic
    def post(self, request, seg_id):
        if "prepopulate" in request.data and not isinstance(request.data["prepopulate"], bool):
            return Response({"detail": "Choose whether to prepopulate this ROI."}, status=400)
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        # Refused before anything is read off the request, so a locked
        # segmentation answers the lock rather than a complaint about the body.
        #
        # This is the one that shipped open. A segmentation marked done still
        # accepted a new ROI: the row was created, associated with the finished
        # segmentation and made active, so the labeling screen for a locked
        # image gained a ROI window it then refused every operation on -- the
        # per-organelle "mark done" below is locked, and so is `rerun-roi`. A
        # to-do that can never be finished, written past a write lock.
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        target_image = get_segmentation_target_image(segmentation)

        source = request.data.get("source", "AUTO") or "AUTO"
        seed = request.data.get("seed")
        if source == "DEFAULT":
            source = "AUTO"

        coords = {}
        for key in ("x", "y", "width", "height"):
            if key in request.data:
                coords[key] = int(request.data[key])

        if coords and len(coords) != 4:
            return Response(
                {"error": "x, y, width, and height are required together"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if coords:
            roi_x = coords["x"]
            roi_y = coords["y"]
            roi_w = coords["width"]
            roi_h = coords["height"]
        else:
            if source == "MANUAL":
                return Response(
                    {"error": "Manual ROI creation requires coordinates"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            roi_min_size = int(os.environ.get("ROI_MIN_IMAGE_SIZE", "512"))
            roi_size = int(os.environ.get("ROI_SIZE", "1024"))
            if target_image.width >= roi_min_size and target_image.height >= roi_min_size:
                roi_result = select_roi_for_image(target_image, roi_size=roi_size, seed=seed)
                roi_x, roi_y, roi_w, roi_h = (
                    roi_result.x,
                    roi_result.y,
                    roi_result.width,
                    roi_result.height,
                )
            else:
                roi_x, roi_y, roi_w, roi_h = (
                    0,
                    0,
                    target_image.width,
                    target_image.height,
                )

        roi_image = create_roi_image_from_image(
            target_image,
            x=roi_x,
            y=roi_y,
            width=roi_w,
            height=roi_h,
            source=source,
            is_active=True,
            segmentation=segmentation,
        )
        if "prepopulate" in request.data:
            from quantem.segmentation.roi_initialization import initialize_roi

            initialize_roi(segmentation, roi_image, prepopulate=request.data["prepopulate"] is True)
        serializer = SegmentationRoiSerializer(
            roi_image,
            context={"segmentation": segmentation},
        )
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class SegmentationRoiCompleteView(APIView):
    """Mark the active ROI as complete."""

    @transaction.atomic
    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        # This compatibility endpoint uses the same per-organelle state as the
        # explicit ROI completion endpoint.
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        get_segmentation_target_image(segmentation)
        roi_image = get_active_roi_for_segmentation(segmentation)
        if not roi_image:
            return Response(
                {"error": "No active ROI found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        with transaction.atomic():
            ImageSegmentation.objects.select_for_update().get(pk=segmentation.pk)
            roi_image = get_object_or_404(
                _roi_queryset_for_segmentation(segmentation), pk=roi_image.pk
            )
            status_row, _ = RoiSegmentationStatus.objects.get_or_create(
                image_roi=roi_image, segmentation=segmentation
            )
            status_row.set_complete(True)
            status_row.save()
            from quantem.segmentation.mutations import record_change

            record_change(segmentation)
            candidate_cleanup = _delete_candidates_for_completed_roi(
                segmentation,
                roi_image,
            )
        overlay = _register_candidate_cleanup_overlay(segmentation, candidate_cleanup)
        removed = delete_probability_maps_for_roi(segmentation, roi_image)
        logging.getLogger(__name__).info(
            "Reclaimed %d probability map(s) for completed active ROI %s on segmentation %s.",
            removed,
            roi_image.id,
            segmentation.id,
        )
        serializer = SegmentationRoiSerializer(
            roi_image,
            context={"segmentation": segmentation},
        )
        return Response(
            {
                **serializer.data,
                "candidate_cleanup": candidate_cleanup.as_payload(),
                "overlay": overlay,
            },
            status=status.HTTP_200_OK,
        )


class SegmentationRoiSegmentationCompleteView(APIView):
    """Mark a specific ROI complete/incomplete for THIS segmentation (organelle).

    This is the per-organelle "mark ROI as done" — finer-grained than the flat
    ``ImageROI.is_complete`` flag and the per-image ``SegmentationCompleteView``.
    ``POST`` marks the ROI complete for the segmentation; ``DELETE`` reverts it.
    """

    @transaction.atomic
    def post(self, request, seg_id, roi_id):
        return self._set_complete(seg_id, roi_id, is_complete=True)

    @transaction.atomic
    def delete(self, request, seg_id, roi_id):
        return self._set_complete(seg_id, roi_id, is_complete=False)

    def _set_complete(self, seg_id, roi_id, *, is_complete: bool):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        # Per-(ROI, segmentation) proofreading state, so it follows this
        # segmentation's lock. See the module docstring for where that line
        # falls on the shared rectangle itself.
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        roi_image = get_object_or_404(
            _roi_queryset_for_segmentation(segmentation),
            id=roi_id,
        )
        candidate_cleanup = None
        overlay = None
        with transaction.atomic():
            ImageSegmentation.objects.select_for_update().get(pk=segmentation.pk)
            roi_image = get_object_or_404(
                _roi_queryset_for_segmentation(segmentation), pk=roi_image.pk
            )
            status_row, _ = RoiSegmentationStatus.objects.get_or_create(
                image_roi=roi_image,
                segmentation=segmentation,
            )
            status_row.set_complete(is_complete)
            from quantem.segmentation.mutations import record_change

            record_change(segmentation)
            status_row.save(update_fields=["is_complete", "completed_at", "updated_at"])
            if is_complete:
                candidate_cleanup = _delete_candidates_for_completed_roi(
                    segmentation,
                    roi_image,
                )
        if candidate_cleanup is not None:
            overlay = _register_candidate_cleanup_overlay(segmentation, candidate_cleanup)
        if is_complete:
            removed = delete_probability_maps_for_roi(segmentation, roi_image)
            logging.getLogger(__name__).info(
                "Reclaimed %d probability map(s) for completed ROI %s on segmentation %s.",
                removed,
                roi_image.id,
                segmentation.id,
            )
        serializer = SegmentationRoiSerializer(
            roi_image,
            context={"segmentation": segmentation},
        )
        return Response(
            {
                **serializer.data,
                "candidate_cleanup": (
                    candidate_cleanup.as_payload()
                    if candidate_cleanup is not None
                    else {"deleted": 0, "updated": 0, "created": 0}
                ),
                "overlay": overlay,
            },
            status=status.HTTP_200_OK,
        )


class SegmentationRoiDetailView(APIView):
    """Remove only this segmentation's ROI membership and completion record.

    Migrated crops can still be referenced by other organelles. Keep those
    crops and their review records intact. Annotation objects are never removed.
    """

    @transaction.atomic
    def delete(self, request, seg_id, roi_id):
        segmentation = get_object_or_404(ImageSegmentation.objects.select_for_update(), id=seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        roi_image = get_object_or_404(
            _roi_queryset_for_segmentation(segmentation).select_for_update(),
            id=roi_id,
        )
        from quantem.segmentation.mutations import record_change

        active = get_active_roi_for_segmentation(segmentation)
        was_active = active is not None and active.pk == roi_image.pk
        record_change(segmentation)
        RoiSegmentationStatus.objects.filter(
            image_roi=roi_image, segmentation=segmentation
        ).delete()
        roi_image.segmentations.remove(segmentation)
        ImageSegmentation.objects.filter(pk=segmentation.pk).update(rois_initialized=True)
        if not roi_image.segmentations.exists() and not roi_image.segmentation_statuses.exists():
            crop_path = ROIS_DIR / f"{roi_image.id}.png"
            roi_image.delete()

            def remove_crop():
                try:
                    crop_path.unlink(missing_ok=True)
                except OSError:
                    logging.getLogger(__name__).warning("Failed to delete ROI crop %s", crop_path)

            transaction.on_commit(remove_crop)

        if was_active:
            next_roi = (
                _roi_queryset_for_segmentation(segmentation).order_by("-created_at", "-id").first()
            )
            if next_roi is not None:
                activate_roi(next_roi, segmentation)

        return Response(status=status.HTTP_204_NO_CONTENT)


class SegmentationRoiActivateView(APIView):
    """Select an ROI in this organelle's workspace, including when browsing completed work."""

    @transaction.atomic
    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation.objects.select_for_update(), id=seg_id)
        get_segmentation_target_image(segmentation)
        serializer = SegmentationRoiActivateSerializer(data=request.data or {})
        serializer.is_valid(raise_exception=True)

        roi_image = get_object_or_404(
            _roi_queryset_for_segmentation(segmentation),
            id=serializer.validated_data["roi_id"],
        )
        activate_roi(roi_image, segmentation)
        response_serializer = SegmentationRoiSerializer(
            roi_image,
            context={"segmentation": segmentation},
        )
        return Response(response_serializer.data, status=status.HTTP_200_OK)


class SegmentationRoiSegmentsView(APIView):
    """Return segments within the active ROI bounding box."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        get_segmentation_target_image(segmentation)
        roi_image = get_active_roi_for_segmentation(segmentation)
        if not roi_image:
            return Response([], status=status.HTTP_200_OK)

        roi_geom = make_bbox(
            roi_image.x,
            roi_image.y,
            roi_image.x + roi_image.width,
            roi_image.y + roi_image.height,
        )
        qs = SegmentObject.objects.filter(segmentation=segmentation).filter(
            bbox_intersects_filter(roi_geom)
        )
        source_filter = source_model_queryset_filter(
            normalize_source_model(request.query_params.get("source_model"))
        )
        if source_filter is not None:
            qs = qs.filter(source_filter)
        serializer = SegmentObjectSerializer(qs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


def _roi_queryset_for_segmentation(segmentation: ImageSegmentation):
    return rois_for_segmentation(segmentation)
