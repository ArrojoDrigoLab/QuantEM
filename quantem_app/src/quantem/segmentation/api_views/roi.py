"""ROI endpoints for segmentation workflows."""

from __future__ import annotations

import logging
import os

from django.shortcuts import get_object_or_404
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.models import ImageROI
from quantem.assets.roi_state import activate_roi, get_active_roi_for_asset
from quantem.assets.utils import create_roi_image_from_image
from quantem.core.config import ROIS_DIR
from quantem.segmentation.models import ImageSegmentation, RoiSegmentationStatus, SegmentObject
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
from quantem.segmentation.services.spatial_lookup import bbox_intersects_filter, make_bbox
from quantem.segmentation.source_models import normalize_source_model, source_model_queryset_filter

from .shared import (
    completion_lock_response,
    get_or_create_roi_image,
    get_segmentation_target_image,
)


class SegmentationRoiActivateSerializer(serializers.Serializer):
    roi_id = serializers.UUIDField()


class CompletedRoiListCreateView(APIView):
    """List or create user-marked completed ROIs for a segmentation."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        serializer = CompletedRoiSerializer(
            list_completed_rois(segmentation),
            many=True,
        )
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        # A completed area is proofreading state that belongs to this
        # segmentation, unlike the ROI rectangles below, which are shared by
        # every organelle on the asset.
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
                "-is_active",
                "-created_at",
            )
        )
        if not rois:
            roi_image = get_or_create_roi_image(target_image)
            roi_image.segmentations.add(segmentation)
            rois = [roi_image]

        serializer = SegmentationRoiSerializer(
            rois,
            many=True,
            context={"segmentation": segmentation},
        )
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
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

            roi_min_size = int(os.environ.get("ROI_MIN_IMAGE_SIZE", "6000"))
            roi_size = int(os.environ.get("ROI_SIZE", "3000"))
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
        )
        roi_image.segmentations.add(segmentation)
        serializer = SegmentationRoiSerializer(
            roi_image,
            context={"segmentation": segmentation},
        )
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class SegmentationRoiCompleteView(APIView):
    """Mark the active ROI as complete."""

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        get_segmentation_target_image(segmentation)
        roi_image = get_active_roi_for_asset(segmentation.asset)
        if not roi_image:
            return Response(
                {"error": "No active ROI found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        roi_image.is_complete = True
        roi_image.save(update_fields=["is_complete", "updated_at"])
        serializer = SegmentationRoiSerializer(
            roi_image,
            context={"segmentation": segmentation},
        )
        return Response(serializer.data, status=status.HTTP_200_OK)


class SegmentationRoiSegmentationCompleteView(APIView):
    """Mark a specific ROI complete/incomplete for THIS segmentation (organelle).

    This is the per-organelle "mark ROI as done" — finer-grained than the flat
    ``ImageROI.is_complete`` flag and the per-image ``SegmentationCompleteView``.
    ``POST`` marks the ROI complete for the segmentation; ``DELETE`` reverts it.
    """

    def post(self, request, seg_id, roi_id):
        return self._set_complete(seg_id, roi_id, is_complete=True)

    def delete(self, request, seg_id, roi_id):
        return self._set_complete(seg_id, roi_id, is_complete=False)

    def _set_complete(self, seg_id, roi_id, *, is_complete: bool):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        # Per-(ROI, segmentation) proofreading state, so it follows this
        # segmentation's lock. The ROI rectangle itself does not: it is shared
        # by every organelle on the asset and locking one must not freeze them.
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        roi_image = get_object_or_404(
            _roi_queryset_for_segmentation(segmentation),
            id=roi_id,
        )
        # Keep the ROI <-> segmentation association in sync with the status row.
        roi_image.segmentations.add(segmentation)
        status_row, _ = RoiSegmentationStatus.objects.get_or_create(
            image_roi=roi_image,
            segmentation=segmentation,
        )
        status_row.set_complete(is_complete)
        status_row.save(update_fields=["is_complete", "completed_at", "updated_at"])
        serializer = SegmentationRoiSerializer(
            roi_image,
            context={"segmentation": segmentation},
        )
        return Response(serializer.data, status=status.HTTP_200_OK)


class SegmentationRoiDetailView(APIView):
    """Delete a specific ROI for a segmentation's asset.

    ROIs are asset-scoped (shared across the asset's segmentations/organelles),
    so deleting one removes it for every segmentation on the asset. The cascade
    drops the per-(ROI, segmentation) completion rows (``RoiSegmentationStatus``)
    and the ROI<->segmentation associations; segment objects labeled inside the
    ROI window are left untouched. If the deleted ROI was the active one, the
    most recently created remaining ROI for the asset is activated to preserve
    the "one active ROI" invariant.
    """

    def delete(self, request, seg_id, roi_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        roi_image = get_object_or_404(
            _roi_queryset_for_segmentation(segmentation),
            id=roi_id,
        )
        asset = roi_image.asset
        was_active = roi_image.is_active

        # Best-effort removal of the on-disk ROI PNG crop (named by ROI id).
        try:
            (ROIS_DIR / f"{roi_image.id}.png").unlink(missing_ok=True)
        except OSError:
            logging.getLogger(__name__).warning(
                "Failed to delete ROI crop file for %s", roi_image.id, exc_info=True
            )

        roi_image.delete()

        if was_active and asset is not None:
            next_roi = (
                ImageROI.objects.filter(asset=asset).order_by("-created_at").first()
            )
            if next_roi is not None:
                activate_roi(next_roi)

        return Response(status=status.HTTP_204_NO_CONTENT)


class SegmentationRoiActivateView(APIView):
    """Activate an existing ROI for the segmentation image."""

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        get_segmentation_target_image(segmentation)
        serializer = SegmentationRoiActivateSerializer(data=request.data or {})
        serializer.is_valid(raise_exception=True)

        roi_image = get_object_or_404(
            _roi_queryset_for_segmentation(segmentation),
            id=serializer.validated_data["roi_id"],
        )
        roi_image.segmentations.add(segmentation)
        activate_roi(roi_image)
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
        roi_image = get_active_roi_for_asset(segmentation.asset)
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
    return ImageROI.objects.filter(asset=segmentation.asset)
