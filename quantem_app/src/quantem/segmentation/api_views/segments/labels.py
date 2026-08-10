"""Label-oriented segment API views."""

from __future__ import annotations

from quantem.segmentation.segment_status import status_for_segment_lifecycle
from quantem.segmentation.source_models import normalize_source_model

from .shared import (
    APIView,
    ImageSegmentation,
    Response,
    SegmentObject,
    SegmentObjectLabelUpdateSerializer,
    SegmentObjectSerializer,
    _enqueue_segment_feature_refresh,
    _invalidate_tiles_for_segmentation,
    _run_with_sqlite_lock_retry,
    completion_lock_response,
    full_image_dirty_bbox,
    get_object_or_404,
    logger,
    merge_dirty_bboxes,
    register_overlay_mutation,
    register_overlay_mutation_all_bundles,
    status,
)

_BBOX_ONLY_FIELDS = (
    "id",
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
)


class SegmentLabelUpdateView(APIView):
    """Update a segment's label state."""

    def post(self, request, segment_id):
        segment = get_object_or_404(
            SegmentObject.objects.select_related("segmentation__asset"),
            id=segment_id,
        )
        locked = completion_lock_response(segment.segmentation)
        if locked is not None:
            return locked
        serializer = SegmentObjectLabelUpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        new_label_state = serializer.validated_data["label_state"]
        old_label_state = segment.label_state
        label_changed = new_label_state != old_label_state

        segment.label_state = new_label_state
        segment.refined = "UNREFINED"
        segment.status = status_for_segment_lifecycle(
            label_state=segment.label_state,
            refined=segment.refined,
        )
        _run_with_sqlite_lock_retry(lambda: segment.save())
        _invalidate_tiles_for_segmentation(str(segment.segmentation_id))
        # A flip to/from CONFIRMED changes all-bundle membership, so the edit must
        # fan out to every bundle; otherwise it is scoped to the one source bundle.
        confirmed_membership_changed = (
            new_label_state == "CONFIRMED" or old_label_state == "CONFIRMED"
        )
        overlay_mutation = (
            register_overlay_mutation_all_bundles
            if confirmed_membership_changed
            else register_overlay_mutation
        )
        overlay = overlay_mutation(
            segment.segmentation,
            dirty_bbox=merge_dirty_bboxes(segment.segmentation, [segment.geometry]),
            source_model=normalize_source_model(request.data.get("source_model")),
        )

        if label_changed and (
            new_label_state in ["CONFIRMED", "EXCLUDED"]
            or old_label_state in ["CONFIRMED", "EXCLUDED"]
        ):
            try:
                _enqueue_segment_feature_refresh(
                    segmentation_id=str(segment.segmentation_id),
                    segment_ids=[],
                    recompute_features=True,
                )
                logger.info(
                    "Updated label for segment %s from %s to %s; queued feature refresh",
                    segment_id,
                    old_label_state,
                    new_label_state,
                )
            except Exception as exc:
                logger.error(
                    "Error queueing feature refresh for segment %s: %s",
                    segment_id,
                    exc,
                    exc_info=True,
                )

        segment.refresh_from_db()
        serializer = SegmentObjectSerializer(segment)
        response_payload = serializer.data
        response_payload["overlay"] = overlay
        return Response(response_payload, status=status.HTTP_200_OK)


class SegmentBatchLabelUpdateView(APIView):
    """Batch update segment labels."""

    def post(self, request):
        labels = request.data.get("labels", [])
        if not isinstance(labels, list) or not labels:
            return Response(
                {"error": "labels must be a non-empty list"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        segment_ids = [item.get("id") for item in labels if item.get("id")]
        segments = SegmentObject.objects.select_related("segmentation__asset").filter(
            id__in=segment_ids
        )
        if segments.count() == 0:
            return Response(
                {"error": "No segments found for provided IDs"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # A group action can span segmentations. Refusing the whole batch when
        # any one of them is locked is the only outcome that does not leave a
        # partially applied edit the caller was never told about.
        locked = completion_lock_response(
            *{
                str(segment.segmentation_id): segment.segmentation
                for segment in segments
            }.values()
        )
        if locked is not None:
            return locked

        valid_states = {choice[0] for choice in SegmentObject.LABEL_STATE_CHOICES}
        segments_by_id = {str(segment.id): segment for segment in segments}
        segments_to_update: list[SegmentObject] = []
        touched_segmentations: set[str] = set()
        geometries_by_segmentation: dict[str, list] = {}
        confirmed_membership_changed_by_segmentation: dict[str, bool] = {}
        active_source_model = normalize_source_model(request.data.get("source_model"))

        for item in labels:
            raw_id = item.get("id")
            if not raw_id:
                continue
            segment = segments_by_id.get(str(raw_id))
            if segment is None:
                continue

            new_label = item.get("label_state")
            if not new_label or new_label not in valid_states:
                continue

            if segment.label_state == new_label:
                continue

            old_label = segment.label_state
            segment.label_state = new_label
            segment.refined = "UNREFINED"
            segment.status = status_for_segment_lifecycle(
                label_state=segment.label_state,
                refined=segment.refined,
            )
            segments_to_update.append(segment)
            segmentation_key = str(segment.segmentation_id)
            touched_segmentations.add(segmentation_key)
            geometries_by_segmentation.setdefault(segmentation_key, []).append(
                segment.geometry
            )
            if old_label == "CONFIRMED" or new_label == "CONFIRMED":
                confirmed_membership_changed_by_segmentation[segmentation_key] = True

        updated_count = 0
        updated_ids: list[str] = []
        overlays: dict[str, dict] = {}

        if segments_to_update:
            _run_with_sqlite_lock_retry(
                lambda: SegmentObject.objects.bulk_update(
                    segments_to_update,
                    ["label_state", "refined", "status"],
                )
            )
            updated_count += len(segments_to_update)
            updated_ids.extend(str(segment.id) for segment in segments_to_update)
            segmentations_by_id = {
                str(segment.segmentation_id): segment.segmentation
                for segment in segments_to_update
            }
            for seg_id in touched_segmentations:
                _invalidate_tiles_for_segmentation(seg_id)
                try:
                    _enqueue_segment_feature_refresh(
                        segmentation_id=seg_id,
                        segment_ids=[],
                        recompute_features=True,
                    )
                except Exception as exc:
                    # The batch labels are already written, so the request still
                    # succeeded; what is lost is the refresh. Say so, as the
                    # single-segment path and both mutation paths do -- a bare
                    # `continue` left no trace at all, so a queue that was
                    # rejecting work looked exactly like one with nothing to do.
                    logger.warning(
                        "Failed to queue feature refresh for segmentation %s "
                        "after a batch label update: %s",
                        seg_id,
                        exc,
                        exc_info=True,
                    )
                    continue
            for seg_id, segmentation in segmentations_by_id.items():
                # A flip to/from CONFIRMED changes all-bundle membership, so the
                # edit must fan out to every bundle; otherwise it is scoped to the
                # one source bundle.
                overlay_mutation = (
                    register_overlay_mutation_all_bundles
                    if confirmed_membership_changed_by_segmentation.get(seg_id, False)
                    else register_overlay_mutation
                )
                overlays[seg_id] = overlay_mutation(
                    segmentation,
                    dirty_bbox=merge_dirty_bboxes(
                        segmentation,
                        geometries_by_segmentation.get(seg_id, []),
                    ),
                    source_model=active_source_model,
                )

        return Response(
            {
                "created": 0,
                "updated": updated_count,
                "deleted": 0,
                "created_ids": [],
                "updated_ids": updated_ids,
                "deleted_ids": [],
                "overlays": overlays,
            },
            status=status.HTTP_200_OK,
        )


class SegmentationClearManualLabelsView(APIView):
    """Delete confirmed/excluded segments for a segmentation."""

    def post(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset"),
            id=seg_id,
        )
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        manual_qs = SegmentObject.objects.filter(
            segmentation=segmentation,
            label_state__in=["CONFIRMED", "EXCLUDED"],
        )
        manual_bboxes = [segment.bbox for segment in manual_qs.only(*_BBOX_ONLY_FIELDS)]
        deleted, _ = manual_qs.delete()
        if deleted > 0:
            _invalidate_tiles_for_segmentation(str(segmentation.id))
            # Deleting confirmed/excluded objects removes them from every bundle.
            overlay = register_overlay_mutation_all_bundles(
                segmentation,
                dirty_bbox=merge_dirty_bboxes(segmentation, manual_bboxes)
                or full_image_dirty_bbox(segmentation),
            )
        else:
            overlay = None
        return Response({"deleted": deleted, "overlay": overlay}, status=status.HTTP_200_OK)
