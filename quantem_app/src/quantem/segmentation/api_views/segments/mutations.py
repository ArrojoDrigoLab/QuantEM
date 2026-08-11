"""Mutation-oriented segment API views."""

from __future__ import annotations

from quantem.segmentation.features.measure import measure_segments

from .shared import (
    _MERGE_ELIGIBLE_STATES,
    APIView,
    BaseGeometry,
    ImageSegmentation,
    MeasurementOutcome,
    Polygon,
    Response,
    SegmentObject,
    SegmentObjectSerializer,
    _enqueue_segment_feature_refresh,
    _extract_polygons,
    _filter_supported_polygons,
    _geometries_overlap,
    _invalidate_tiles_for_segmentation,
    _parse_optional_sam_score,
    bbox_intersects_filter,
    completion_lock_response,
    confirm_segment_geometries,
    filter_supported_confirmed_polygons,
    full_image_dirty_bbox,
    get_object_or_404,
    has_narrow_bbox,
    logger,
    measurement_response_status,
    merge_dirty_bboxes,
    outline_geometry,
    parse_drawn_outline,
    parse_outline_pieces,
    register_confirmation_overlay_mutation,
    register_overlay_mutation,
    register_overlay_mutation_all_bundles,
    segmentation_image_size,
    separated_outlines_payload,
    status,
    transaction,
)

_BBOX_ONLY_FIELDS = (
    "id",
    "bbox_minx",
    "bbox_miny",
    "bbox_maxx",
    "bbox_maxy",
)


class SegmentCreateView(APIView):
    """Create a new segment from geometry coordinates."""

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked

        polygon, geometry_error = parse_drawn_outline(
            request.data.get("geometry_coords"),
            image_size=segmentation_image_size(segmentation),
        )
        if polygon is None:
            return Response(
                {"error": geometry_error},
                status=status.HTTP_400_BAD_REQUEST,
            )
        bbox = polygon.envelope

        label_state = request.data.get("label_state", "CONFIRMED")
        valid_states = {choice[0] for choice in SegmentObject.LABEL_STATE_CHOICES}
        if label_state not in valid_states:
            return Response(
                {"error": f"Invalid label_state: {label_state}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if label_state in _MERGE_ELIGIBLE_STATES and has_narrow_bbox(bbox):
            return Response(
                {
                    "error": (
                        "geometry bbox must span more than 1 pixel in both "
                        "dimensions for candidate or confirmed segments"
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        segment = SegmentObject.objects.create(
            segmentation=segmentation,
            label_state=label_state,
            source_model="manual",
            confidence_score=None,
            # No sam_score: this route is the hand-drawn one, and a fabricated
            # 1.0 was the only thing such an object used to carry. Objects the
            # box tool creates do get a real score, but they come through
            # confirm_segment_geometries, not here.
            features={},
            geometry=polygon,
            centroid=polygon.centroid,
            bbox=polygon.envelope,
        )
        # Measure now, not via the queued refresh: that job's trigger is off by
        # default, so a drawn object reached objects.csv with every morphometric
        # column blank while `calibrated` said True next to it.
        measurement = measure_segments(segmentation, [segment])
        _invalidate_tiles_for_segmentation(str(segmentation.id))
        # A CONFIRMED object is a member of every bundle; a candidate is scoped
        # to its one source bundle.
        overlay_mutation = (
            register_overlay_mutation_all_bundles
            if label_state == "CONFIRMED"
            else register_overlay_mutation
        )
        overlay = overlay_mutation(
            segmentation,
            dirty_bbox=merge_dirty_bboxes(segmentation, [polygon]),
        )

        try:
            _enqueue_segment_feature_refresh(
                segmentation_id=str(segmentation.id),
                segment_ids=[str(segment.id)],
                recompute_features=label_state in ["CONFIRMED", "EXCLUDED"],
            )
        except Exception as exc:
            logger.error(
                "Error queueing segment feature refresh for %s: %s",
                segment.id,
                exc,
                exc_info=True,
            )

        serializer = SegmentObjectSerializer(segment)
        response_payload = serializer.data
        response_payload["overlay"] = overlay
        # 201 stands: the object exists and the response carries it. What the
        # block says is that its morphometric columns are empty, which a client
        # otherwise only discovers by reading `features` and finding nothing.
        response_payload["measurement"] = measurement.as_payload()
        return Response(response_payload, status=status.HTTP_201_CREATED)


class SegmentationConfirmBatchView(APIView):
    """Confirm one or more polygons, with optional merge or manual-create cleanup.

    This is the endpoint every drawing tool posts to (``useDrawing`` closes the
    raw freehand path and ``shared/api/segmentations/annotations.ts`` sends it
    here), so a stroke that crosses itself arrives here routinely. Each outline
    is stored as **every** area it encloses, not only its largest lobe, and the
    ``outlines`` block in the response names any outline that separated -- see
    ``parse_outline_pieces`` for what dropping the other lobes used to cost.

    The same block also names an outline that was **not stored at all**.
    ``filter_supported_confirmed_polygons`` refuses a polygon spanning a pixel
    or less in either dimension; a batch of one such outline used to come back
    ``200 {"created": 0}`` with no ``outlines`` block, i.e. indistinguishable
    from a success, while ``POST .../segments/`` refuses the identical shape
    with "geometry bbox must span more than 1 pixel in both dimensions". Two
    endpoints applying one rule must not disagree about whether it is worth
    mentioning. It stays a 2xx -- the rest of the batch is stored and the
    request was well-formed -- but the response now says what is missing.
    """

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        raw_segments = request.data.get("segments")

        if not isinstance(raw_segments, list) or len(raw_segments) == 0:
            return Response(
                {"error": "segments must be a non-empty list"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        def _bool_flag(name: str, default: bool = False) -> bool:
            raw_value = request.data.get(name, default)
            if isinstance(raw_value, str):
                return raw_value.strip().lower() not in {"", "0", "false", "no", "off"}
            return bool(raw_value)

        # Read before the loop because it decides what can be said about each
        # outline. Without merging, ``filter_supported_confirmed_polygons`` runs
        # on the outline's own pieces, so "kept" below is exactly what the
        # service will store. With merging, each outline is unioned with
        # whatever it overlaps *before* that filter, so a piece too thin to
        # stand alone may well survive inside a larger object and the same
        # arithmetic would be a guess. Only the certain half is reported.
        merge_overlaps = _bool_flag("merge_overlaps")

        incoming: list[dict[str, object]] = []
        # One entry per outline whose storage will not match the gesture: it
        # enclosed more than one area, or some of what it enclosed is too thin
        # to store, or both.
        outline_outcomes: list[dict[str, int]] = []
        for index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                return Response(
                    {"error": f"segments[{index}] must be an object"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            pieces = parse_outline_pieces(raw_segment.get("geometry_coords"))
            if not pieces:
                return Response(
                    {
                        "error": (
                            f"segments[{index}].geometry_coords must be a valid polygon "
                            "with at least 3 points"
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            # Only the counts are computed here; the pieces themselves go
            # through unfiltered so the service keeps sole ownership of which
            # polygons it is willing to store. Asking the same predicate the
            # service will apply is what makes "kept" a fact rather than a
            # guess -- and asking it for *every* outline, not only the ones
            # that separated, is what stopped a single sub-pixel outline being
            # dropped in silence.
            kept = (
                len(pieces)
                if merge_overlaps
                else len(filter_supported_confirmed_polygons(pieces))
            )
            if len(pieces) > 1 or kept < len(pieces):
                outline_outcomes.append(
                    {"index": index, "areas": len(pieces), "kept": kept}
                )

            sam_score_raw = raw_segment.get("sam_score")
            sam_score = _parse_optional_sam_score(sam_score_raw)
            if sam_score_raw is not None and sam_score is None:
                return Response(
                    {"error": f"segments[{index}].sam_score must be numeric when provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            incoming.append(
                {"geometry": outline_geometry(pieces), "sam_score": sam_score}
            )

        def _bool_flag(name: str, default: bool = False) -> bool:
            raw_value = request.data.get(name, default)
            if isinstance(raw_value, str):
                return raw_value.strip().lower() not in {"", "0", "false", "no", "off"}
            return bool(raw_value)

        result = confirm_segment_geometries(
            segmentation=segmentation,
            incoming=incoming,
            merge_overlaps=merge_overlaps,
            manual_creation=_bool_flag("manual_creation"),
            enqueue_feature_refresh=_bool_flag("enqueue_feature_refresh", True),
        )
        overlay = None
        if result.get("created", 0) or result.get("updated", 0) or result.get("deleted", 0):
            _invalidate_tiles_for_segmentation(str(segmentation.id))
            overlay = register_confirmation_overlay_mutation(
                segmentation=segmentation,
                result=result,
                fallback_geometries=[
                    item["geometry"] for item in incoming if "geometry" in item
                ],
            )
        result.pop("dirty_bbox", None)
        result["overlay"] = overlay
        measurement = result.pop("measurement", None) or MeasurementOutcome()
        result["measurement"] = measurement.as_payload()
        result["outlines"] = separated_outlines_payload(
            outline_outcomes, merged=merge_overlaps
        )

        return Response(result, status=measurement_response_status(measurement))


class SegmentationRemoveAreaView(APIView):
    """Subtract one or more area polygons from confirmed segments.

    The area to erase is the union of everything drawn, including every lobe of
    an outline that crossed itself. Keeping only the largest lobe -- what
    ``_parse_geometry_polygon`` used to do here -- meant an erase stroke rubbed
    out half of what it had been drawn round and still answered 200: measured, a
    figure-of-eight over a 5000 px region removed 2500 px, and the leftover
    stayed in ``objects.csv`` as part of the object.
    """

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        raw_areas = request.data.get("areas")
        if not isinstance(raw_areas, list) or len(raw_areas) == 0:
            return Response(
                {"error": "areas must be a non-empty list"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        area_polygons: list[Polygon] = []
        for index, raw_area in enumerate(raw_areas):
            if not isinstance(raw_area, dict):
                return Response(
                    {"error": f"areas[{index}] must be an object"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            pieces = parse_outline_pieces(raw_area.get("geometry_coords"))
            if not pieces:
                return Response(
                    {
                        "error": (
                            f"areas[{index}].geometry_coords must be a valid polygon "
                            "with at least 3 points"
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            area_polygons.extend(pieces)

        empty_response = {
            "created": 0,
            "updated": 0,
            "deleted": 0,
            "created_ids": [],
            "updated_ids": [],
            "deleted_ids": [],
            "measurement": None,
        }

        remove_geometry: BaseGeometry = area_polygons[0]
        for polygon in area_polygons[1:]:
            remove_geometry = remove_geometry.union(polygon)
        if remove_geometry.is_empty:
            return Response(empty_response, status=status.HTTP_200_OK)

        if not remove_geometry.is_valid:
            try:
                remove_geometry = remove_geometry.buffer(0)
            except Exception:
                return Response(
                    {"error": "failed to process remove area geometry"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if remove_geometry.is_empty:
                return Response(empty_response, status=status.HTTP_200_OK)

        candidate_segments = list(
            SegmentObject.objects.filter(
                segmentation=segmentation,
                label_state="CONFIRMED",
            ).filter(bbox_intersects_filter(remove_geometry))
        )

        created = 0
        updated = 0
        deleted = 0
        created_ids: list[str] = []
        updated_ids: list[str] = []
        deleted_ids: list[str] = []
        feature_refresh_ids: list[str] = []
        affected_geometries: list[BaseGeometry] = [remove_geometry]
        # Every object whose outline this edit changed. Their stored area,
        # perimeter and intensity describe the shape before the cut, so they are
        # re-measured once the transaction has committed the new geometry.
        remeasure: list[SegmentObject] = []

        with transaction.atomic():
            for segment in candidate_segments:
                segment_geometry = segment.geometry
                if segment_geometry is None:
                    continue
                if not _geometries_overlap(segment_geometry, remove_geometry):
                    continue
                affected_geometries.append(segment_geometry)

                try:
                    remaining_geometry = segment_geometry.difference(remove_geometry)
                except Exception:
                    logger.warning(
                        "Failed to subtract remove-area geometry for segment %s",
                        segment.id,
                        exc_info=True,
                    )
                    continue

                if remaining_geometry.is_empty:
                    segment_id = str(segment.id)
                    segment.delete()
                    deleted += 1
                    deleted_ids.append(segment_id)
                    continue

                if not remaining_geometry.is_valid:
                    try:
                        remaining_geometry = remaining_geometry.buffer(0)
                    except Exception:
                        logger.warning(
                            "Invalid remaining geometry after remove-area for segment %s",
                            segment.id,
                            exc_info=True,
                        )
                        continue

                remaining_polygons = _filter_supported_polygons(
                    _extract_polygons(remaining_geometry)
                )
                if not remaining_polygons:
                    segment_id = str(segment.id)
                    segment.delete()
                    deleted += 1
                    deleted_ids.append(segment_id)
                    continue

                remaining_polygons.sort(key=lambda poly: float(poly.area), reverse=True)
                primary_polygon = remaining_polygons[0]
                segment.geometry = primary_polygon
                segment.centroid = primary_polygon.centroid
                segment.bbox = primary_polygon.envelope
                segment.save(update_fields=["geometry", "centroid", "bbox"])
                affected_geometries.append(primary_polygon)
                remeasure.append(segment)
                updated += 1
                updated_ids.append(str(segment.id))
                feature_refresh_ids.append(str(segment.id))

                base_segment = segment.resolve_base_segment_or_self()
                base_features = (
                    dict(segment.features)
                    if isinstance(segment.features, dict)
                    else {}
                )
                for polygon in remaining_polygons[1:]:
                    created_segment = SegmentObject.objects.create(
                        segmentation=segmentation,
                        label_state=segment.label_state,
                        refined=segment.refined,
                        status=segment.status,
                        source_model=segment.source_model,
                        confidence_score=segment.confidence_score,
                        features=dict(base_features),
                        base_segment=base_segment,
                        geometry=polygon,
                        centroid=polygon.centroid,
                        bbox=polygon.envelope,
                    )
                    affected_geometries.append(polygon)
                    remeasure.append(created_segment)
                    created += 1
                    created_ids.append(str(created_segment.id))
                    feature_refresh_ids.append(str(created_segment.id))

        measurement = measure_segments(segmentation, remeasure, geometry_changed=True)

        try:
            _enqueue_segment_feature_refresh(
                segmentation_id=str(segmentation.id),
                segment_ids=feature_refresh_ids,
                recompute_features=(created > 0 or updated > 0 or deleted > 0),
            )
        except Exception as exc:
            logger.warning(
                "Failed to queue feature refresh after remove-area for %s: %s",
                segmentation.id,
                exc,
                exc_info=True,
            )

        overlay = None
        if created > 0 or updated > 0 or deleted > 0:
            _invalidate_tiles_for_segmentation(str(segmentation.id))
            # Edits confirmed objects, which belong to every bundle.
            overlay = register_overlay_mutation_all_bundles(
                segmentation,
                dirty_bbox=merge_dirty_bboxes(segmentation, affected_geometries),
            )

        return Response(
            {
                "created": created,
                "updated": updated,
                "deleted": deleted,
                "created_ids": created_ids,
                "updated_ids": updated_ids,
                "deleted_ids": deleted_ids,
                "overlay": overlay,
                "measurement": measurement.as_payload(),
            },
            status=measurement_response_status(measurement),
        )


class SegmentBatchDeleteView(APIView):
    """Hard-delete SegmentObjects by id.

    Used by the ER "reject group" action, where rejected candidates are removed
    entirely rather than marked EXCLUDED (ER objects are only CANDIDATE or
    CONFIRMED).
    """

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        locked = completion_lock_response(segmentation)
        if locked is not None:
            return locked
        ids = request.data.get("ids")
        if not isinstance(ids, list) or not ids:
            return Response(
                {"error": "ids must be a non-empty list"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        qs = SegmentObject.objects.filter(segmentation=segmentation, id__in=ids)
        bboxes = [
            bbox
            for segment in qs.only(*_BBOX_ONLY_FIELDS)
            if (bbox := segment.bbox) is not None
        ]
        with transaction.atomic():
            deleted, _ = qs.delete()

        overlay = None
        if deleted:
            _invalidate_tiles_for_segmentation(str(segmentation.id))
            # Hard-deletes objects (CANDIDATE or CONFIRMED); fan to every bundle
            # so a deleted confirmed object is removed everywhere.
            overlay = register_overlay_mutation_all_bundles(
                segmentation,
                dirty_bbox=(
                    merge_dirty_bboxes(segmentation, bboxes)
                    or full_image_dirty_bbox(segmentation)
                ),
            )

        return Response(
            {"deleted": deleted, "overlay": overlay},
            status=status.HTTP_200_OK,
        )
