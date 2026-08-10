"""Read/query segment API views."""

from __future__ import annotations

from quantem.segmentation.confidence import segment_confidence_score

from .shared import (
    GEOMETRY_DETAIL_FULL,
    Abs,
    APIView,
    F,
    ImageSegmentation,
    Polygon,
    Response,
    SegmentObject,
    SegmentObjectSerializer,
    SegmentQueryRegionSerializer,
    _apply_segment_source_filter,
    _geometry_coords_from_polygon,
    _parse_label_states_param,
    _parse_segment_statuses_param,
    _parse_source_model_param,
    bbox_contains_point_filter,
    bbox_intersects_filter,
    get_object_or_404,
    make_bbox,
    make_point,
    normalize_geometry_detail,
    select_non_overlapping_inferred_segments,
    status,
)

_QUERY_REGION_ONLY_FIELDS = (
    "id",
    "status",
    "label_state",
    "source_model",
    "confidence_score",
    "features",
    "geometry_wkb",
)


class SegmentationUncertainSegmentsView(APIView):
    """Return the most uncertain segments."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        limit = int(request.query_params.get("limit", 50))
        source_model = _parse_source_model_param(request.query_params.get("source_model"))
        qs = _apply_segment_source_filter(
            SegmentObject.objects.filter(
                segmentation=segmentation,
                label_state__in=["INFERRED", "CANDIDATE"],
                confidence_score__isnull=False,
            ),
            source_model,
        )
        qs = (
            qs
            .annotate(uncertainty=Abs(F("confidence_score") - 0.5))
            .order_by("uncertainty")[:limit]
        )
        serializer = SegmentObjectSerializer(qs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class SegmentsAtPointView(APIView):
    """Find segments at a specific point."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        try:
            x_coord = float(request.query_params.get("x"))
            y_coord = float(request.query_params.get("y"))
        except (ValueError, TypeError):
            return Response(
                {"error": "Missing or invalid point parameters: x, y"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        point = make_point(x_coord, y_coord)
        geometry_detail = normalize_geometry_detail(
            request.query_params.get("geometry_detail", GEOMETRY_DETAIL_FULL)
        )
        source_model = _parse_source_model_param(request.query_params.get("source_model"))
        # bbox prefilter on the indexed columns, then an exact shapely
        # ``contains`` refine.
        qs = SegmentObject.objects.filter(segmentation=segmentation).filter(
            bbox_contains_point_filter(x_coord, y_coord)
        )
        states_list = _parse_label_states_param(request.query_params.get("states"))
        status_list = _parse_segment_statuses_param(request.query_params.get("statuses"))
        if status_list:
            qs = qs.filter(status__in=status_list)
        if states_list:
            qs = qs.filter(label_state__in=states_list)
        qs = _apply_segment_source_filter(qs, source_model)

        segments = [
            segment
            for segment in qs[:2000]
            if (geometry := segment.geometry) is not None and geometry.contains(point)
        ]

        # Rank the shapes under the cursor. A scored object wins over an
        # unscored one, and objects with no score are ordered
        # smallest-area-first -- the tightest shape around the click.
        #
        # The previous fallback scored "no score" as 0.0, which made every
        # hand-drawn and every unscored object an exact tie and left the order
        # to whatever the database happened to return. A missing score is not a
        # low score, and a click has to resolve to the same object twice
        # running. It also read only ``sam_score``, so an object whose
        # confidence lives in ``features["mean_prob"]`` sorted as unscored while
        # the same endpoint's serializer reported its 0.82.
        def _rank_key(segment) -> tuple[bool, float, float]:
            score = segment_confidence_score(segment)
            geometry = segment.geometry
            area = float(geometry.area) if geometry is not None else float("inf")
            return (score is not None, score if score is not None else 0.0, -area)

        selected_segments = sorted(segments, key=_rank_key, reverse=True)[:20]

        serializer = SegmentObjectSerializer(
            selected_segments,
            many=True,
            context={"geometry_detail": geometry_detail},
        )
        return Response(serializer.data, status=status.HTTP_200_OK)


class SegmentsQueryRegionView(APIView):
    """Find exact segment hits within a bbox or polygon region."""

    def post(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        serializer = SegmentQueryRegionSerializer(data=request.data or {})
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        payload = serializer.validated_data
        source_model = _parse_source_model_param(payload.get("source_model"))
        if "bbox" in payload:
            bbox = payload["bbox"]
            region_geometry = make_bbox(
                bbox["x0"],
                bbox["y0"],
                bbox["x1"],
                bbox["y1"],
            )
        else:
            polygon_coords = payload["polygon_coords"]
            coords = [(float(x), float(y)) for x, y in polygon_coords]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                region_geometry = Polygon(coords)
            except Exception:
                return Response(
                    {"error": "Invalid polygon_coords geometry"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if not region_geometry.is_valid:
                try:
                    region_geometry = region_geometry.buffer(0)
                except Exception:
                    return Response(
                        {"error": "Invalid polygon_coords geometry"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

        qs = SegmentObject.objects.filter(segmentation=segmentation).filter(
            bbox_intersects_filter(region_geometry)
        )
        states_list = payload.get("states") or []
        status_list = payload.get("statuses") or []
        if status_list:
            qs = qs.filter(status__in=[int(value) for value in status_list])
        if states_list:
            qs = qs.filter(label_state__in=states_list)
        qs = _apply_segment_source_filter(qs, source_model)

        include_geometry = bool(payload.get("include_geometry"))
        segments_payload = []
        for segment in qs.only(*_QUERY_REGION_ONLY_FIELDS).iterator():
            geometry = segment.geometry
            if geometry is None or not geometry.intersects(region_geometry):
                continue
            item = {
                "id": str(segment.id),
                "status": int(segment.status),
                "status_label": segment.status_label,
                "source_model": segment.source_model,
                "label_state": segment.label_state,
                # The same answer /segments/at-point gives for this row. It used
                # to fall back to sam_score alone and so reported null for an
                # object whose confidence is in features["mean_prob"] -- the
                # measurement the column itself is filled from.
                #
                # Still null, never 0.0, when there is no score of any kind: a
                # hand-drawn outline a human confirmed has no confidence, and
                # "0.0" reads as the model's lowest possible certainty about it.
                "confidence_score": segment_confidence_score(segment),
            }
            if include_geometry:
                item["geometry_coords"] = _geometry_coords_from_polygon(geometry)
            segments_payload.append(item)

        return Response({"segments": segments_payload}, status=status.HTTP_200_OK)


class InferredSegmentsView(APIView):
    """Return confirmed and non-overlapping inferred segments."""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(ImageSegmentation, id=seg_id)
        source_model = _parse_source_model_param(request.query_params.get("source_model"))
        try:
            threshold = float(request.query_params.get("threshold", 0.99))
        except (ValueError, TypeError):
            threshold = 0.99

        if not 0.0 <= threshold <= 1.0:
            return Response(
                {"error": "threshold must be between 0.0 and 1.0"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        viewport_geom = None
        viewport_params = ["x_min", "y_min", "x_max", "y_max"]
        if all(param in request.query_params for param in viewport_params):
            try:
                x_min = float(request.query_params.get("x_min"))
                y_min = float(request.query_params.get("y_min"))
                x_max = float(request.query_params.get("x_max"))
                y_max = float(request.query_params.get("y_max"))
                if x_min < x_max and y_min < y_max:
                    viewport_geom = make_bbox(x_min, y_min, x_max, y_max)
            except (ValueError, TypeError):
                pass

        confirmed_qs = SegmentObject.objects.filter(
            segmentation=segmentation,
            label_state="CONFIRMED",
        )
        if viewport_geom is not None:
            confirmed_qs = confirmed_qs.filter(bbox_intersects_filter(viewport_geom))

        inferred_segments = select_non_overlapping_inferred_segments(
            segmentation,
            threshold,
            viewport_geom,
            max_candidates=5000,
            source_model=source_model,
        )

        confirmed_serializer = SegmentObjectSerializer(confirmed_qs, many=True)
        inferred_serializer = SegmentObjectSerializer(inferred_segments, many=True)
        return Response(
            {
                "confirmed": confirmed_serializer.data,
                "inferred": inferred_serializer.data,
            },
            status=status.HTTP_200_OK,
        )
