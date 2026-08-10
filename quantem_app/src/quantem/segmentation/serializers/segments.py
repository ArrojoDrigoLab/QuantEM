"""Segment workflow serializers."""

from rest_framework import serializers

from quantem.segmentation.confidence import segment_confidence_score
from quantem.segmentation.geometry_serialization import (
    GEOMETRY_DETAIL_FULL,
    geometry_coords_from_polygon,
    normalize_geometry_detail,
)
from quantem.segmentation.models import SegmentObject
from quantem.segmentation.segment_status import (
    SEGMENT_STATUS_CANDIDATE,
    SEGMENT_STATUS_CONFIRMED,
    SEGMENT_STATUS_REFINED,
    segment_status_label,
)


class SegmentObjectSerializer(serializers.ModelSerializer):
    geometry_coords = serializers.SerializerMethodField()
    confidence_score = serializers.SerializerMethodField()
    status_label = serializers.SerializerMethodField()

    class Meta:
        model = SegmentObject
        fields = [
            "id",
            "segmentation",
            "status",
            "status_label",
            "source_model",
            "label_state",
            "refined",
            "confidence_score",
            # Per-object morphometrics (area, perimeter, circularity, intensity
            # statistics, ...) computed by
            # ``quantem.segmentation.features.extraction``. The analysis suite
            # reads them straight off this payload.
            "features",
            "geometry_coords",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_confidence_score(self, obj):
        # One rule for every endpoint that reports a confidence; see
        # quantem.segmentation.confidence for why the order is what it is.
        return segment_confidence_score(obj)

    def get_geometry_coords(self, obj):
        return geometry_coords_from_polygon(
            obj.geometry,
            geometry_detail=normalize_geometry_detail(
                self.context.get("geometry_detail", GEOMETRY_DETAIL_FULL)
            ),
        )

    def get_status_label(self, obj):
        return segment_status_label(obj.status)


class SegmentObjectLabelUpdateSerializer(serializers.Serializer):
    label_state = serializers.ChoiceField(choices=SegmentObject.LABEL_STATE_CHOICES, required=False)
    status = serializers.ChoiceField(
        choices=[
            SEGMENT_STATUS_CANDIDATE,
            SEGMENT_STATUS_CONFIRMED,
            SEGMENT_STATUS_REFINED,
        ],
        required=False,
    )

    def validate_label_state(self, value):
        if value not in [choice[0] for choice in SegmentObject.LABEL_STATE_CHOICES]:
            raise serializers.ValidationError(f"Invalid label_state: {value}")
        return value

    def validate(self, attrs):
        if ("label_state" in attrs) == ("status" in attrs):
            raise serializers.ValidationError("Provide exactly one of label_state or status.")
        if "status" in attrs:
            status_value = int(attrs["status"])
            if status_value == SEGMENT_STATUS_CONFIRMED:
                attrs["label_state"] = "CONFIRMED"
            else:
                attrs["label_state"] = "CANDIDATE"
        return attrs


class SegmentQueryRegionSerializer(serializers.Serializer):
    bbox = serializers.DictField(required=False)
    polygon_coords = serializers.ListField(
        child=serializers.ListField(child=serializers.FloatField(), min_length=2, max_length=2),
        required=False,
    )
    states = serializers.ListField(
        child=serializers.ChoiceField(choices=SegmentObject.LABEL_STATE_CHOICES),
        required=False,
        allow_empty=True,
    )
    statuses = serializers.ListField(
        child=serializers.ChoiceField(
            choices=[
                SEGMENT_STATUS_CANDIDATE,
                SEGMENT_STATUS_CONFIRMED,
                SEGMENT_STATUS_REFINED,
            ]
        ),
        required=False,
        allow_empty=True,
    )
    source_model = serializers.CharField(max_length=128, required=False, allow_blank=True)
    include_geometry = serializers.BooleanField(required=False, default=False)

    def validate(self, attrs):
        has_bbox = "bbox" in attrs
        has_polygon = "polygon_coords" in attrs
        if has_bbox == has_polygon:
            raise serializers.ValidationError("Provide exactly one of bbox or polygon_coords.")

        if has_bbox:
            bbox = attrs["bbox"]
            required_keys = {"x0", "y0", "x1", "y1"}
            if not isinstance(bbox, dict) or not required_keys.issubset(bbox.keys()):
                raise serializers.ValidationError("bbox must include x0, y0, x1, and y1.")
            try:
                x0 = float(bbox["x0"])
                y0 = float(bbox["y0"])
                x1 = float(bbox["x1"])
                y1 = float(bbox["y1"])
            except (TypeError, ValueError):
                raise serializers.ValidationError("bbox values must be numeric.") from None
            if x1 <= x0 or y1 <= y0:
                raise serializers.ValidationError("bbox requires x1>x0 and y1>y0.")
            attrs["bbox"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

        polygon_coords = attrs.get("polygon_coords")
        if polygon_coords is not None and len(polygon_coords) < 3:
            raise serializers.ValidationError("polygon_coords must include at least 3 points.")
        return attrs
