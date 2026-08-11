"""Feedback serializers."""

from rest_framework import serializers

from quantem.segmentation.models import UserFeedback


class UserFeedbackSerializer(serializers.ModelSerializer):
    point = serializers.SerializerMethodField()

    class Meta:
        model = UserFeedback
        fields = [
            "id",
            "segmentation",
            "input_type",
            "point",
            "feedback_type",
            "utilized_status",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_point(self, obj):
        # TODO(quantem): confirm the final column names once models.py lands --
        # this assumes the GeoDjango PointField became plain float columns.
        x = getattr(obj, "pt_x", None)
        y = getattr(obj, "pt_y", None)
        if x is None or y is None:
            return None
        return {"x": float(x), "y": float(y)}


class UserFeedbackCreateSerializer(serializers.Serializer):
    input_type = serializers.ChoiceField(
        choices=UserFeedback.INPUT_TYPE_CHOICES,
        default=UserFeedback.INPUT_TYPE_POINT,
    )
    point = serializers.DictField(required=False)
    feedback_type = serializers.ChoiceField(choices=UserFeedback.FEEDBACK_TYPE_CHOICES)

    def validate_point(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("point must be an object with x and y.")
        if "x" not in value or "y" not in value:
            raise serializers.ValidationError("point must include x and y.")
        try:
            x_val = float(value["x"])
            y_val = float(value["y"])
        except (TypeError, ValueError):
            raise serializers.ValidationError("point.x and point.y must be numbers.") from None
        return {"x": x_val, "y": y_val}

    def validate(self, attrs):
        input_type = attrs.get("input_type", UserFeedback.INPUT_TYPE_POINT)
        if input_type != UserFeedback.INPUT_TYPE_POINT:
            raise serializers.ValidationError({"input_type": "Only point feedback is supported."})
        if attrs.get("point") is None:
            raise serializers.ValidationError(
                {"point": "point is required when input_type='point'."}
            )
        return attrs

    def create(self, validated_data):
        raise NotImplementedError

    def to_feedback_kwargs(self):
        data = self.validated_data
        point_data = data["point"]
        return {
            "input_type": data["input_type"],
            "pt_x": float(point_data["x"]),
            "pt_y": float(point_data["y"]),
            "feedback_type": data["feedback_type"],
        }
