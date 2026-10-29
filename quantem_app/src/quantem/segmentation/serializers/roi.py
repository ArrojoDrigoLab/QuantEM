"""ROI-related serializers."""

from rest_framework import serializers

from quantem.assets.models import ImageROI
from quantem.assets.roi_state import get_active_roi_for_segmentation
from quantem.segmentation.models import CompletedROI, RoiSegmentationStatus


class SegmentationRoiSerializer(serializers.ModelSerializer):
    segmentation = serializers.SerializerMethodField()
    completed_for_segmentation = serializers.SerializerMethodField()
    is_active = serializers.SerializerMethodField()
    is_complete = serializers.SerializerMethodField()

    class Meta:
        model = ImageROI
        fields = [
            "id",
            "segmentation",
            "x",
            "y",
            "width",
            "height",
            "source",
            "is_active",
            "is_complete",
            "completed_for_segmentation",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_segmentation(self, obj):
        segmentation = self.context.get("segmentation")
        return str(segmentation.id) if segmentation else None

    def get_is_active(self, obj):
        segmentation = self.context.get("segmentation")
        if segmentation is None:
            return obj.is_active
        if "active_roi_id" not in self.context:
            active = get_active_roi_for_segmentation(segmentation)
            self.context["active_roi_id"] = active.pk if active else None
        return obj.pk == self.context["active_roi_id"]

    def get_is_complete(self, obj):
        if self.context.get("segmentation") is None:
            return obj.is_complete
        return self.get_completed_for_segmentation(obj)

    def get_completed_for_segmentation(self, obj):
        """Per-organelle completion of this ROI for the context segmentation.

        Returns ``None`` when no segmentation context is supplied. Views that
        serialize many ROIs may pass a prebuilt ``roi_completion_map``
        (``{roi_id: bool}``) in the context to avoid an N+1 query.
        """
        segmentation = self.context.get("segmentation")
        if segmentation is None:
            return None
        completion_map = self.context.get("roi_completion_map")
        if completion_map is None:
            completion_map = dict(
                RoiSegmentationStatus.objects.filter(segmentation=segmentation).values_list(
                    "image_roi_id", "is_complete"
                )
            )
            self.context["roi_completion_map"] = completion_map
        return bool(completion_map.get(obj.id, False))


class CompletedRoiSerializer(serializers.ModelSerializer):
    polygon_coords = serializers.SerializerMethodField()
    holes = serializers.SerializerMethodField()
    bbox = serializers.SerializerMethodField()

    class Meta:
        model = CompletedROI
        fields = [
            "id",
            "segmentation",
            "polygon_coords",
            "holes",
            "bbox",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_polygon_coords(self, obj):
        geometry = obj.geometry
        if geometry is None or geometry.is_empty or geometry.geom_type != "Polygon":
            return []
        return [[float(x), float(y)] for x, y in geometry.exterior.coords]

    def get_holes(self, obj):
        geometry = obj.geometry
        if geometry is None or geometry.is_empty or geometry.geom_type != "Polygon":
            return []
        return [[[float(x), float(y)] for x, y in ring.coords] for ring in geometry.interiors]

    def get_bbox(self, obj):
        bbox = obj.bbox
        if bbox is None or bbox.is_empty:
            return None
        min_x, min_y, max_x, max_y = bbox.bounds
        return {
            "x0": float(min_x),
            "y0": float(min_y),
            "x1": float(max_x),
            "y1": float(max_y),
        }


class CompletedRoiCreateSerializer(serializers.Serializer):
    polygon_coords = serializers.ListField(
        child=serializers.ListField(
            child=serializers.FloatField(),
            min_length=2,
            max_length=2,
        ),
        min_length=3,
    )
