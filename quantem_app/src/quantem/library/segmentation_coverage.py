"""Compact coverage summaries from saved results and explicitly annotated areas."""

from django.db.models import Exists, OuterRef, Prefetch, Q
from shapely.geometry import box
from shapely.ops import unary_union

from quantem.segmentation.models import (
    CompletedROI,
    GlobalMask,
    ProbabilityMap,
    RoiSegmentationStatus,
    SegmentObject,
)


def with_coverage(queryset):
    live = SegmentObject.objects.filter(segmentation_id=OuterRef("pk"), superseded_at__isnull=True)
    return queryset.annotate(
        has_annotations=Exists(live.filter(Q(label_state="CONFIRMED") | Q(source_model="manual"))),
        has_full_result=Exists(live.filter(features__run__scope="full")),
    ).prefetch_related(
        Prefetch(
            "completed_rois", queryset=CompletedROI.objects.only("segmentation_id", "geometry_wkb")
        ),
        Prefetch(
            "roi_statuses",
            queryset=RoiSegmentationStatus.objects.filter(is_complete=True).select_related(
                "image_roi"
            ),
        ),
        Prefetch(
            "probability_maps", queryset=ProbabilityMap.objects.only("segmentation_id", "metadata")
        ),
        Prefetch(
            "global_mask",
            queryset=GlobalMask.objects.only(
                "segmentation_id", "source", "metadata", "foreground_pixels"
            ),
        ),
    )


def _whole_map(metadata):
    # ROI composites also have image-sized canvases; their dimensions do not
    # establish that the areas outside the ROI were ever segmented.
    return bool(metadata and metadata.get("run_scope") == "full" and not metadata.get("composite"))


def coverage(segmentation):
    asset = segmentation.asset
    width, height = asset.logical_width or 0, asset.logical_height or 0
    regions = [row.geometry for row in segmentation.completed_rois.all()]
    for status in segmentation.roi_statuses.all():
        roi = status.image_roi
        regions.append(box(roi.x, roi.y, roi.x + roi.width, roi.y + roi.height))
    merged = unary_union(regions) if regions else None
    if merged is not None and width and height:
        merged = merged.intersection(box(0, 0, width, height))
    area = merged.area if merged is not None else 0
    count = (len(merged.geoms) if hasattr(merged, "geoms") else 1) if area else 0
    percent = min(100.0, area / (width * height) * 100) if width and height else None
    mask = getattr(segmentation, "global_mask", None)
    whole = (
        segmentation.status_stage == "COMPLETED"
        or segmentation.has_full_result
        or any(_whole_map(row.metadata) for row in segmentation.probability_maps.all())
        or bool(mask and _whole_map(mask.metadata))
        or bool(width and height and area >= width * height)
    )
    partial = bool(count or segmentation.has_annotations or (mask and mask.foreground_pixels))
    return {
        "coverage": "whole" if whole else "partial" if partial else "none",
        "annotated_area_count": count,
        "annotated_percent": percent,
    }
