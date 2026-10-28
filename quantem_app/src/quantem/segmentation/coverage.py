"""Whether a segmentation describes the whole image, without a review lock."""

from shapely.geometry import box
from shapely.ops import unary_union

from quantem.segmentation.models import CompletedROI, RoiSegmentationStatus


def completed_geometry(segmentation):
    if segmentation.manual_full_image and segmentation.asset:
        return box(0, 0, segmentation.asset.logical_width, segmentation.asset.logical_height)
    outlines = [
        row.geometry
        for row in CompletedROI.objects.filter(segmentation=segmentation)
        if row.geometry is not None and not row.geometry.is_empty
    ]
    for row in RoiSegmentationStatus.objects.filter(
        segmentation=segmentation, is_complete=True
    ).select_related("image_roi"):
        roi = row.image_roi
        if roi and roi.width > 0 and roi.height > 0:
            outlines.append(box(roi.x, roi.y, roi.x + roi.width, roi.y + roi.height))
    return unary_union(outlines)


def whole_image_manually_covered(segmentation):
    if segmentation.manual_full_image:
        return True
    asset = segmentation.asset
    if not asset or not asset.logical_width or not asset.logical_height:
        return False
    return completed_geometry(segmentation).covers(
        box(0, 0, asset.logical_width, asset.logical_height)
    )


def coverage_state(segmentation):
    if segmentation.status_stage in {
        "RUNNING_INFERENCE",
        "EXTRACTING_CANDIDATES",
        "UPDATING",
        "COMPUTING_FEATURES",
        "FAILED",
    }:
        return {"ready": False, "kind": "processing", "result_id": None}
    canonical = segmentation.saved_results.filter(is_canonical=True, deleted_at=None).first()
    if canonical and canonical.full_image:
        return {"ready": True, "kind": "model", "result_id": str(canonical.id)}
    if whole_image_manually_covered(segmentation):
        return {"ready": True, "kind": "manual", "result_id": None}
    # Existing completed results remain usable. Their recorded model identity is
    # retained as-is; an unknown legacy model is never labelled as a newer run.
    if segmentation.status_stage == "COMPLETED":
        return {"ready": True, "kind": "legacy", "result_id": None}
    return {"ready": False, "kind": "partial", "result_id": None}
