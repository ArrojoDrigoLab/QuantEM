"""Start an annotation ROI from model objects or an empty model-derived area."""

from django.db import transaction
from shapely.geometry import box

from quantem.segmentation.coverage import completed_geometry
from quantem.segmentation.models import ImageSegmentation, SegmentObject
from quantem.segmentation.saved_results import _replace_geometry


@transaction.atomic
def initialize_roi(segmentation, roi, *, prepopulate):
    ImageSegmentation.objects.select_for_update().get(pk=segmentation.pk)
    region = box(roi.x, roi.y, roi.x + roi.width, roi.y + roi.height).difference(
        completed_geometry(segmentation)
    )
    if region.is_empty:
        return
    from quantem.segmentation.global_masks import (
        _rasterize_geometries,
        load_global_mask,
        save_global_mask,
    )
    from quantem.segmentation.mutations import (
        register_overlay_mutation_all_bundles,
    )
    from quantem.segmentation.overlay_ngff import (
        merge_dirty_bboxes,
    )
    from quantem.segmentation.revisions import save_revision

    if segmentation.segmentation_type.measurement_mode == "global":
        if not prepopulate:
            save_revision(segmentation, "Before starting a blank ROI")
            mask = load_global_mask(segmentation)
            mask[_rasterize_geometries([region], height=mask.shape[0], width=mask.shape[1])] = False
            previous = getattr(segmentation, "global_mask", None)
            save_global_mask(
                segmentation, mask, source="manual", metadata=previous.metadata if previous else {}
            )
    else:
        rows = SegmentObject.objects.filter(segmentation=segmentation, superseded_at=None).exclude(
            source_model="manual"
        )
        rows = rows.filter(
            bbox_maxx__gt=roi.x,
            bbox_minx__lt=roi.x + roi.width,
            bbox_maxy__gt=roi.y,
            bbox_miny__lt=roi.y + roi.height,
        )
        if prepopulate:
            rows = rows.filter(label_state__in=["CANDIDATE", "INFERRED"])
        if not rows.exists():
            return
        save_revision(segmentation, "Before preparing annotation ROI")
        for row in rows.iterator():
            inside = row.geometry.intersection(region)
            if inside.is_empty:
                continue
            outside = row.geometry.difference(region)
            _replace_geometry(row, outside, label_state=row.label_state)
            if prepopulate:
                _replace_geometry(row, inside, label_state="CONFIRMED")
    register_overlay_mutation_all_bundles(
        segmentation,
        dirty_bbox=merge_dirty_bboxes(segmentation, [region]),
        allow_sync_partial=False,
        force_full_rebuild=segmentation.segmentation_type.measurement_mode == "global",
    )
