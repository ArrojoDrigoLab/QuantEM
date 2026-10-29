"""Archive per-organelle ROI review state without overwriting another organelle's ROIs."""

import json
from zipfile import ZipFile

from django.db.models import Q
from django.utils.dateparse import parse_datetime

from quantem.assets.models import ImageROI
from quantem.assets.roi_state import activate_roi, get_active_roi_for_segmentation
from quantem.segmentation.models import ImageSegmentation, RoiSegmentationStatus

ARCHIVE_MEMBER = "roi-reviews.json"
ROI_FIELDS = ("display_name", "x", "y", "width", "height", "source")
BOUNDS = ("x", "y", "width", "height")


def save_roi_reviews(archive: ZipFile, segmentation: ImageSegmentation) -> None:
    statuses = {
        str(state.image_roi_id): state
        for state in RoiSegmentationStatus.objects.filter(segmentation=segmentation)
    }
    rois = ImageROI.objects.filter(
        Q(segmentations=segmentation) | Q(id__in=statuses), asset_id=segmentation.asset_id
    ).distinct()
    rows = []
    active = get_active_roi_for_segmentation(segmentation)
    for roi in rois:
        state = statuses.get(str(roi.id))
        rows.append(
            {
                "id": str(roi.id),
                "is_active": active is not None and active.pk == roi.pk,
                **{field: getattr(roi, field) for field in ROI_FIELDS},
                "is_complete": bool(state and state.is_complete),
                "completed_at": state.completed_at.isoformat()
                if state and state.completed_at
                else None,
            }
        )
    archive.writestr(ARCHIVE_MEMBER, json.dumps(rows))


def restore_roi_reviews(archive: ZipFile, segmentation: ImageSegmentation) -> None:
    # Older archives did not record review state. Unknown completion must never
    # turn restored, potentially partial annotations into exhaustive training GT.
    RoiSegmentationStatus.objects.filter(segmentation=segmentation).delete()
    if ARCHIVE_MEMBER not in archive.namelist():
        return
    rows = json.loads(archive.read(ARCHIVE_MEMBER))
    restored_rois = []
    active_roi = None
    for row in rows:
        roi = ImageROI.objects.filter(
            id=row["id"], asset_id=segmentation.asset_id, segmentations=segmentation
        ).first()
        if roi is None or any(getattr(roi, key) != row[key] for key in BOUNDS):
            # ImageROI geometry is shared. Keep its newer geometry for other
            # segmentations and restore a separate window for this segmentation.
            roi = ImageROI.objects.filter(
                asset_id=segmentation.asset_id,
                segmentations=segmentation,
                **{key: row[key] for key in BOUNDS},
            ).first()
        if roi is None:
            roi = ImageROI.objects.create(
                asset_id=segmentation.asset_id,
                **{key: row[key] for key in ROI_FIELDS},
            )
        restored_rois.append(roi)
        if row.get("is_active"):
            active_roi = roi
        RoiSegmentationStatus.objects.update_or_create(
            image_roi=roi,
            segmentation=segmentation,
            defaults={
                "is_complete": row["is_complete"],
                "completed_at": parse_datetime(row["completed_at"])
                if row["completed_at"]
                else None,
            },
        )
    segmentation.rois.set(restored_rois)
    ImageSegmentation.objects.filter(pk=segmentation.pk).update(rois_initialized=True)
    if restored_rois:
        activate_roi(active_roi or restored_rois[-1], segmentation)
