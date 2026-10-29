"""Independent ROI membership and selection for each segmentation.

Existing memberships may share an immutable crop. Editing replaces a membership
with a new crop; deleting unlinks only that segmentation. Image-level flags are
retained for preprocessing templates and older database recovery only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.db import transaction

from .models import Asset, ImageROI

if TYPE_CHECKING:
    from quantem.segmentation.models import ImageSegmentation


def rois_for_segmentation(segmentation: ImageSegmentation):
    return ImageROI.objects.filter(asset_id=segmentation.asset_id, segmentations=segmentation)


def get_active_roi_for_segmentation(segmentation: ImageSegmentation) -> ImageROI | None:
    rois = rois_for_segmentation(segmentation)
    active = rois.filter(
        segmentation_statuses__segmentation=segmentation,
        segmentation_statuses__is_active=True,
    ).first()
    return active or rois.order_by("-created_at", "-id").first()


def get_active_roi_for_asset(asset: Asset | None) -> ImageROI | None:
    """Return an unassigned preprocessing template, never another organelle's ROI."""
    if asset is None:
        return None
    return (
        ImageROI.objects.filter(
            asset=asset,
            source="AUTO",
            is_active=True,
            segmentations__isnull=True,
            segmentation_statuses__isnull=True,
        )
        .order_by("-created_at", "-id")
        .first()
    )


@transaction.atomic
def initialize_roi_from_template(
    segmentation: ImageSegmentation, template: ImageROI | None = None
) -> ImageROI | None:
    """Adopt a preprocessing crop only before this workspace has been opened.

    Uploads may finish after a user has created or removed ROIs. Serialize with
    those edits and keep their choice, including an intentionally empty list.
    An upload can explicitly share its immutable crop with several organelles.
    """
    from quantem.segmentation.models import ImageSegmentation

    locked = ImageSegmentation.objects.select_for_update().get(pk=segmentation.pk)
    segmentation.rois_initialized = locked.rois_initialized
    active = get_active_roi_for_segmentation(locked)
    if locked.rois_initialized or active is not None:
        return active
    explicit_template = template is not None
    template = template or get_active_roi_for_asset(locked.asset)
    if template is None:
        return None
    template = (
        ImageROI.objects.select_for_update()
        .filter(pk=template.pk, asset_id=locked.asset_id)
        .first()
    )
    if template is None:
        return None
    if not explicit_template and (
        not template.is_active
        or template.segmentations.exists()
        or template.segmentation_statuses.exists()
    ):
        return None
    template.segmentations.add(locked)
    # Once used for labeling it is never an unassigned preprocessing template
    # again, even if a revision later removes its final membership.
    ImageROI.objects.filter(pk=template.pk).update(is_active=False)
    activate_roi(template, locked)
    segmentation.rois_initialized = True
    return template


@transaction.atomic
def activate_roi(roi: ImageROI, segmentation: ImageSegmentation | None = None) -> ImageROI:
    if segmentation is None:
        if roi.segmentations.exists():
            raise ValueError("Choose a segmentation when activating a labeling ROI.")
        ImageROI.objects.filter(asset_id=roi.asset_id, segmentations__isnull=True).exclude(
            id=roi.id
        ).update(is_active=False)
        roi.is_active = True
        roi.save(update_fields=["is_active", "updated_at"])
        return roi

    from quantem.segmentation.models import ImageSegmentation, RoiSegmentationStatus

    ImageSegmentation.objects.select_for_update().get(pk=segmentation.pk)
    if not rois_for_segmentation(segmentation).filter(pk=roi.pk).exists():
        raise ValueError("This ROI does not belong to the selected segmentation.")
    RoiSegmentationStatus.objects.filter(segmentation=segmentation, is_active=True).update(
        is_active=False
    )
    RoiSegmentationStatus.objects.update_or_create(
        segmentation=segmentation, image_roi=roi, defaults={"is_active": True}
    )
    ImageSegmentation.objects.filter(pk=segmentation.pk).update(rois_initialized=True)
    segmentation.rois_initialized = True
    return roi
