"""Helpers for resolving and updating active ROI state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.db import transaction

from .models import Asset, ImageROI

if TYPE_CHECKING:
    from quantem.segmentation.models import ImageSegmentation


def get_active_roi_for_segmentation(segmentation: ImageSegmentation) -> ImageROI | None:
    roi = segmentation.rois.filter(is_active=True).order_by("-created_at").first()
    if roi is not None:
        return roi
    roi = segmentation.rois.order_by("-created_at").first()
    if roi is not None:
        return roi
    if segmentation.asset_id:
        return get_active_roi_for_asset(segmentation.asset)
    return None


def get_active_roi_for_asset(asset: Asset | None) -> ImageROI | None:
    if asset is None:
        return None
    active = ImageROI.objects.filter(asset=asset, is_active=True).order_by("-created_at").first()
    if active is not None:
        return active
    return ImageROI.objects.filter(asset=asset).order_by("-created_at").first()


@transaction.atomic
def activate_roi(roi: ImageROI) -> ImageROI:
    if roi.asset_id:
        ImageROI.objects.filter(asset=roi.asset).exclude(id=roi.id).update(is_active=False)
    roi.is_active = True
    roi.save(update_fields=["is_active", "updated_at"])
    return roi
