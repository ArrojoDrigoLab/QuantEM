"""Area-only reporting for binary global segmentations."""

from __future__ import annotations

import uuid

import numpy as np

from quantem.analysis.loaders import image_shape, segmentation_mask
from quantem.segmentation.models import ImageSegmentation


def _area_row(*, name: str, numerator: int, denominator: int, segmentation_id=None) -> dict:
    return {
        "segmentation_id": str(segmentation_id) if segmentation_id is not None else None,
        "name": name,
        "foreground_pixels": int(numerator),
        "denominator_pixels": int(denominator),
        "foreground_percent": (
            (100.0 * float(numerator) / float(denominator)) if denominator > 0 else None
        ),
    }


def global_area_report(
    segmentation: ImageSegmentation,
    *,
    analysis_mask_ids: list[str] | tuple[str, ...] = (),
) -> dict:
    """Foreground percentage for the whole image and selected analysis masks."""
    if segmentation.segmentation_type.measurement_mode != "global":
        raise ValueError("Area-only analysis is available only for global segmentations.")
    foreground = segmentation_mask(segmentation, image_shape(segmentation))
    whole_numerator = int(np.count_nonzero(foreground))
    rows = []
    requested = []
    for value in analysis_mask_ids:
        if not str(value).strip():
            continue
        try:
            mask_id = str(uuid.UUID(str(value)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Every selected analysis mask must have a valid identifier.") from exc
        if mask_id not in requested:
            requested.append(mask_id)
    if requested:
        masks = {
            str(mask.id): mask
            for mask in ImageSegmentation.objects.select_related("segmentation_type").filter(
                id__in=requested,
                asset_id=segmentation.asset_id,
                segmentation_type__category="area",
                superseded_by__isnull=True,
                segmentation_type__measurement_mode="global",
            )
        }
        missing = [value for value in requested if value not in masks]
        if missing:
            raise ValueError(
                "Every selected measurement region must be an area definition on this image."
            )
        for mask_id in requested:
            selected = masks[mask_id]
            denominator_mask = segmentation_mask(selected, image_shape(segmentation))
            denominator = int(np.count_nonzero(denominator_mask))
            numerator = int(np.count_nonzero(foreground & denominator_mask))
            rows.append(
                _area_row(
                    name=selected.display_name or selected.segmentation_type.long_name,
                    numerator=numerator,
                    denominator=denominator,
                    segmentation_id=selected.id,
                )
            )

    return {
        "measurement_mode": "global",
        "metric": "foreground_area_percent",
        "segmentation_id": str(segmentation.id),
        "whole_image": _area_row(
            name="Whole image",
            numerator=whole_numerator,
            denominator=int(foreground.size),
        ),
        "analysis_masks": rows,
    }
