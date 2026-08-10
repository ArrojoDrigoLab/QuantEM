"""Asset-backed image dimensions for segmentation overlay stores."""

from __future__ import annotations

from quantem.segmentation.models import ImageSegmentation


def segmentation_dimensions(segmentation: ImageSegmentation) -> tuple[int, int]:
    asset = segmentation.asset
    if asset is None:
        raise ValueError(f"Segmentation {segmentation.id} has no target asset.")
    width = int(asset.logical_width or 0)
    height = int(asset.logical_height or 0)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Segmentation {segmentation.id} asset {asset.id} is missing dimensions."
        )
    return width, height
