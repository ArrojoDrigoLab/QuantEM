"""Dirty-region helpers for incremental overlay rebuilds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from shapely.geometry.base import BaseGeometry

from quantem.segmentation.models import ImageSegmentation

from .constants import OVERLAY_CHUNK_SIZE
from .dimensions import segmentation_dimensions


@dataclass(frozen=True)
class DirtyBBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_dict(self) -> dict[str, int]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }


def _dirty_bbox_from_geometry(
    geometry: BaseGeometry | None,
    *,
    image_width: int,
    image_height: int,
) -> DirtyBBox | None:
    if geometry is None or geometry.is_empty:
        return None
    try:
        min_x, min_y, max_x, max_y = geometry.bounds
    except Exception:
        return None
    if max_x <= min_x or max_y <= min_y:
        return None
    return _normalize_dirty_bbox(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        image_width=image_width,
        image_height=image_height,
    )


def _normalize_dirty_bbox(
    *,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    image_width: int,
    image_height: int,
) -> DirtyBBox | None:
    padded_min_x = max(0, int(math.floor(min_x)) - 1)
    padded_min_y = max(0, int(math.floor(min_y)) - 1)
    padded_max_x = min(image_width, int(math.ceil(max_x)) + 1)
    padded_max_y = min(image_height, int(math.ceil(max_y)) + 1)
    if padded_max_x <= padded_min_x or padded_max_y <= padded_min_y:
        return None
    return DirtyBBox(
        x_min=padded_min_x,
        y_min=padded_min_y,
        x_max=padded_max_x,
        y_max=padded_max_y,
    )


def merge_dirty_bboxes(
    segmentation: ImageSegmentation,
    geometries: list[BaseGeometry | None],
) -> DirtyBBox | None:
    image_width, image_height = segmentation_dimensions(segmentation)
    bboxes = [
        bbox
        for bbox in (
            _dirty_bbox_from_geometry(
                geometry,
                image_width=image_width,
                image_height=image_height,
            )
            for geometry in geometries
        )
        if bbox is not None
    ]
    if not bboxes:
        return None
    return DirtyBBox(
        x_min=min(bbox.x_min for bbox in bboxes),
        y_min=min(bbox.y_min for bbox in bboxes),
        x_max=max(bbox.x_max for bbox in bboxes),
        y_max=max(bbox.y_max for bbox in bboxes),
    )


def full_image_dirty_bbox(segmentation: ImageSegmentation) -> DirtyBBox:
    width, height = segmentation_dimensions(segmentation)
    return DirtyBBox(
        x_min=0,
        y_min=0,
        x_max=width,
        y_max=height,
    )


def dirty_bbox_to_chunk_coords(dirty_bbox: DirtyBBox) -> set[tuple[int, int]]:
    if dirty_bbox.width <= 0 or dirty_bbox.height <= 0:
        return set()
    min_chunk_x = dirty_bbox.x_min // OVERLAY_CHUNK_SIZE
    max_chunk_x = (dirty_bbox.x_max - 1) // OVERLAY_CHUNK_SIZE
    min_chunk_y = dirty_bbox.y_min // OVERLAY_CHUNK_SIZE
    max_chunk_y = (dirty_bbox.y_max - 1) // OVERLAY_CHUNK_SIZE
    return {
        (chunk_x, chunk_y)
        for chunk_y in range(min_chunk_y, max_chunk_y + 1)
        for chunk_x in range(min_chunk_x, max_chunk_x + 1)
    }


def _dirty_run_payload(
    *,
    revision: int,
    dirty_bbox: DirtyBBox,
) -> dict[str, Any]:
    chunk_coords = dirty_bbox_to_chunk_coords(dirty_bbox)
    chunk_x_values = [chunk_x for chunk_x, _ in chunk_coords]
    chunk_y_values = [chunk_y for _, chunk_y in chunk_coords]
    return {
        "revision": revision,
        "bbox": dirty_bbox.as_dict(),
        "chunk_x_min": min(chunk_x_values) if chunk_x_values else 0,
        "chunk_x_max": max(chunk_x_values) if chunk_x_values else -1,
        "chunk_y_min": min(chunk_y_values) if chunk_y_values else 0,
        "chunk_y_max": max(chunk_y_values) if chunk_y_values else -1,
    }


def _merge_dirty_runs_to_bbox(
    segmentation: ImageSegmentation,
    runs: list[dict[str, Any]],
) -> DirtyBBox | None:
    dirty_boxes: list[DirtyBBox] = []
    for run in runs:
        bbox = run.get("bbox")
        if not isinstance(bbox, dict):
            continue
        try:
            dirty_boxes.append(
                DirtyBBox(
                    x_min=int(bbox["x_min"]),
                    y_min=int(bbox["y_min"]),
                    x_max=int(bbox["x_max"]),
                    y_max=int(bbox["y_max"]),
                )
            )
        except Exception:
            continue
    if not dirty_boxes:
        return None
    width, height = segmentation_dimensions(segmentation)
    return DirtyBBox(
        x_min=max(0, min(item.x_min for item in dirty_boxes)),
        y_min=max(0, min(item.y_min for item in dirty_boxes)),
        x_max=min(width, max(item.x_max for item in dirty_boxes)),
        y_max=min(height, max(item.y_max for item in dirty_boxes)),
    )
