"""Box, crop and mask geometry.

Three coordinate systems meet here and getting the round trip wrong is the
classic way this port breaks, so each conversion is one named function with a
test rather than arithmetic inlined at a call site:

``global``
    Full-image pixels. What the client sends, and what a stored
    :class:`~quantem.segmentation.models.SegmentObject` is in.
``crop``
    Pixels within the window handed to the encoder. Its origin is the window's
    top-left corner in global pixels.
``mask``
    Rows and columns of the array the backend returns. Usually the same
    resolution as the crop, but not guaranteed to be, so the scale is carried
    explicitly instead of assumed to be 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from quantem.sam.config import BBOX_CONTEXT_RADIUS, CROP_GRID


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectangle, ``x0 < x1`` and ``y0 < y1``."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def normalized(cls, x0: float, y0: float, x1: float, y1: float) -> Box:
        """A box from two corners in any order."""
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class Crop:
    """A window of the image, in global pixels, with integer bounds."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        return self.x + self.width

    @property
    def y1(self) -> int:
        return self.y + self.height

    def contains(self, box: Box) -> bool:
        return (
            box.x0 >= self.x
            and box.y0 >= self.y
            and box.x1 <= self.x1
            and box.y1 <= self.y1
        )

    def key(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


def _clamp_window(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    image_width: int,
    image_height: int,
) -> Crop:
    cx0 = max(0, int(math.floor(x0)))
    cy0 = max(0, int(math.floor(y0)))
    cx1 = min(int(image_width), int(math.ceil(x1)))
    cy1 = min(int(image_height), int(math.ceil(y1)))
    return Crop(cx0, cy0, max(cx1 - cx0, 1), max(cy1 - cy0, 1))


def plan_crop(box: Box, image_width: int, image_height: int) -> Crop:
    """The window the encoder should see for ``box``.

    Preferred window: the :data:`~quantem.sam.config.CROP_GRID` cell holding the
    box's centre, grown by :data:`~quantem.sam.config.BBOX_CONTEXT_RADIUS` on
    every side and clamped to the image. Because it is derived from the *cell*
    and not from the box, every box centred in that cell gets the same window
    and so shares one encode -- which is the entire point of the cache.

    A box that does not fit inside that window (too large, or centred near a
    cell edge and extending well past the margin) falls back to its own padded
    rect. That answer is correct but shares nothing; the alternative would be to
    grow the shared window until it fits, and a single big box would then poison
    the cell for every small box after it.
    """
    center_x, center_y = box.center
    cell_x = math.floor(center_x / CROP_GRID) * CROP_GRID
    cell_y = math.floor(center_y / CROP_GRID) * CROP_GRID
    shared = _clamp_window(
        cell_x - BBOX_CONTEXT_RADIUS,
        cell_y - BBOX_CONTEXT_RADIUS,
        cell_x + CROP_GRID + BBOX_CONTEXT_RADIUS,
        cell_y + CROP_GRID + BBOX_CONTEXT_RADIUS,
        image_width,
        image_height,
    )
    # A box clipped by the image edge is still "contained": the clamp above
    # already cut the window at the same edge, so compare against the box as
    # the image sees it.
    visible = Box(
        max(box.x0, 0.0),
        max(box.y0, 0.0),
        min(box.x1, float(image_width)),
        min(box.y1, float(image_height)),
    )
    if shared.contains(visible):
        return shared
    return _clamp_window(
        box.x0 - BBOX_CONTEXT_RADIUS,
        box.y0 - BBOX_CONTEXT_RADIUS,
        box.x1 + BBOX_CONTEXT_RADIUS,
        box.y1 + BBOX_CONTEXT_RADIUS,
        image_width,
        image_height,
    )


def box_to_crop(box: Box, crop: Crop) -> Box | None:
    """``box`` in crop-relative pixels, or ``None`` if it misses the crop.

    Clipped to the crop rather than allowed to run outside it: the backend is
    handed only the crop's pixels, and a prompt naming coordinates it cannot see
    is not a prompt it can answer.
    """
    lx0 = max(0.0, box.x0 - crop.x)
    ly0 = max(0.0, box.y0 - crop.y)
    lx1 = min(float(crop.width), box.x1 - crop.x)
    ly1 = min(float(crop.height), box.y1 - crop.y)
    if lx1 <= lx0 or ly1 <= ly0:
        return None
    return Box(lx0, ly0, lx1, ly1)


def binarize(mask: np.ndarray) -> np.ndarray:
    """A boolean mask from whatever the backend returned.

    Backends differ: booleans, ``{0, 1}`` integers, ``{0, 255}`` images and
    ``[0, 1]`` probabilities all turn up, and reading one as another silently
    yields either an empty object or a full-crop one.
    """
    if mask.dtype == bool:
        return mask
    if np.issubdtype(mask.dtype, np.floating):
        peak = float(mask.max()) if mask.size else 0.0
        return mask > (0.5 if peak <= 1.0 else 0.0)
    peak = int(mask.max()) if mask.size else 0
    if peak <= 1:
        return mask.astype(bool)
    return mask > 127


def mask_to_global_polygon(
    mask: np.ndarray,
    crop: Crop,
) -> tuple[Polygon, float] | None:
    """The largest blob in ``mask``, as a polygon in global image pixels.

    Returns the polygon and its area in global pixels, or ``None`` when the mask
    holds nothing that survives becoming a polygon.

    Only the largest contour is kept, and holes are discarded -- the same
    treatment the rest of this codebase gives a model mask. For organelles that
    is the right answer; a box prompt is a request for *one* object.

    ``mask`` is allowed to be at a different resolution from ``crop``; the scale
    is derived from the two shapes rather than assumed, because a backend that
    returns masks on its own grid would otherwise put every object in the wrong
    place at the wrong size.
    """
    from quantem.segmentation.utils import mask_to_polygon

    binary = binarize(mask)
    if not binary.any():
        return None

    rows, cols = binary.shape[-2], binary.shape[-1]
    scale_x = crop.width / float(cols)
    scale_y = crop.height / float(rows)

    polygon, _centroid, _bbox = mask_to_polygon(binary)
    if polygon.is_empty:
        return None
    # (a, b, d, e, xoff, yoff): x' = a*x + b*y + xoff, y' = d*x + e*y + yoff.
    placed = affine_transform(
        polygon,
        [scale_x, 0.0, 0.0, scale_y, float(crop.x), float(crop.y)],
    )
    if placed.is_empty or placed.area <= 0.0:
        return None
    return placed, float(placed.area)


def polygon_coords(polygon: Polygon) -> list[list[float]]:
    """The exterior ring as ``[[x, y], ...]``, for a JSON response."""
    return [[float(x), float(y)] for x, y in polygon.exterior.coords]
