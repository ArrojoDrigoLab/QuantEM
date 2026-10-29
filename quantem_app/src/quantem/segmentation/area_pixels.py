"""Exact area counts with memory bounded by occupied raster tiles."""

import math
from collections import defaultdict

import numpy as np

from quantem.seg_core.rasterize import paint_rings
from quantem.segmentation.geometry import extract_polygons


def count_geometry_pixels(geometries, *, height: int, width: int, tile_size: int = 512) -> int:
    """Count the union using the same pixel convention as canonical masks.

    Sparse annotations on a montage must not allocate or scan the entire image
    just to return an edit response. Overlap and holes still count exactly once.
    """
    tiles = defaultdict(list)
    for geometry in geometries:
        for polygon in extract_polygons(geometry):
            x0, y0, x1, y1 = polygon.bounds
            left, top = max(0, math.ceil(x0)), max(0, math.ceil(y0))
            right, bottom = min(width, math.ceil(x1)), min(height, math.ceil(y1))
            if right <= left or bottom <= top:
                continue
            rings = [
                np.asarray(polygon.exterior.coords, dtype=np.float64),
                *(np.asarray(ring.coords, dtype=np.float64) for ring in polygon.interiors),
            ]
            for cy in range(top // tile_size, (bottom - 1) // tile_size + 1):
                for cx in range(left // tile_size, (right - 1) // tile_size + 1):
                    tiles[cy, cx].append(rings)
    count = 0
    for (cy, cx), polygons in tiles.items():
        x0, y0 = cx * tile_size, cy * tile_size
        block = np.zeros((min(tile_size, height - y0), min(tile_size, width - x0)), dtype=bool)
        for rings in polygons:
            paint_rings(block, rings, 1, x0=x0, y0=y0)
        count += int(np.count_nonzero(block))
    return count
