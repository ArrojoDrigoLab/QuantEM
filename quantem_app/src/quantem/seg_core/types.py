"""
Shared Dataclasses for Organelle Segmentation
===============================================

DB-agnostic data types used across all segmenter implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ExtractedSegment:
    """A single extracted segment from a probability map."""

    polygon_coords: list[tuple[float, float]]  # Closed polygon [(x,y), ...]
    centroid_xy: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    area: int
    features: dict[str, Any]
    confidence_score: float | None
    region_mask: np.ndarray | None = None  # For rasterization; not persisted


@dataclass
class InferenceResult:
    """Result of organelle inference containing named probability maps."""

    # Values are normally NumPy arrays. Replaying a very large stored result
    # supplies a sliceable uint8 Zarr array so extraction never materializes it.
    prob_maps: dict[str, Any]  # Named DL outputs: {"DINO": arr, ...}
    prob: Any  # Foreground probability map or stored uint8 threshold authority
    extracted_segments: list[ExtractedSegment] | None = None
    artifacts: dict[str, Any] | None = None
