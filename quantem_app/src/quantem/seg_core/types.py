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

    prob_maps: dict[str, np.ndarray]  # Named DL outputs: {"DINO": arr, ...}
    prob: np.ndarray  # Foreground probability map instances are extracted from [0, 1]
    extracted_segments: list[ExtractedSegment] | None = None
    artifacts: dict[str, Any] | None = None
