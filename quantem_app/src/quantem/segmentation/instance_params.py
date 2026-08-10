"""Helpers for per-segmentation instance extraction parameterization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .type_definitions import LIPID_DROPLETS, MITOCHONDRIA, NUCLEUS

INSTANCE_PARAM_CENTER_MIN_DISTANCE = "center_min_distance"
INSTANCE_PARAM_CENTER_CONFIDENCE_THRESHOLD = "center_confidence_threshold"
INSTANCE_PARAM_SEGMENTATION_THRESHOLD = "segmentation_threshold"
INSTANCE_PARAM_DOWNSAMPLING_FACTOR = "downsampling_factor"

INSTANCE_PARAM_KEYS = (
    INSTANCE_PARAM_CENTER_MIN_DISTANCE,
    INSTANCE_PARAM_CENTER_CONFIDENCE_THRESHOLD,
    INSTANCE_PARAM_SEGMENTATION_THRESHOLD,
    INSTANCE_PARAM_DOWNSAMPLING_FACTOR,
)

SUPPORTED_INSTANCE_PARAM_INTERNAL_NAMES = frozenset(
    {
        MITOCHONDRIA.internal_name,
        NUCLEUS.internal_name,
        LIPID_DROPLETS.internal_name,
    }
)

_DEFAULT_INSTANCE_PARAMS: dict[str, int | float | None] = {
    INSTANCE_PARAM_CENTER_MIN_DISTANCE: 8,
    INSTANCE_PARAM_CENTER_CONFIDENCE_THRESHOLD: 0.30,
    INSTANCE_PARAM_SEGMENTATION_THRESHOLD: 0.50,
    INSTANCE_PARAM_DOWNSAMPLING_FACTOR: None,
}


def instance_params_defaults() -> dict[str, int | float | None]:
    """Return a fresh copy of default instance parameters."""
    return dict(_DEFAULT_INSTANCE_PARAMS)


def supports_instance_params(segmentation_type_internal_name: str | None) -> bool:
    """Return whether a segmentation type supports instance params."""
    normalized = (segmentation_type_internal_name or "").strip()
    return normalized in SUPPORTED_INSTANCE_PARAM_INTERNAL_NAMES


def coerce_instance_params(
    params: Mapping[str, Any] | None,
) -> dict[str, int | float | None]:
    """Merge persisted params over defaults and coerce basic types."""
    merged = instance_params_defaults()
    if not params:
        return merged

    raw_center_min_distance = params.get(INSTANCE_PARAM_CENTER_MIN_DISTANCE)
    if raw_center_min_distance is not None:
        try:
            value = int(raw_center_min_distance)
            if value >= 1:
                merged[INSTANCE_PARAM_CENTER_MIN_DISTANCE] = value
        except (TypeError, ValueError):
            pass

    raw_center_confidence = params.get(INSTANCE_PARAM_CENTER_CONFIDENCE_THRESHOLD)
    if raw_center_confidence is not None:
        try:
            value = float(raw_center_confidence)
            if 0.0 <= value <= 1.0:
                merged[INSTANCE_PARAM_CENTER_CONFIDENCE_THRESHOLD] = value
        except (TypeError, ValueError):
            pass

    raw_segmentation_threshold = params.get(INSTANCE_PARAM_SEGMENTATION_THRESHOLD)
    if raw_segmentation_threshold is not None:
        try:
            value = float(raw_segmentation_threshold)
            if 0.0 <= value <= 1.0:
                merged[INSTANCE_PARAM_SEGMENTATION_THRESHOLD] = value
        except (TypeError, ValueError):
            pass

    raw_downsampling_factor = params.get(INSTANCE_PARAM_DOWNSAMPLING_FACTOR)
    if raw_downsampling_factor is None:
        merged[INSTANCE_PARAM_DOWNSAMPLING_FACTOR] = None
    else:
        try:
            value = int(raw_downsampling_factor)
            if value >= 1:
                merged[INSTANCE_PARAM_DOWNSAMPLING_FACTOR] = value
        except (TypeError, ValueError):
            pass

    return merged

