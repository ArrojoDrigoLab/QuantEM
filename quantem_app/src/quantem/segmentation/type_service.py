"""Helpers for canonical segmentation type creation and lookup."""

from __future__ import annotations

from .models import SegmentationType
from .type_definitions import (
    ANALYSIS_MASK,
    BUILTIN_SEGMENTATION_TYPES,
    ER,
    LIPID_DROPLETS,
    MITOCHONDRIA,
    NUCLEUS,
    TISSUE,
    SegmentationTypeDefinition,
    find_builtin_segmentation_type,
)


def _clean_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("Segmentation type name cannot be empty.")
    return cleaned


def ensure_segmentation_type(
    definition: SegmentationTypeDefinition,
) -> SegmentationType:
    """Get/create a canonical built-in segmentation type and keep fields synced."""
    seg_type, _ = SegmentationType.objects.get_or_create(
        internal_name=definition.internal_name,
        defaults={
            "short_name": definition.short_name,
            "long_name": definition.long_name,
            "measurement_mode": definition.measurement_mode,
        },
    )

    updated_fields: list[str] = []
    if seg_type.short_name != definition.short_name:
        seg_type.short_name = definition.short_name
        updated_fields.append("short_name")
    if seg_type.long_name != definition.long_name:
        seg_type.long_name = definition.long_name
        updated_fields.append("long_name")
    if seg_type.measurement_mode != definition.measurement_mode:
        seg_type.measurement_mode = definition.measurement_mode
        updated_fields.append("measurement_mode")

    if updated_fields:
        seg_type.save(update_fields=updated_fields)
    return seg_type


def ensure_builtin_segmentation_types() -> list[SegmentationType]:
    """Ensure all canonical built-in segmentation types exist."""
    return [ensure_segmentation_type(definition) for definition in BUILTIN_SEGMENTATION_TYPES]


def get_or_create_mitochondria_type() -> SegmentationType:
    return ensure_segmentation_type(MITOCHONDRIA)


def get_or_create_er_type() -> SegmentationType:
    return ensure_segmentation_type(ER)


def get_or_create_nucleus_type() -> SegmentationType:
    return ensure_segmentation_type(NUCLEUS)


def get_or_create_lipid_droplet_type() -> SegmentationType:
    return ensure_segmentation_type(LIPID_DROPLETS)


def get_or_create_tissue_type() -> SegmentationType:
    return ensure_segmentation_type(TISSUE)


def get_or_create_analysis_mask_type() -> SegmentationType:
    return ensure_segmentation_type(ANALYSIS_MASK)


def resolve_or_create_segmentation_type(
    name: str,
    *,
    measurement_mode: str = SegmentationType.MEASUREMENT_MODE_OBJECTS,
) -> SegmentationType:
    """Resolve a user-provided type name to canonical built-in or custom type.

    A custom type's mode is established on first creation and then retained:
    changing it later would make measurements from the same reusable type mean
    different things across images.
    """
    cleaned = _clean_name(name)

    valid_modes = {choice[0] for choice in SegmentationType.MEASUREMENT_MODE_CHOICES}
    if measurement_mode not in valid_modes:
        raise ValueError("Measurement mode must be 'objects' or 'global'.")

    builtin = find_builtin_segmentation_type(cleaned)
    if builtin is not None:
        return ensure_segmentation_type(builtin)

    seg_type, _ = SegmentationType.objects.get_or_create(
        internal_name=cleaned,
        defaults={
            "short_name": cleaned,
            "long_name": cleaned,
            "measurement_mode": measurement_mode,
        },
    )

    # Custom segmentation types intentionally keep all three names aligned.
    updated_fields: list[str] = []
    if seg_type.short_name != cleaned:
        seg_type.short_name = cleaned
        updated_fields.append("short_name")
    if seg_type.long_name != cleaned:
        seg_type.long_name = cleaned
        updated_fields.append("long_name")
    if updated_fields:
        seg_type.save(update_fields=updated_fields)

    return seg_type
