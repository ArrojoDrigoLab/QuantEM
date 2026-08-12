"""Canonical segmentation type definitions used across the backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationTypeDefinition:
    """Canonical naming contract for a segmentation type."""

    internal_name: str
    short_name: str
    long_name: str
    measurement_mode: str = "objects"


MITOCHONDRIA = SegmentationTypeDefinition(
    internal_name="quantem_internal_mito",
    short_name="Mitos",
    long_name="Mitochondria",
)

ER = SegmentationTypeDefinition(
    internal_name="quantem_internal_er",
    short_name="ER",
    long_name="Endoplasmic Reticulum",
    measurement_mode="global",
)

NUCLEUS = SegmentationTypeDefinition(
    internal_name="quantem_internal_nucleus",
    short_name="Nucleus",
    long_name="Nucleus",
)

LIPID_DROPLETS = SegmentationTypeDefinition(
    internal_name="quantem_internal_ld",
    short_name="LD",
    long_name="Lipid Droplets",
)

# Manual-only foreground mask used to exclude white space (vessels, resin,
# padding around the imaged area) from an image. It has no ML segmenter and no
# source model -- the whole mask is drawn by hand with the brush/polygon tools,
# and the analysis suite uses it as the denominator for area fractions.
TISSUE = SegmentationTypeDefinition(
    internal_name="quantem_internal_tissue",
    short_name="Tissue",
    long_name="Tissue Mask",
    measurement_mode="global",
)

# A manual mask that belongs to one image rather than to a reusable
# segmentation type. Its label is stored on ``ImageSegmentation.display_name``
# so one image can have, for example, both a tissue mask and a cells mask
# without making either name appear as a reusable option on another image.
ANALYSIS_MASK = SegmentationTypeDefinition(
    internal_name="quantem_internal_analysis_mask",
    short_name="Analysis mask",
    long_name="Analysis Segmentation Mask",
    measurement_mode="global",
)

# The four released organelles. TISSUE is deliberately excluded: it is a
# user-painted mask layer, not something a model produces.
ORGANELLE_SEGMENTATION_TYPES: tuple[SegmentationTypeDefinition, ...] = (
    MITOCHONDRIA,
    ER,
    NUCLEUS,
    LIPID_DROPLETS,
)

BUILTIN_SEGMENTATION_TYPES: tuple[SegmentationTypeDefinition, ...] = (
    *ORGANELLE_SEGMENTATION_TYPES,
    TISSUE,
    ANALYSIS_MASK,
)

BUILTIN_SEGMENTATION_TYPES_BY_INTERNAL_NAME: dict[str, SegmentationTypeDefinition] = {
    definition.internal_name: definition for definition in BUILTIN_SEGMENTATION_TYPES
}

# Segmentation types that are labeled entirely by hand: creating one never
# enqueues a segmenter job and the segmentation is immediately ready to label.
MANUAL_ONLY_INTERNAL_NAMES = frozenset({TISSUE.internal_name, ANALYSIS_MASK.internal_name})

ORGANELLE_INTERNAL_NAMES = frozenset(
    definition.internal_name for definition in ORGANELLE_SEGMENTATION_TYPES
)


def find_builtin_segmentation_type(
    name: str | None,
) -> SegmentationTypeDefinition | None:
    """Find a built-in segmentation type by exact internal/short/long name."""
    if not name:
        return None
    candidate = name.strip().casefold()
    if not candidate:
        return None

    for definition in BUILTIN_SEGMENTATION_TYPES:
        if candidate in {
            definition.internal_name.casefold(),
            definition.short_name.casefold(),
            definition.long_name.casefold(),
        }:
            return definition
    return None
