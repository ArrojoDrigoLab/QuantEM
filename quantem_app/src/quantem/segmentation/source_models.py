"""Code-backed source model catalog and resolver for organelle segmentations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from django.db.models import Q

from .type_definitions import (
    ER,
    LIPID_DROPLETS,
    MITOCHONDRIA,
    NUCLEUS,
    SegmentationTypeDefinition,
)

SOURCE_MODEL_MANUAL = "manual"
SOURCE_MODEL_UNKNOWN = "unknown"

MODEL_FAMILY_QUANTEM = "quantem"
MODEL_FAMILY_OMNIEM = "omniem"


@dataclass(frozen=True)
class SourceModelDefinition:
    value: str
    label: str
    organelle_internal_name: str
    segmenter_internal_name: str
    model_family: str
    variant: str = ""
    is_default: bool = False


# The eight released models: four organelles x {QuantEM ViT-B, OmniEM ViT-L}.
# Both families route to the DINO segmenters (registered as ``dino_<organelle>``),
# which pick the checkpoint from the source model. QuantEM is the default for
# every organelle.
SOURCE_MODEL_DEFINITIONS: tuple[SourceModelDefinition, ...] = (
    # === Mitochondria ===
    SourceModelDefinition(
        value="quantem:mito",
        label="QuantEM",
        organelle_internal_name=MITOCHONDRIA.internal_name,
        segmenter_internal_name="dino_mito",
        model_family=MODEL_FAMILY_QUANTEM,
        is_default=True,
    ),
    SourceModelDefinition(
        value="omniem:mito",
        label="OmniEM",
        organelle_internal_name=MITOCHONDRIA.internal_name,
        segmenter_internal_name="dino_mito",
        model_family=MODEL_FAMILY_OMNIEM,
    ),
    # === Endoplasmic reticulum ===
    SourceModelDefinition(
        value="quantem:er",
        label="QuantEM",
        organelle_internal_name=ER.internal_name,
        segmenter_internal_name="dino_er",
        model_family=MODEL_FAMILY_QUANTEM,
        is_default=True,
    ),
    SourceModelDefinition(
        value="omniem:er",
        label="OmniEM",
        organelle_internal_name=ER.internal_name,
        segmenter_internal_name="dino_er",
        model_family=MODEL_FAMILY_OMNIEM,
    ),
    # === Lipid droplets ===
    SourceModelDefinition(
        value="quantem:ld",
        label="QuantEM",
        organelle_internal_name=LIPID_DROPLETS.internal_name,
        segmenter_internal_name="dino_ld",
        model_family=MODEL_FAMILY_QUANTEM,
        is_default=True,
    ),
    SourceModelDefinition(
        value="omniem:ld",
        label="OmniEM",
        organelle_internal_name=LIPID_DROPLETS.internal_name,
        segmenter_internal_name="dino_ld",
        model_family=MODEL_FAMILY_OMNIEM,
    ),
    # === Nucleus ===
    SourceModelDefinition(
        value="quantem:nucleus",
        label="QuantEM",
        organelle_internal_name=NUCLEUS.internal_name,
        segmenter_internal_name="dino_nucleus",
        model_family=MODEL_FAMILY_QUANTEM,
        is_default=True,
    ),
    SourceModelDefinition(
        value="omniem:nucleus",
        label="OmniEM",
        organelle_internal_name=NUCLEUS.internal_name,
        segmenter_internal_name="dino_nucleus",
        model_family=MODEL_FAMILY_OMNIEM,
    ),
)

SOURCE_MODEL_DEFINITIONS_BY_VALUE = {
    definition.value: definition for definition in SOURCE_MODEL_DEFINITIONS
}


def normalize_source_model(value: str | None) -> str:
    return (value or "").strip().lower()


def default_source_model_for_organelle(internal_name: str | None) -> str:
    for definition in SOURCE_MODEL_DEFINITIONS:
        if definition.organelle_internal_name == internal_name and definition.is_default:
            return definition.value
    return SOURCE_MODEL_UNKNOWN


def source_models_for_organelle(internal_name: str | None) -> tuple[SourceModelDefinition, ...]:
    return tuple(
        definition
        for definition in SOURCE_MODEL_DEFINITIONS
        if definition.organelle_internal_name == internal_name
    )


def get_source_model_definition(value: str | None) -> SourceModelDefinition | None:
    return SOURCE_MODEL_DEFINITIONS_BY_VALUE.get(normalize_source_model(value))


def resolve_create_segmentation_request(
    segmentation_type_definition: SegmentationTypeDefinition,
    source_model: str | None = None,
) -> tuple[SegmentationTypeDefinition, str]:
    requested = normalize_source_model(source_model)
    if requested:
        return segmentation_type_definition, requested

    return (
        segmentation_type_definition,
        default_source_model_for_organelle(segmentation_type_definition.internal_name),
    )


def resolve_segmenter_internal_name(
    *,
    segmentation_type_internal_name: str,
    source_model: str | None = None,
) -> str:
    normalized = normalize_source_model(source_model)
    if not normalized:
        normalized = default_source_model_for_organelle(segmentation_type_internal_name)

    definition = get_source_model_definition(normalized)
    if definition is None:
        return segmentation_type_internal_name
    if definition.organelle_internal_name != segmentation_type_internal_name:
        raise ValueError(
            f"Source model {normalized!r} is not valid for segmentation type "
            f"{segmentation_type_internal_name!r}."
        )
    return definition.segmenter_internal_name


def infer_source_model_from_features(
    *,
    segmentation_type_internal_name: str | None,
    features: object,
    label_state: str | None = None,
) -> str:
    """Best-effort source model for an object that has none recorded.

    The extractors stamp ``model_family`` (``quantem``/``omniem``) into
    ``features``; rows written before that carry only a per-organelle
    ``<organelle>_generated`` marker and fall back to the organelle default.
    Anything hand-labeled resolves to ``manual``.
    """
    if isinstance(features, dict):
        model_family = str(features.get("model_family") or "").strip().lower()
        if model_family in {MODEL_FAMILY_QUANTEM, MODEL_FAMILY_OMNIEM}:
            for definition in source_models_for_organelle(segmentation_type_internal_name):
                if definition.model_family == model_family:
                    return definition.value
        if any(
            features.get(f"{marker}_generated") for marker in ("mito", "er", "nucleus", "lipid")
        ):
            return default_source_model_for_organelle(segmentation_type_internal_name)

    if label_state in {"CONFIRMED", "EXCLUDED"}:
        return SOURCE_MODEL_MANUAL
    return default_source_model_for_organelle(segmentation_type_internal_name)


def overlay_bundle_source_filter(source_model: str | None) -> Q | None:
    """Objects stored in one model-specific raster bundle.

    The source-less bundle is the shared confirmed-display raster and keeps all
    objects so a later state-only confirmation can reveal an already-painted
    label without rebuilding it.  A named bundle keeps that model's objects
    **and every hand-drawn one**, and the client chooses candidate or confirmed
    visibility through the LUT.

    Hand-drawn objects are in here because the viewer composites exactly two
    layers, and no other arrangement can show one.  The candidate layer reads
    the named bundle with ``confirmed`` and ``excluded`` hidden; the confirmed
    layer reads the source-less bundle with ``candidate`` and ``excluded``
    hidden (``useOverlayLayerControls``).  Narrowing this to
    ``Q(source_model=<this model>)`` alone therefore left an outline the user
    had drawn but not yet confirmed with nowhere to be painted -- filtered out
    of the named bundle, and hidden in the source-less one by the confirmed-only
    LUT -- so it disappeared from the screen as soon as any model bundle
    existed.  That is precisely what owner ruling R13 forbids ("a run must not
    overwrite or hide what the user annotated"), and
    :mod:`~quantem.segmentation.tests.test_annotation_preservation_invariant`
    pins it.

    Membership is state-independent on purpose: a hand-drawn object belongs to
    the named bundle whether it is CANDIDATE or CONFIRMED, so an ordinary
    Confirm still moves no pixels between bundles and stays a LUT-only update.

    What is *not* in here is another model's CONFIRMED objects.  Those already
    have a home in the confirmed display, and admitting them into every model
    layer was what made one confirmation dirty every bundle.  Hand-drawn objects
    do not reintroduce that cost: there are a handful of them per image, and
    they are created by a deliberate gesture rather than by a run.

    This is *bundle membership*, deliberately narrower than
    :func:`source_model_queryset_filter`.  The two were briefly the same
    function; collapsing them made the segment click/hover/region/ROI endpoints
    stop returning hand-drawn and other-model confirmed objects, so those
    objects stayed painted by the confirmed display while silently becoming
    un-hoverable and un-deletable.  Keep them separate.
    """
    normalized = normalize_source_model(source_model)
    if not normalized:
        return None
    return Q(source_model=normalized) | Q(source_model=SOURCE_MODEL_MANUAL)


def source_model_queryset_filter(source_model: str | None) -> Q | None:
    """Objects the user can see and act on under one model selection.

    The model picker chooses whose *candidates* are under review, not which
    objects exist.  Everything already accepted into the segmentation stays
    addressable regardless of which model produced it, and hand-drawn objects
    are never hidden by a model selection -- the user drew them, so they must
    remain hoverable, un-markable, rejectable and deletable under every picker
    value.  Only another model's still-unreviewed candidates are filtered out.

    This is the *user selection* rule for the segment point/region/ROI query
    endpoints and inferred-segment selection.  For raster bundle membership use
    :func:`overlay_bundle_source_filter` instead.
    """
    normalized = normalize_source_model(source_model)
    if not normalized:
        return None
    return (
        Q(label_state="CONFIRMED")
        | Q(source_model=SOURCE_MODEL_MANUAL)
        | Q(source_model=normalized)
    )


def source_model_payload(definition: SourceModelDefinition, *, count: int = 0) -> dict[str, object]:
    return {
        "value": definition.value,
        "label": definition.label,
        "model_family": definition.model_family,
        "variant": definition.variant,
        "is_default": definition.is_default,
        "count": int(count),
    }


def unique_source_models(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_value in values:
        value = normalize_source_model(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "MODEL_FAMILY_OMNIEM",
    "MODEL_FAMILY_QUANTEM",
    "SOURCE_MODEL_DEFINITIONS",
    "SOURCE_MODEL_MANUAL",
    "SOURCE_MODEL_UNKNOWN",
    "SourceModelDefinition",
    "default_source_model_for_organelle",
    "get_source_model_definition",
    "infer_source_model_from_features",
    "normalize_source_model",
    "overlay_bundle_source_filter",
    "resolve_create_segmentation_request",
    "resolve_segmenter_internal_name",
    "source_model_payload",
    "source_model_queryset_filter",
    "source_models_for_organelle",
    "unique_source_models",
]
