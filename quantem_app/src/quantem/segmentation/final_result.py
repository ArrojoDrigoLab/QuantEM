"""Durable provenance for a completed visible segmentation result."""

from __future__ import annotations

from django.utils import timezone

from quantem import __version__
from quantem.segmentation.models import GlobalMask, ImageSegmentation, ProbabilityMap


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _provenance_probability_map(
    segmentation: ImageSegmentation,
) -> ProbabilityMap | None:
    """Newest canonical/ROI map, never a lossy composite when one exists.

    ROI composites deliberately keep only enough metadata to render the canvas;
    they omit the pack and adapter identity carried by the canonical ROI map.
    Choosing only by ``updated_at`` therefore makes the final note depend on
    which row happened to be saved last and can silently erase provenance.
    """
    fallback = None
    for probability_map in ProbabilityMap.objects.filter(segmentation=segmentation).order_by(
        "-updated_at", "-created_at"
    ):
        if fallback is None:
            fallback = probability_map
        metadata = probability_map.metadata
        if not isinstance(metadata, dict) or metadata.get("composite") is not True:
            return probability_map
    return fallback


def persist_final_result_provenance(segmentation: ImageSegmentation) -> dict:
    """Write the completion note once and return the immutable stored value.

    The conditional update is important: retries cannot rewrite the statement
    about the result that is currently marked final. Explicitly unlocking the
    segmentation clears the note because the result is no longer final; a
    later completion then records the new model and threshold.
    """
    current = segmentation.final_result_provenance
    if isinstance(current, dict) and current:
        return current

    probability_map = _provenance_probability_map(segmentation)
    metadata = probability_map.metadata if probability_map is not None else {}
    if not isinstance(metadata, dict):
        metadata = {}
    global_mask = GlobalMask.objects.filter(segmentation=segmentation).first()
    global_metadata = global_mask.metadata if global_mask is not None else {}
    if not isinstance(global_metadata, dict):
        global_metadata = {}
    if probability_map is None:
        metadata = global_metadata

    model_identifier = (
        _text(metadata.get("pack_id"))
        or _text(metadata.get("source_model"))
        or _text(metadata.get("model_id"))
        or _text(metadata.get("model_name"))
        or _text(metadata.get("model_type"))
        or (_text(probability_map.name) if probability_map is not None else "")
        or ("manual" if global_mask is not None and global_mask.source.startswith("manual") else "")
        or "unknown"
    )
    adapter_identifier = _text(metadata.get("adapter_id"))
    if not adapter_identifier:
        adapter_identifier = "manual" if model_identifier == "manual" else "unknown"

    level = segmentation.include_level
    level_kind = "include_level"
    if level is None and metadata.get("threshold") is not None:
        try:
            level = float(metadata["threshold"])
            level_kind = "model_threshold"
        except (TypeError, ValueError):
            level = None
    if level is None:
        level_kind = "manual" if model_identifier == "manual" else "unknown"

    note = {
        "model_identifier": model_identifier,
        "quantem_version": __version__,
        "final_level": level,
        "final_level_kind": level_kind,
        "adapter_identifier": adapter_identifier,
        "finalized_at": timezone.now().isoformat(),
    }
    written = ImageSegmentation.objects.filter(
        id=segmentation.id,
        final_result_provenance={},
    ).update(final_result_provenance=note)
    if written:
        segmentation.final_result_provenance = note
        return note
    stored = (
        ImageSegmentation.objects.filter(id=segmentation.id)
        .values_list("final_result_provenance", flat=True)
        .first()
    )
    segmentation.final_result_provenance = stored or note
    return segmentation.final_result_provenance
