"""Retained predictions and the result currently used for measurement."""

import logging
import shutil
from copy import copy, deepcopy

from django.db import transaction
from django.utils import timezone

from quantem import __version__
from quantem.core.local_storage import StorageError
from quantem.segmentation.coverage import completed_geometry
from quantem.segmentation.geometry import extract_polygons
from quantem.segmentation.models import (
    ImageSegmentation,
    SavedSegmentationResult,
    SegmentObject,
    SegmentObjectAnnotation,
)

logger = logging.getLogger(__name__)


def _replace_geometry(row, geometry, *, label_state):
    """Keep every polygon component, including interior holes, after clipping."""
    row.superseded_at = timezone.now()
    row.save(update_fields=["superseded_at"])
    annotation = SegmentObjectAnnotation.objects.filter(object=row).first()
    for polygon in extract_polygons(geometry):
        if polygon.is_empty or polygon.area <= 0:
            continue
        part = copy(row)
        part.pk = None
        part._state = copy(row._state)
        part._state.fields_cache = dict(row._state.fields_cache)
        part._state.fields_cache.pop("annotation", None)
        part._state.adding = True
        part.superseded_at = None
        part.geometry = polygon
        part.centroid = polygon.centroid
        part.bbox = polygon.envelope
        part.label_state = label_state
        # ``status`` is the compatibility mirror still read by older clients
        # and some exports. A clipped candidate promoted to confirmed must not
        # retain its old numeric candidate state and render red after reload.
        if label_state == "CONFIRMED":
            part.status = SegmentObject.STATUS_CONFIRMED
        part.features = {}  # clipped measurements must be recomputed
        part.base_segment = None
        part.save()
        if annotation is not None:
            SegmentObjectAnnotation.objects.create(
                object=part,
                display_name=annotation.display_name,
                tags=deepcopy(annotation.tags),
                metadata=deepcopy(annotation.metadata),
            )


def retain_completed_annotations(segmentation):
    """Replace the model-derived area while preserving completed manual areas."""
    protected = completed_geometry(segmentation)
    rows = SegmentObject.objects.filter(segmentation=segmentation, superseded_at=None)
    if protected.is_empty:
        rows.update(superseded_at=timezone.now())
        return
    for row in rows.iterator(chunk_size=500):
        kept = row.geometry.intersection(protected)
        if kept.equals(row.geometry):
            # A completed area's objects and empty spaces are authoritative.
            if row.label_state in {"INFERRED", "CANDIDATE"}:
                row.label_state = "CONFIRMED"
                row.status = SegmentObject.STATUS_CONFIRMED
                row.save(update_fields=["label_state", "status"])
        else:
            _replace_geometry(row, kept, label_state=row.label_state)


def accept_model_area(segmentation):
    protected = completed_geometry(segmentation)
    candidates = SegmentObject.objects.filter(
        segmentation=segmentation, superseded_at=None, label_state__in=["CANDIDATE", "INFERRED"]
    )
    if protected.is_empty:
        candidates.update(label_state="CONFIRMED", status=SegmentObject.STATUS_CONFIRMED)
        return
    for row in candidates.iterator(chunk_size=500):
        allowed = row.geometry.difference(protected)
        if allowed.equals(row.geometry):
            row.label_state = "CONFIRMED"
            row.status = SegmentObject.STATUS_CONFIRMED
            row.save(update_fields=["label_state", "status"])
        else:
            _replace_geometry(row, allowed, label_state="CONFIRMED")


@transaction.atomic
def publish_result(
    segmentation, probability_map, threshold, source_model, *, selected=None, delete_ids=()
):
    """Publish coverage only after extraction succeeds; deletions are explicit."""
    ImageSegmentation.objects.select_for_update().get(pk=segmentation.pk)
    from quantem.analysis.superseded import mark_superseded

    mark_superseded(segmentation)
    metadata = dict(probability_map.metadata or {})
    if metadata.get("roi_id") or metadata.get("composite"):
        raise ValueError("Only a full-image prediction can be used as a full-image result.")
    if selected is not None and selected.deleted_at is not None:
        raise ValueError("The selected result was deleted.")
    segmentation.saved_results.filter(is_canonical=True).update(is_canonical=False)
    if selected is None:
        adapter_id = str(metadata.get("adapter_id") or "")
        from django.core.exceptions import ValidationError

        from quantem.finetune.models import Adapter

        try:
            adapter_name = (
                Adapter.objects.filter(pk=adapter_id).values_list("name", flat=True).first()
                if adapter_id
                else None
            )
        except ValidationError:
            adapter_name = None
        selected = SavedSegmentationResult.objects.create(
            segmentation=segmentation,
            name=f"{adapter_name or source_model or probability_map.name} · threshold {threshold:g}",
            source_model=source_model or "",
            adapter_id=adapter_id,
            threshold=threshold,
            probability_map=probability_map,
            provenance=metadata,
            full_image=True,
            object_count=SegmentObject.objects.filter(
                segmentation=segmentation, superseded_at=None, label_state="CONFIRMED"
            ).count(),
        )
    selected.is_canonical = True
    selected.save(update_fields=["is_canonical", "updated_at"])
    deleted_results = segmentation.saved_results.filter(id__in=delete_ids).exclude(pk=selected.pk)
    deleted_result_ids = tuple(deleted_results.values_list("pk", flat=True))
    deleted_results.update(
        deleted_at=timezone.now(),
        is_canonical=False,
    )
    if deleted_result_ids:
        transaction.on_commit(
            lambda: cleanup_deleted_results(segmentation, result_ids=deleted_result_ids)
        )
    segmentation.include_level = threshold
    segmentation.status_stage = "CANDIDATES_READY"
    segmentation.status_progress = 100
    segmentation.status_error = ""
    segmentation.final_result_provenance = {
        **metadata,
        "quantem_version": __version__,
        "finalized_at": timezone.now().isoformat(),
        "result_id": str(selected.pk),
        "model_identifier": source_model,
        "adapter_identifier": selected.adapter_id or "default",
        "final_level": threshold,
        "final_level_kind": "include_level",
    }
    segmentation.save(
        update_fields=[
            "include_level",
            "manual_full_image",
            "status_stage",
            "status_progress",
            "status_error",
            "final_result_provenance",
            "updated_at",
        ]
    )
    return selected


def cleanup_deleted_results(segmentation, *, result_ids):
    """Reclaim derived comparison data, then any now-unreferenced source maps."""
    from quantem.segmentation.prob_maps.io import resolve_probability_map_path
    from quantem.segmentation.prob_maps.preview import result_preview_cache_path

    results = segmentation.saved_results.filter(pk__in=result_ids).select_related("probability_map")
    for result in results:
        if result.probability_map is None:
            continue
        try:
            cache = result_preview_cache_path(resolve_probability_map_path(result.probability_map))
            shutil.rmtree(cache / str(result.pk), ignore_errors=True)
            for preview in cache.glob(f"{result.pk}.*.png"):
                preview.unlink(missing_ok=True)
        except (OSError, StorageError, ValueError):
            logger.warning(
                "Could not remove derived caches for saved result %s.",
                result.pk,
                exc_info=True,
            )
    prune_deleted_maps(segmentation)


def prune_deleted_maps(segmentation):
    """Reclaim a deleted prediction only when no retained version shares it."""
    from quantem.seg_core.db.prob_maps import _unlink_unshared_probability_map_file
    from quantem.segmentation.models import ProbabilityMap

    candidates = ProbabilityMap.objects.filter(
        segmentation=segmentation, saved_results__deleted_at__isnull=False
    ).distinct()
    for probability_map in candidates:
        if probability_map.saved_results.filter(deleted_at=None).exists():
            continue
        probability_map.delete()
        _unlink_unshared_probability_map_file(probability_map)
