"""Shared input dependencies and whole-image coverage requirements."""

from quantem.segmentation.coverage import coverage_state
from quantem.segmentation.models import ImageSegmentation


def dependency_ids(segmentation_id: object, params: dict) -> set[str]:
    """Include area sources retained by definition consolidation."""
    ids = {str(segmentation_id)} if segmentation_id else set()
    ids.update(str(value) for value in params.get("compartments", {}).values())
    ids.update(str(value) for value in params.get("analysis_mask_ids", []))
    for key in ("tissue_segmentation_id", "cell_segmentation_id"):
        if params.get(key):
            ids.add(str(params[key]))
    pending = ids.copy()
    while pending:
        sources = {
            str(value)
            for value in ImageSegmentation.objects.filter(superseded_by_id__in=pending).values_list(
                "id", flat=True
            )
        } - ids
        ids.update(sources)
        pending = sources
    return ids


def require_complete_organelles(segmentation: ImageSegmentation, params: dict) -> None:
    """Reject partial organelles even when a smaller measurement area is selected.

    Area definitions describe a measurement region, not exhaustive organelle
    annotation. Their availability is validated separately.
    """
    from quantem.analysis.loaders import AnalysisInputError
    from quantem.jobs.constants import ACTIVE_SEGMENTATION_JOB_TYPES
    from quantem.segmentation.api_views.shared import active_segmentation_job

    if segmentation is None:
        raise AnalysisInputError("This segmentation is no longer available.")
    ids = {str(segmentation.id), *map(str, params.get("compartments", {}).values())}
    for item in ImageSegmentation.objects.filter(id__in=ids).select_related(
        "segmentation_type", "asset"
    ):
        if item.segmentation_type.category != "organelle":
            continue
        name = item.display_name or item.segmentation_type.long_name
        if item.superseded_by_id or not coverage_state(item)["ready"]:
            raise AnalysisInputError(
                f"{name}: the whole image has not been segmented. Run a full-image model "
                "or finish manual annotation of the whole image before measuring."
            )
        if active_segmentation_job(item, job_types=ACTIVE_SEGMENTATION_JOB_TYPES):
            raise AnalysisInputError(f"{name}: a segmentation task is queued or running.")


def has_saved_area(segmentation: ImageSegmentation) -> bool:
    if segmentation.analysis_mask_objects.exclude(geometry_wkb__isnull=True).exists():
        return True
    if ImageSegmentation.objects.filter(
        pk=segmentation.pk, area_vectors_authoritative=True
    ).exists():
        return False
    from quantem.segmentation.models import GlobalMask

    return GlobalMask.objects.filter(
        segmentation_id__in=dependency_ids(segmentation.id, {}), foreground_pixels__gt=0
    ).exists()
