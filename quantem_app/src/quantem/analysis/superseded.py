"""Flag saved measurements without rewriting their numbers or export files."""

from django.utils import timezone

from quantem.analysis.input_contract import dependency_ids
from quantem.analysis.models import AnalysisRun


def mark_asset_superseded(asset):
    """Calibration and biological grouping affect every analysis on an image."""
    AnalysisRun.objects.filter(
        segmentation__asset=asset, status="SUCCESS", superseded_at=None
    ).update(superseded_at=timezone.now())


def mark_superseded(segmentation):
    candidates = AnalysisRun.objects.filter(
        segmentation__asset_id=segmentation.asset_id,
        status="SUCCESS",
        superseded_at=None,
    ).values("id", "segmentation_id", "params")
    changed = str(segmentation.id)
    ids = []
    for run in candidates:
        params = run["params"] or {}
        dependencies = dependency_ids(run["segmentation_id"], params)
        if changed in dependencies:
            ids.append(run["id"])
    if ids:
        AnalysisRun.objects.filter(id__in=ids, superseded_at=None).update(
            superseded_at=timezone.now()
        )
