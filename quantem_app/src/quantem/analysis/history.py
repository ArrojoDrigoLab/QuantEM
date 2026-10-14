"""A shared history for experiment batches and individual image analyses."""

from django.db.models import Prefetch, Q
from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.analysis.models import AnalysisBatch, AnalysisBatchItem, AnalysisRun
from quantem.library.models import Experiment


def _state(statuses):
    if any(s in ("RUNNING", "PENDING") for s in statuses):
        return "RUNNING" if "RUNNING" in statuses else "PENDING"
    if statuses and all(s == "SUCCESS" for s in statuses):
        return "SUCCESS"
    if "SUCCESS" in statuses:
        return "PARTIAL"
    return (
        "FAILED" if "FAILED" in statuses else "CANCELLED" if "CANCELLED" in statuses else "SKIPPED"
    )


def experiment_history(experiment):
    """List without reading object geometry, checking fingerprints or opening exports."""
    assets = {str(a.id): a.display_name for a in experiment.assets.all().only("id", "display_name")}
    items = AnalysisBatchItem.objects.select_related(
        "run", "job", "segmentation__segmentation_type"
    )
    batches = AnalysisBatch.objects.filter(experiment=experiment).prefetch_related(
        Prefetch("items", queryset=items)
    )
    entries = []
    for batch in batches:
        images, structures, completed, statuses = {}, {}, set(), []
        for item in batch.items.all():
            identity = item.identity or {}
            asset_id = identity.get("asset_id")
            if asset_id:
                images[asset_id] = identity.get("image") or assets.get(asset_id, "Deleted image")
            internal = identity.get("organelle", "")
            seg_type = item.segmentation.segmentation_type if item.segmentation else None
            if internal:
                structures[internal] = (
                    seg_type.short_name or seg_type.long_name if seg_type else internal
                )
            state = item.run.status if item.run else "SKIPPED"
            if item.job and item.job.status in ("FAILED", "CANCELLED"):
                state = item.job.status
            statuses.append(state)
            if state == "SUCCESS" and asset_id:
                completed.add(asset_id)
        entries.append(
            {
                "id": str(batch.id),
                "kind": "batch",
                "name": batch.name,
                "created_at": batch.created_at,
                "status": _state(statuses),
                "images": [{"id": key, "name": value} for key, value in images.items()],
                "structures": list(structures.values()),
                "measurement_count": len(statuses),
                "measurement_area": batch.settings.get("mask_name") or "Whole image",
                "completed_image_ids": sorted(completed),
            }
        )
    # Reused standalone runs remain their own historical event. Runs first
    # created inside a batch appear only in that batch, not once per organelle.
    batch_run_ids = AnalysisBatchItem.objects.filter(reused=False, run__isnull=False).values(
        "run_id"
    )
    runs = (
        AnalysisRun.objects.filter(
            Q(segmentation__asset__experiment=experiment)
            | Q(input_identity__experiment_id=str(experiment.id))
            | Q(input_identity__asset_id__in=list(assets))
            | Q(results__image_key__in=list(assets))
        )
        .exclude(id__in=batch_run_ids)
        .select_related("segmentation__segmentation_type", "segmentation__asset")
    )
    for run in runs:
        identity = run.input_identity or {}
        # Prefer the original experiment when it was recorded, even after a move.
        if identity.get("experiment_id") and identity["experiment_id"] != str(experiment.id):
            continue
        seg = run.segmentation
        asset_id = (
            identity.get("asset_id")
            or (run.results or {}).get("image_key")
            or (str(seg.asset_id) if seg else None)
        )
        name = (
            identity.get("image")
            or assets.get(asset_id)
            or (seg.asset.display_name if seg and seg.asset else "Deleted image")
        )
        structure = (
            (seg.segmentation_type.short_name or seg.segmentation_type.long_name)
            if seg
            else identity.get("organelle", "Deleted segmentation")
        )
        entries.append(
            {
                "id": str(run.id),
                "kind": "run",
                "name": f"{name} · {structure}",
                "created_at": run.created_at,
                "status": run.status,
                "images": [{"id": asset_id, "name": name}] if asset_id else [],
                "structures": [structure],
                "measurement_count": 1,
                "measurement_area": identity.get("measurement_area")
                or (
                    "Selected mask"
                    if run.params.get("tissue_segmentation_id")
                    or run.params.get("analysis_mask_ids")
                    else "Whole image"
                ),
                "completed_image_ids": [asset_id] if asset_id and run.status == "SUCCESS" else [],
            }
        )
    return sorted(entries, key=lambda row: (row["created_at"], row["id"]), reverse=True)


class ExperimentAnalysisHistoryView(APIView):
    def get(self, request, experiment_id):
        experiment = get_object_or_404(Experiment, pk=experiment_id)
        entries = experiment_history(experiment)
        asset_id = request.query_params.get("asset_id")
        if asset_id:
            entries = [row for row in entries if asset_id in row["completed_image_ids"]]
        return Response(entries)
