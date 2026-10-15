"""Bounded history summaries and database-level image counts."""

import uuid

from django.db.models import CharField, Count, Prefetch, Q, Value
from django.db.models.fields.json import KeyTextTransform, KeyTransform
from django.db.models.functions import Cast, Coalesce, NullIf, Replace
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.analysis.models import AnalysisBatch, AnalysisBatchItem, AnalysisRun
from quantem.library.models import Experiment

PAGE_SIZE = 25
MAX_PAGE_SIZE = 100


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


def _key(expression):
    # SQLite stores UUID columns without hyphens; JSON snapshots use strings.
    return Replace(Cast(expression, CharField()), Value("-"), Value(""))


def _json_text(key, field):
    # Cast also uses ordinary text lookups instead of JSON-value exact lookups.
    return NullIf(NullIf(Cast(KeyTextTransform(key, field), CharField()), Value("null")), Value(""))


def _image_id(value):
    return str(uuid.UUID(value)) if value else None


def standalone_runs(experiment):
    """Prefer saved experiment identity; fall back to live membership for old runs."""
    assets = experiment.assets.order_by().annotate(key=_key("id")).values("key")
    batch_runs = AnalysisBatchItem.objects.filter(reused=False, run__isnull=False).values("run_id")
    runs = AnalysisRun.objects.annotate(
        history_experiment=_json_text("experiment_id", "input_identity"),
        history_asset=Coalesce(
            _json_text("asset_id", "input_identity"),
            _json_text("image_key", "results"),
            Cast("segmentation__asset_id", CharField()),
            output_field=CharField(),
        ),
    ).annotate(history_asset_key=_key("history_asset"))
    unrecorded = (
        Q(history_experiment__isnull=True) | Q(history_experiment="") | Q(history_experiment="null")
    )
    return runs.filter(
        Q(history_experiment=str(experiment.id))
        | (
            unrecorded
            & (Q(segmentation__asset__experiment=experiment) | Q(history_asset_key__in=assets))
        )
    ).exclude(id__in=batch_runs)


def _successful_items():
    return AnalysisBatchItem.objects.filter(run__status="SUCCESS").exclude(
        job__status__in=("FAILED", "CANCELLED")
    )


def analysis_counts(experiment) -> dict[str, int]:
    """Count an event once per image without fetching its results or job payload."""
    counts = {}
    single = (
        standalone_runs(experiment)
        .filter(status="SUCCESS")
        .order_by()
        .values("history_asset_key")
        .annotate(total=Count("id"))
    )
    for row in single:
        asset_id = _image_id(row["history_asset_key"])
        if asset_id:
            counts[asset_id] = row["total"]
    batches = (
        _successful_items()
        .filter(batch__experiment=experiment)
        .annotate(
            asset_key=_key(KeyTextTransform("asset_id", "identity")),
        )
        .order_by()
        .values("asset_key")
        .annotate(total=Count("batch_id", distinct=True))
    )
    for row in batches:
        asset_id = _image_id(row["asset_key"])
        if asset_id:
            counts[asset_id] = counts.get(asset_id, 0) + row["total"]
    return counts


def _batch_entry(batch):
    images, structures, completed, statuses = {}, {}, set(), []
    for item in batch.items.all():
        identity = item.identity or {}
        asset_id = identity.get("asset_id")
        if asset_id:
            images[asset_id] = identity.get("image") or "Deleted image"
        internal = identity.get("organelle", "")
        seg_type = item.segmentation.segmentation_type if item.segmentation else None
        if internal:
            structures[internal] = identity.get("structure") or (
                seg_type.short_name or seg_type.long_name if seg_type else internal
            )
        state = item.run.status if item.run else "SKIPPED"
        if item.job and item.job.status in ("FAILED", "CANCELLED"):
            state = item.job.status
        statuses.append(state)
        if state == "SUCCESS" and asset_id:
            completed.add(asset_id)
    return {
        "id": str(batch.id),
        "kind": "batch",
        "name": batch.name,
        "created_at": batch.created_at,
        "status": _state(statuses),
        "images": [{"id": key, "name": value} for key, value in images.items()],
        "structures": list(structures.values()),
        "measurement_count": len(statuses),
        "measurement_area": batch.history_mask or "Whole image",
        "completed_image_ids": sorted(completed),
    }


def _run_entry(run):
    identity = run.input_identity or {}
    seg = run.segmentation
    asset_id = _image_id(run.history_asset)
    name = identity.get("image") or (seg.asset.display_name if seg else "Deleted image")
    structure = identity.get("structure") or (
        (seg.segmentation_type.short_name or seg.segmentation_type.long_name)
        if seg
        else identity.get("organelle", "Deleted segmentation")
    )
    return {
        "id": str(run.id),
        "kind": "run",
        "name": f"{name} · {structure}",
        "created_at": run.created_at,
        "status": run.status,
        "images": [{"id": asset_id, "name": name}] if asset_id else [],
        "structures": [structure],
        "measurement_count": 1,
        "measurement_area": identity.get("measurement_area")
        or ("Selected mask" if run.history_mask or run.history_masks else "Whole image"),
        "completed_image_ids": [asset_id] if asset_id and run.status == "SUCCESS" else [],
    }


def history_page(experiment, *, offset=0, limit=PAGE_SIZE, asset_id=None) -> dict:
    batches = AnalysisBatch.objects.filter(experiment=experiment)
    runs = standalone_runs(experiment)
    if asset_id:
        try:
            asset_id = str(uuid.UUID(asset_id))
        except ValueError as exc:
            raise ValidationError("Choose a valid image.") from exc
        batches = batches.filter(
            id__in=_successful_items().filter(identity__asset_id=asset_id).values("batch_id")
        )
        runs = runs.filter(history_asset_key=asset_id.replace("-", ""), status="SUCCESS")
    # Only IDs and timestamps cross the union. Slice before fetching identities,
    # and never materialize full results or job payloads to display a list.
    index = (
        batches.order_by()
        .values("id", "created_at")
        .annotate(kind=Value("batch", output_field=CharField()))
        .union(
            runs.order_by()
            .values("id", "created_at")
            .annotate(kind=Value("run", output_field=CharField()))
        )
        .order_by("-created_at", "-id", "kind")
    )
    total = index.count()
    selected = list(index[offset : offset + limit])
    items = AnalysisBatchItem.objects.select_related(
        "run", "job", "segmentation__segmentation_type"
    ).only(
        "id",
        "batch_id",
        "identity",
        "run_id",
        "run__status",
        "job_id",
        "job__status",
        "segmentation_id",
        "segmentation__segmentation_type_id",
        "segmentation__segmentation_type__short_name",
        "segmentation__segmentation_type__long_name",
    )
    batch_rows = (
        batches.filter(id__in=[row["id"] for row in selected if row["kind"] == "batch"])
        .annotate(
            history_mask=_json_text("mask_name", "settings"),
        )
        .only("id", "name", "created_at")
        .prefetch_related(Prefetch("items", queryset=items))
    )
    run_rows = (
        runs.filter(id__in=[row["id"] for row in selected if row["kind"] == "run"])
        .annotate(
            history_mask=_json_text("tissue_segmentation_id", "params"),
            history_masks=KeyTransform("analysis_mask_ids", "params"),
        )
        .select_related("segmentation__segmentation_type", "segmentation__asset")
        .only(
            "id",
            "status",
            "created_at",
            "input_identity",
            "segmentation_id",
            "segmentation__asset_id",
            "segmentation__asset__display_name",
            "segmentation__segmentation_type_id",
            "segmentation__segmentation_type__short_name",
            "segmentation__segmentation_type__long_name",
        )
    )
    entries = {("batch", str(batch.id)): _batch_entry(batch) for batch in batch_rows}
    entries.update({("run", str(run.id)): _run_entry(run) for run in run_rows})
    return {
        "results": [entries[(row["kind"], str(row["id"]))] for row in selected],
        "count": total,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + limit if offset + limit < total else None,
    }


def experiment_history(experiment) -> list[dict]:
    """The first page of history; use history_page to traverse older events."""
    return history_page(experiment)["results"]


class ExperimentAnalysisHistoryView(APIView):
    def get(self, request, experiment_id):
        experiment = get_object_or_404(Experiment, pk=experiment_id)
        try:
            offset = int(request.query_params.get("offset", 0))
            limit = int(request.query_params.get("limit", PAGE_SIZE))
        except ValueError as exc:
            raise ValidationError("History offset and limit must be integers.") from exc
        if offset < 0 or not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValidationError(
                f"Choose a nonnegative offset and a limit from 1 to {MAX_PAGE_SIZE}."
            )
        return Response(
            history_page(
                experiment,
                offset=offset,
                limit=limit,
                asset_id=request.query_params.get("asset_id"),
            )
        )
