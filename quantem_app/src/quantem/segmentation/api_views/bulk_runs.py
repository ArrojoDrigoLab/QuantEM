"""Experiment-wide application with durable, per-image queue outcomes."""

import math
from uuid import UUID

from django.db import transaction
from django.shortcuts import get_object_or_404
from django.urls import path
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.finetune.models import Adapter
from quantem.jobs.constants import JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE, QUEUE_P4_FULL
from quantem.jobs.models import Job
from quantem.library.models import Experiment
from quantem.segmentation.api_views.runs import ORGANELLE_TYPES, _build_plan
from quantem.segmentation.api_views.shared import active_segmentation_job
from quantem.segmentation.models import ImageSegmentation, ModelApplicationBatch, SegmentationConfig
from quantem.segmentation.source_models import resolve_create_segmentation_request
from quantem.segmentation.type_service import ensure_segmentation_type


def plan_application(experiment, data):
    try:
        raw_ids = data.get("asset_ids", [])
        if not isinstance(raw_ids, list) or not 1 <= len(raw_ids) <= 5000:
            raise ValueError
        ids = {UUID(str(value)) for value in raw_ids}
    except (TypeError, ValueError):
        raise ValidationError("Select between 1 and 5,000 images in this experiment.") from None
    assets = list(
        experiment.assets.filter(id__in=ids, lifecycle_status="ACTIVE").order_by(
            "display_name", "id"
        )
    )
    if len(assets) != len(ids):
        raise ValidationError("Every selected image must belong to this experiment.")
    organelles = data.get("organelles", [])
    if (
        not isinstance(organelles, list)
        or not organelles
        or any(key not in ORGANELLE_TYPES for key in organelles)
    ):
        raise ValidationError("Choose at least one available organelle.")
    organelles = list(dict.fromkeys(organelles))
    family = data.get("family", "quantem")
    if family not in {"quantem", "omniem"}:
        raise ValidationError("Choose QuantEM or OmniEM.")
    adapter = None
    if data.get("adapter_id"):
        try:
            adapter_id = UUID(str(data["adapter_id"]))
        except (TypeError, ValueError):
            raise ValidationError("Choose a saved model in this experiment.") from None
        adapter = get_object_or_404(
            Adapter, pk=adapter_id, experiment=experiment, status="SUCCESS", deleted_at=None
        )
        organelle = adapter.base_model.split(":")[-1]
        if organelles != [organelle]:
            raise ValidationError("Choose the saved model's organelle.")
    requested_sources = data.get("source_models") or {}
    if not isinstance(requested_sources, dict):
        raise ValidationError("Model selections must name a model for each organelle.")
    sources = {
        key: adapter.base_model if adapter else requested_sources.get(key, f"{family}:{key}")
        for key in organelles
    }
    if any(source not in {f"quantem:{key}", f"omniem:{key}"} for key, source in sources.items()):
        raise ValidationError("Choose an available model for every selected organelle.")
    threshold = data.get("threshold")
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValidationError(
            "Threshold must be a number between 0 and 1, or blank for the model default."
        )
    rows = []
    for asset in assets:
        reason = ""
        if asset.preprocess_stage not in {"DONE", "SKIPPED"}:
            reason = "Image preparation is not finished."
        try:
            plan = _build_plan(asset, organelles, sources)
        except (OSError, ValueError):
            plan = {"organelles": [], "tiles_total": None}
            reason = "The local image could not be read. Check its source and preparation."
        if plan.get("tiles_total") is None:
            reason = reason or "The local image is unavailable or has no usable dimensions."
        selected = [entry for entry in plan["organelles"] if entry["organelle"] in organelles]
        for entry in selected:
            if entry.get("model_installed") and entry.get("model_blocked_reason"):
                reason = entry["model_blocked_reason"]
        segs = ImageSegmentation.objects.filter(
            asset=asset,
            segmentation_type__internal_name__in=[
                ORGANELLE_TYPES[key].internal_name for key in organelles
            ],
        )
        if any(active_segmentation_job(seg) for seg in segs):
            reason = "A segmentation task is already queued or running."
        rows.append(
            {
                "asset_id": str(asset.id),
                "image": asset.display_name,
                "ready": not reason,
                "reason": reason,
                "tiles": plan.get("tiles_total"),
                "warnings": (
                    ["Pixel size is unknown; inference will use native resolution."]
                    if not asset.pixel_size_nm
                    else []
                ),
                "download_bytes": plan.get("download_bytes_total", 0),
                "models": sources,
                "adapter_id": str(adapter.id) if adapter else "default",
                "threshold": threshold,
            }
        )
    return rows


def batch_body(batch):
    jobs = {str(job.id): job for job in Job.objects.filter(batch_id=f"model-batch:{batch.id}")}
    rows = []
    for item in batch.items:
        job = jobs.get(item.get("job_id"))
        rows.append(
            {
                **item,
                "status": job.status if job else "SKIPPED",
                "progress": job.progress if job else None,
                "message": job.message if job else item.get("reason", ""),
            }
        )
    return {
        "id": str(batch.id),
        "created_at": batch.created_at,
        "settings": batch.settings,
        "items": rows,
    }


class ExperimentModelRunsView(APIView):
    def get(self, request, experiment_id):
        experiment = get_object_or_404(Experiment, pk=experiment_id)
        return Response(
            [
                batch_body(batch)
                for batch in ModelApplicationBatch.objects.filter(experiment=experiment).order_by(
                    "-created_at"
                )[:25]
            ]
        )

    def post(self, request, experiment_id):
        experiment = get_object_or_404(Experiment, pk=experiment_id)
        rows = plan_application(experiment, request.data)
        if request.data.get("preview") is True:
            return Response({"items": rows})
        if not any(row["ready"] for row in rows):
            raise ValidationError("No selected image is ready. Check the per-image reasons.")
        with transaction.atomic():
            # Serializes submissions from this experiment, including a repeated click.
            Experiment.objects.select_for_update().get(pk=experiment.pk)
            rows = plan_application(experiment, request.data)
            if not any(row["ready"] for row in rows):
                raise ValidationError(
                    "The selected images now have active tasks. Refresh the preview."
                )
            batch = ModelApplicationBatch.objects.create(
                experiment=experiment, settings=dict(request.data)
            )
            for row in rows:
                if not row["ready"]:
                    continue
                legs = []
                for key, source in row["models"].items():
                    definition, source = resolve_create_segmentation_request(
                        ORGANELLE_TYPES[key], source
                    )
                    seg_type = ensure_segmentation_type(definition)
                    seg, _ = ImageSegmentation.objects.get_or_create(
                        asset_id=row["asset_id"], segmentation_type=seg_type
                    )
                    SegmentationConfig.objects.get_or_create(segmentation=seg)
                    legs.append(
                        {
                            "segmentation_id": str(seg.id),
                            "segmentation_type": seg_type.internal_name,
                            "source_model": source,
                            "adapter_id": row["adapter_id"],
                            "threshold": row["threshold"],
                        }
                    )
                job = Job.enqueue(
                    job_type=JOB_TYPE_RUN_SEGMENTATION_FOR_IMAGE,
                    payload={
                        "asset_id": row["asset_id"],
                        "scope": "full",
                        "legs": legs,
                        "force_recompute_prob_maps": True,
                    },
                    resource_class="gpu",
                    queue_name=QUEUE_P4_FULL,
                    max_attempts=1,
                    batch_id=f"model-batch:{batch.id}",
                    tags=[f"segmentation:{leg['segmentation_id']}" for leg in legs],
                )
                row["job_id"] = str(job.id)
            batch.items = rows
            batch.save(update_fields=["items"])
        return Response(batch_body(batch), status=202)


class ModelRunDetailView(APIView):
    def get(self, request, batch_id):
        return Response(batch_body(get_object_or_404(ModelApplicationBatch, pk=batch_id)))


urlpatterns = [
    path("experiments/<uuid:experiment_id>/model-runs/", ExperimentModelRunsView.as_view()),
    path("model-runs/<uuid:batch_id>/", ModelRunDetailView.as_view()),
]
