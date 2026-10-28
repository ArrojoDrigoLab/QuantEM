"""Standalone evaluation of released or saved models on completed annotations."""

import hashlib
import math
from uuid import UUID

import numpy as np
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.finetune.models import Adapter
from quantem.finetune.scope import resolve_scope
from quantem.inference.specs import MODEL_SPECS
from quantem.jobs.constants import JOB_DEFAULTS, JOB_TYPE_EVALUATE_MODEL
from quantem.jobs.models import Job
from quantem.segmentation.models import SegmentationType
from quantem.segmentation.services.adapt.extract_crops import collect_crops_for_scope


def crop_fingerprint(crop):
    digest = hashlib.sha256()
    digest.update(
        str((crop.image_key, crop.x, crop.y, crop.width, crop.height, crop.pixel_size_nm)).encode()
    )
    digest.update(np.asarray(crop.gt).tobytes())
    digest.update(np.asarray(crop.valid).tobytes())
    return digest.hexdigest()


def evaluation_inputs(data, *, load_em=False):
    try:
        type_id = UUID(str(data.get("segmentation_type", "")))
        asset_ids = [str(UUID(str(value))) for value in data.get("asset_ids", [])]
        dataset_ids = [str(UUID(str(value))) for value in data.get("dataset_ids", [])]
    except (TypeError, ValueError):
        raise ValidationError("Choose an organelle and images to evaluate.") from None
    seg_type = get_object_or_404(SegmentationType, pk=type_id)
    scope = resolve_scope(asset_ids=asset_ids, dataset_ids=dataset_ids)
    if not scope.eligible:
        raise ValidationError(scope.blockers or ["Choose images in one experiment."])
    base = str(data.get("base_model") or "")
    adapter = None
    if data.get("adapter_id"):
        try:
            adapter_id = UUID(str(data["adapter_id"]))
        except (TypeError, ValueError):
            raise ValidationError("Choose a saved model in this experiment.") from None
        adapter = get_object_or_404(
            Adapter,
            pk=adapter_id,
            deleted_at=None,
            status="SUCCESS",
            experiment_id=scope.experiment_id,
            segmentation_type=seg_type,
        )
        base = adapter.base_model
    if (
        base not in MODEL_SPECS
        or seg_type.internal_name != f"quantem_internal_{base.split(':')[-1]}"
    ):
        raise ValidationError("Choose a model for this organelle.")
    crops = collect_crops_for_scope(
        str(type_id), scope.asset_ids, load_em=load_em, load_prob=False, context_model=base
    ).crops
    exposure = (
        set(adapter.training_exposure)
        | {str(value) for value in adapter.scope_assets.values_list("id", flat=True)}
        if adapter
        else set()
    )
    return scope, base, adapter, crops, exposure


def evaluation_body(job):
    return {
        "id": str(job.id),
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "created_at": job.created_at,
        "settings": job.payload_json,
        "results": job.result_json if job.status == "SUCCESS" else None,
    }


class ModelEvaluationsView(APIView):
    def get(self, request):
        jobs = Job.objects.filter(type=JOB_TYPE_EVALUATE_MODEL)
        if request.query_params.get("experiment"):
            jobs = jobs.filter(payload_json__experiment_id=request.query_params["experiment"])
        return Response([evaluation_body(job) for job in jobs.order_by("-created_at")[:25]])

    def post(self, request):
        scope, base, adapter, crops, exposure = evaluation_inputs(request.data)
        rois = [
            {
                "id": crop.id,
                "name": crop.name,
                "image_id": crop.image_key,
                "training_overlap": crop.image_key in exposure,
                "input_fingerprint": crop_fingerprint(crop),
            }
            for crop in crops
        ]
        default = (
            adapter.calibrated_threshold
            if adapter and adapter.calibrated_threshold is not None
            else MODEL_SPECS[base].threshold
        )
        if request.data.get("preview") is True:
            return Response(
                {"rois": rois, "threshold": default, "experiment_id": scope.experiment_id}
            )
        ids = request.data.get("roi_ids", [])
        if not isinstance(ids, list) or not ids or set(ids) - {row["id"] for row in rois}:
            raise ValidationError("Choose completed ROIs from the preview.")
        selected = [row for row in rois if row["id"] in ids]
        try:
            threshold = float(request.data.get("threshold", default))
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError
        except (TypeError, ValueError):
            raise ValidationError("Choose a threshold between 0 and 1.") from None
        job = Job.enqueue(
            job_type=JOB_TYPE_EVALUATE_MODEL,
            payload={
                "segmentation_type": str(request.data["segmentation_type"]),
                "asset_ids": scope.asset_ids,
                "experiment_id": scope.experiment_id,
                "base_model": base,
                "adapter_id": str(adapter.id) if adapter else None,
                "threshold": threshold,
                "rois": selected,
            },
            max_attempts=1,
            **JOB_DEFAULTS[JOB_TYPE_EVALUATE_MODEL],
        )
        return Response(evaluation_body(job), status=202)


class ModelEvaluationDetailView(APIView):
    def get(self, request, job_id):
        return Response(
            evaluation_body(get_object_or_404(Job, pk=job_id, type=JOB_TYPE_EVALUATE_MODEL))
        )
