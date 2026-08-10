"""Guided fine-tuning endpoints, per ``API_CONTRACT.md``.

Four routes: what the user has annotated, start an adaptation, read the result,
apply it. No authentication anywhere — QuantEM is single-user and loopback-only.

Every response that carries a Dice also carries its ``split_mode``, the names of
the crops the threshold was fit on, and the oracle ceiling. That is enforced
here rather than left to the frontend, because the honesty rules are about what
the number *means*, and the backend is where the number is made.
"""

from __future__ import annotations

from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.finetune.adapt import torch_available
from quantem.finetune.job import MODE_HEAD, MODE_THRESHOLD_ONLY, MODES
from quantem.finetune.models import Adapter
from quantem.inference.specs import MODEL_SPECS
from quantem.jobs.constants import JOB_DEFAULTS, JOB_TYPE_TRAIN_ORGANELLE_ADAPTER
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation
from quantem.segmentation.services.adapt import collect_crops


def _error(message: str, code: int = status.HTTP_400_BAD_REQUEST) -> Response:
    return Response({"error": message}, status=code)


def serialize_adapter(adapter: Adapter) -> dict[str, object]:
    """The ``GET /api/adapters/<id>/`` body."""
    sweep = adapter.sweep or {}
    params = adapter.params or {}
    return {
        "id": str(adapter.id),
        "base_model": adapter.base_model,
        "name": adapter.name,
        "status": adapter.status,
        "mode": adapter.mode,
        "steps": params.get("steps", 0),
        "trainable_params": adapter.trainable_params,
        "segmentation_id": (
            str(adapter.segmentation_id) if adapter.segmentation_id else None
        ),
        "split_mode": adapter.split_mode,
        "train_crop_names": sweep.get("train_crop_names", []),
        "heldout_crop_names": sweep.get("heldout_crop_names", []),
        "sweep": sweep,
        "calibrated_threshold": adapter.calibrated_threshold,
        "heldout_dice": adapter.heldout_dice,
        "verified_reload": adapter.verified_reload,
        "train_seconds": adapter.train_seconds,
        "applied_at": adapter.applied_at,
        "created_at": adapter.created_at,
        "error": adapter.error,
        "caveats": adapter.caveats(),
    }


class AdaptCropsView(APIView):
    """``GET /api/segmentations/<seg_id>/adapt/crops/``"""

    def get(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        crop_set = collect_crops(segmentation)
        body = crop_set.as_api_dict()
        # Which rungs this machine can actually offer. threshold_only needs no
        # torch and no GPU, which is exactly why it is listed unconditionally.
        body["modes"] = [MODE_THRESHOLD_ONLY] + (
            [MODE_HEAD] if torch_available() else []
        )
        return Response(body, status=status.HTTP_200_OK)


class AdaptStartView(APIView):
    """``POST /api/segmentations/<seg_id>/adapt/``"""

    def post(self, request, seg_id):
        segmentation = get_object_or_404(
            ImageSegmentation.objects.select_related("asset", "segmentation_type"),
            id=seg_id,
        )
        data = request.data if isinstance(request.data, dict) else {}

        base_model = str(data.get("base_model") or "").strip()
        if base_model not in MODEL_SPECS:
            return _error(
                f"Unknown model {base_model!r}. Choose one of: "
                f"{', '.join(sorted(MODEL_SPECS))}."
            )

        mode = str(data.get("mode") or MODE_THRESHOLD_ONLY).strip().lower()
        if mode not in MODES:
            return _error(f"mode must be one of {', '.join(MODES)}.")
        if mode == MODE_HEAD and not torch_available():
            return _error(
                "Head training needs PyTorch, which is not installed here. "
                "Threshold calibration works without it."
            )

        crop_set = collect_crops(segmentation)
        if not crop_set.ready:
            return _error(crop_set.blockers[0])

        params = {
            "steps": int(data.get("steps") or 300),
            "lr": float(data.get("lr") or 1e-4),
            "seed": int(data.get("seed") or 0),
        }
        adapter = Adapter.objects.create(
            segmentation=segmentation,
            base_model=base_model,
            name=str(data.get("name") or "").strip(),
            mode=mode,
            params=params,
            split_mode=crop_set.split_mode,
        )

        defaults = JOB_DEFAULTS[JOB_TYPE_TRAIN_ORGANELLE_ADAPTER]
        job = Job.enqueue(
            job_type=JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
            payload={
                "segmentation_id": str(segmentation.id),
                "adapter_id": str(adapter.id),
                "base_model": base_model,
                "mode": mode,
                **params,
                "name": adapter.name,
            },
            priority=defaults["priority"],
            resource_class=defaults["resource_class"],
            queue_name=defaults["queue_name"],
            tags=[
                f"segmentation:{segmentation.id}",
                f"adapter:{adapter.id}",
            ],
        )
        return Response(
            {"job_id": str(job.id), "adapter_id": str(adapter.id)},
            status=status.HTTP_202_ACCEPTED,
        )


class AdapterDetailView(APIView):
    """``GET /api/adapters/<adapter_id>/``"""

    def get(self, request, adapter_id):
        adapter = get_object_or_404(Adapter, id=adapter_id)
        return Response(serialize_adapter(adapter), status=status.HTTP_200_OK)


class AdapterApplyView(APIView):
    """``POST /api/adapters/<adapter_id>/apply/``

    Stamps the adapter as the one to use for subsequent runs on its
    segmentation. Refused before the run has succeeded: applying a pending
    adapter would mean running with a threshold that has not been fitted yet.
    """

    def post(self, request, adapter_id):
        adapter = get_object_or_404(Adapter, id=adapter_id)
        if adapter.status != "SUCCESS":
            return _error(
                f"This adapter is {adapter.status.lower()}; only a finished "
                "adapter can be applied.",
                status.HTTP_409_CONFLICT,
            )
        adapter.applied_at = timezone.now()
        adapter.save(update_fields=["applied_at", "updated_at"])
        return Response(serialize_adapter(adapter), status=status.HTTP_200_OK)
