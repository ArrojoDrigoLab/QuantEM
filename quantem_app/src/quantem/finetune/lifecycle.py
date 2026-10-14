"""Manage saved models without removing their result identities."""

import uuid

from django.db import transaction
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.finetune.models import Adapter, active_adapter_for
from quantem.finetune.run_views import serialize_run
from quantem.jobs.models import Job


def model_jobs(adapter):
    from quantem.segmentation.models import ImageSegmentation

    for job in Job.objects.filter(status__in=Job.OPEN_STATUSES):
        # Includes inference legs and child fine-tune initialization.
        def references(value):
            if isinstance(value, dict):
                # Older and implicit inference requests resolve the applied
                # model at execution time instead of carrying an adapter id.
                if not value.get("adapter_id") and value.get("segmentation_id"):
                    try:
                        segmentation_id = uuid.UUID(str(value["segmentation_id"]))
                    except ValueError:
                        segmentation_id = None
                    segmentation = ImageSegmentation.objects.filter(id=segmentation_id).first()
                    if segmentation:
                        active = active_adapter_for(segmentation)
                        if active and active.id == adapter.id:
                            return True
                return any(
                    (key in ("adapter_id", "parent_adapter_id") and str(v) == str(adapter.id))
                    or references(v)
                    for key, v in value.items()
                )
            return isinstance(value, list) and any(references(v) for v in value)

        if references(job.payload_json):
            return True
    return False


class SavedModelsView(APIView):
    def get(self, request):
        rows = (
            Adapter.objects.filter(deleted_at__isnull=True)
            .select_related("experiment", "segmentation_type", "parent")
            .prefetch_related("scope_assets", "scope_datasets")
        )
        if request.query_params.get("experiment"):
            rows = rows.filter(experiment_id=request.query_params["experiment"])
        if request.query_params.get("archived") != "true":
            rows = rows.filter(archived_at__isnull=True)
        return Response([serialize_run(adapter) for adapter in rows.order_by("-created_at")])


class SavedModelView(APIView):
    def patch(self, request, adapter_id):
        with transaction.atomic():
            adapter = get_object_or_404(
                Adapter.objects.select_for_update(), pk=adapter_id, deleted_at__isnull=True
            )
            if "name" in request.data:
                name = str(request.data["name"]).strip()
                if not name or len(name) > 255:
                    raise ValidationError("Enter a model name up to 255 characters.")
                if (
                    Adapter.objects.filter(name=name, segmentation_type=adapter.segmentation_type)
                    .exclude(pk=adapter.pk)
                    .exists()
                ):
                    raise ValidationError("Another saved model already has that name.")
                adapter.name = name
            if "archived" in request.data:
                if not isinstance(request.data["archived"], bool):
                    raise ValidationError("archived must be true or false.")
                adapter.archived_at = timezone.now() if request.data["archived"] else None
            adapter.save()
        return Response(serialize_run(adapter))

    def delete(self, request, adapter_id):
        with transaction.atomic():
            adapter = get_object_or_404(
                Adapter.objects.select_for_update(), pk=adapter_id, deleted_at__isnull=True
            )
            if model_jobs(adapter) or adapter.status in ("PENDING", "RUNNING"):
                return Response(
                    {"detail": "This model is in use by a task. Wait or cancel the task first."},
                    status=409,
                )
            if adapter.versions.filter(deleted_at__isnull=True).exists():
                return Response(
                    {
                        "detail": "This model was used to train another saved model and cannot be deleted. You can archive it instead."
                    },
                    status=409,
                )
            head = adapter.head_file
            if head is not None:
                from quantem.core.config import MODELS_DIR

                if not head.resolve().is_relative_to(MODELS_DIR.resolve()):
                    return Response(
                        {"detail": "The weights are outside managed model storage."}, status=409
                    )
                head.unlink(missing_ok=True)
            adapter.deleted_at = timezone.now()
            adapter.archived_at = adapter.deleted_at
            adapter.head_path = ""
            adapter.applied_at = None
            adapter.save()
        return Response(status=204)
