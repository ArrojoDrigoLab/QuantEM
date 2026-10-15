from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import transaction
from django.shortcuts import get_object_or_404
from django.urls import path
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.segmentation.api_views.shared import active_segmentation_job, completion_lock_response
from quantem.segmentation.models import CorrectionRevision, ImageSegmentation
from quantem.segmentation.revisions import restore_revision, save_revision


class CorrectionRevisionsView(APIView):
    def get(self, request, seg_id):
        get_object_or_404(ImageSegmentation, id=seg_id)
        return Response(
            list(
                CorrectionRevision.objects.filter(segmentation_id=seg_id)
                .order_by("-created_at")
                .values("id", "name", "created_at", "object_count", "include_level")
            )
        )

    def post(self, request, seg_id):
        with transaction.atomic():
            segmentation = get_object_or_404(
                ImageSegmentation.objects.select_for_update(), id=seg_id
            )
            if active_segmentation_job(segmentation) is not None:
                return Response(
                    {
                        "detail": "Wait for the image's active task before saving or restoring a revision."
                    },
                    status=409,
                )
            try:
                if request.data.get("restore"):
                    locked = completion_lock_response(segmentation)
                    if locked is not None:
                        return locked
                    revision = get_object_or_404(
                        CorrectionRevision, id=request.data["restore"], segmentation=segmentation
                    )
                    overlay = restore_revision(revision)
                    return Response({"restored": str(revision.id), "overlay": overlay})
                name = str(request.data.get("name") or "Saved corrections").strip()
                if len(name) > 255:
                    raise ValueError("Revision names must be at most 255 characters.")
                revision = save_revision(segmentation, name)
            except (ValueError, OSError, DjangoValidationError) as exc:
                raise ValidationError(str(exc)) from exc
        return Response({"id": str(revision.id)}, status=201)


urlpatterns = [path("segmentations/<uuid:seg_id>/revisions/", CorrectionRevisionsView.as_view())]
