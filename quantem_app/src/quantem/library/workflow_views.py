"""Experiment workspace navigation and explicitly entered sample identities."""

import re
import uuid

from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.analysis.history import analysis_counts
from quantem.assets.models import Asset
from quantem.library.models import Experiment, ImageWorkflowMetadata
from quantem.library.sample_identity import sample_identity
from quantem.library.segmentation_coverage import coverage, with_coverage
from quantem.segmentation.models import ImageSegmentation


class ExperimentNavigationView(APIView):
    def post(self, request, experiment_id):
        experiment = get_object_or_404(Experiment, pk=experiment_id)
        route = str(request.data.get("route", ""))
        root = f"/experiments/{experiment.id}"
        asset_match = re.fullmatch(r"/assets/([0-9a-f-]{36})(?:/[^?#]*)?(?:\?[^#]*)?", route)
        allowed = route == root or route.startswith(root + "?")
        if asset_match:
            try:
                asset_id = uuid.UUID(asset_match[1])
            except ValueError:
                allowed = False
            else:
                allowed = Asset.objects.filter(
                    id=asset_id, experiment=experiment, lifecycle_status="ACTIVE"
                ).exists()
        if not allowed or len(route) > 1024:
            return Response({"detail": "Choose an image or page in this experiment."}, status=400)
        Experiment.objects.filter(pk=experiment.pk).update(
            last_opened_at=timezone.now(), last_route=route
        )
        return Response({"saved": True})


class ExperimentWorkspaceView(APIView):
    def get(self, request, experiment_id):
        experiment = get_object_or_404(Experiment, pk=experiment_id)
        assets = (
            experiment.assets.filter(lifecycle_status="ACTIVE")
            .select_related("workflow_metadata")
            .prefetch_related("datasets")
            .order_by("display_name", "id")
        )
        if request.query_params.get("navigation") == "true":
            return Response(
                {
                    "images": [
                        {"id": str(asset.id), "name": asset.display_name}
                        for asset in assets.select_related(None)
                        .prefetch_related(None)
                        .only("id", "display_name", "experiment_id")
                    ]
                }
            )
        segmentations = with_coverage(
            ImageSegmentation.objects.filter(asset__in=assets)
            .select_related("segmentation_type", "asset")
            .annotate(
                confirmed_count=Count(
                    "segments",
                    filter=Q(
                        segments__label_state="CONFIRMED", segments__superseded_at__isnull=True
                    ),
                ),
            )
        )
        by_asset = {}
        completed_counts = analysis_counts(experiment)
        for seg in segmentations:
            by_asset.setdefault(str(seg.asset_id), []).append(
                {
                    "id": str(seg.id),
                    "type_id": str(seg.segmentation_type_id),
                    "type": seg.segmentation_type.internal_name,
                    "name": seg.display_name or seg.segmentation_type.long_name,
                    "short_name": seg.segmentation_type.short_name,
                    **coverage(seg),
                    "measurement_mode": seg.segmentation_type.measurement_mode,
                    "stage": seg.status_stage,
                    "confirmed_count": seg.confirmed_count,
                    "review_complete": seg.status_stage == "COMPLETED",
                    "provenance": seg.final_result_provenance,
                }
            )
        rows = []
        for asset in assets:
            meta = getattr(asset, "workflow_metadata", None)
            rows.append(
                {
                    "id": str(asset.id),
                    "name": asset.display_name,
                    "pixel_size_nm": asset.pixel_size_nm,
                    "stage": asset.preprocess_stage,
                    "analysis_count": completed_counts.get(str(asset.id), 0),
                    "datasets": [{"id": str(d.id), "name": d.name} for d in asset.datasets.all()],
                    **sample_identity(
                        asset.display_name or asset.original_filename,
                        asset.id,
                        meta.sample_id if meta else "",
                    ),
                    "group": meta.group if meta else "",
                    "segmentations": by_asset.get(str(asset.id), []),
                }
            )
        return Response({"images": rows})

    def patch(self, request, experiment_id):
        try:
            asset_id = uuid.UUID(str(request.data.get("asset_id", "")))
        except ValueError:
            return Response({"detail": "Choose a valid image."}, status=400)
        asset = get_object_or_404(
            Asset,
            id=asset_id,
            experiment_id=experiment_id,
            lifecycle_status="ACTIVE",
        )
        values = {}
        for key in ("sample_id", "group"):
            if key in request.data:
                value = request.data[key]
                if not isinstance(value, str) or len(value) > 255:
                    return Response(
                        {"detail": f"{key} must be text up to 255 characters."}, status=400
                    )
                values[key] = value.strip()
        metadata, _ = ImageWorkflowMetadata.objects.update_or_create(asset=asset, defaults=values)
        return Response(
            {
                **sample_identity(
                    asset.display_name or asset.original_filename, asset.id, metadata.sample_id
                ),
                "group": metadata.group,
            }
        )
