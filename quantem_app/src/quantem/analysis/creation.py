"""The shared validation, snapshot and queue lifecycle for all analysis entry points."""

from dataclasses import dataclass
from uuid import UUID

from django.db import transaction
from rest_framework import serializers

from quantem.analysis.identity import input_fingerprint
from quantem.analysis.models import AnalysisRun
from quantem.analysis.serializers import AnalysisRunCreateSerializer
from quantem.assets.models import Asset
from quantem.jobs.constants import JOB_DEFAULTS, JOB_TYPE_RUN_ANALYSIS
from quantem.jobs.models import Job
from quantem.library.sample_identity import sample_identity
from quantem.segmentation.models import ImageSegmentation


class GlobalAnalysisSerializer(serializers.Serializer):
    analysis_mask_ids = serializers.ListField(child=serializers.UUIDField(), default=list)
    group = serializers.CharField(required=False, allow_blank=True, default="")

    def validate_analysis_mask_ids(self, values: list[UUID]) -> list[str]:
        ids = list(dict.fromkeys(str(value) for value in values))
        valid = ImageSegmentation.objects.filter(
            id__in=ids,
            asset_id=self.context["segmentation"].asset_id,
            segmentation_type__internal_name="quantem_internal_analysis_mask",
        ).count()
        if valid != len(ids):
            raise serializers.ValidationError("Choose analysis masks on this image.")
        return ids


@dataclass(frozen=True)
class PreparedAnalysis:
    params: dict
    identity: dict
    fingerprint: str


def image_identity(asset: Asset) -> dict:
    metadata = getattr(asset, "workflow_metadata", None)
    experiment = asset.experiment
    return {
        "asset_id": str(asset.id),
        "image": asset.display_name,
        "experiment_id": str(experiment.id) if experiment else None,
        "experiment": experiment.name if experiment else "",
        "datasets": [dataset.name for dataset in asset.datasets.all()],
        **sample_identity(
            asset.display_name or asset.original_filename,
            asset.id,
            metadata.sample_id if metadata else "",
        ),
        "group": metadata.group if metadata else "",
        "pixel_size_nm": asset.pixel_size_nm,
    }


def prepare_analysis(segmentation: ImageSegmentation, data: dict) -> PreparedAnalysis:
    global_mode = segmentation.segmentation_type.measurement_mode == "global"
    serializer_type = GlobalAnalysisSerializer if global_mode else AnalysisRunCreateSerializer
    serializer = serializer_type(data=data, context={"segmentation": segmentation})
    serializer.is_valid(raise_exception=True)
    params = dict(serializer.validated_data)
    if global_mode:
        params["global_area"] = True
    mask_ids = params.get("analysis_mask_ids", [])
    if params.get("tissue_segmentation_id"):
        mask_ids = [params["tissue_segmentation_id"]]
    masks = (
        ImageSegmentation.objects.filter(id__in=mask_ids)
        .select_related("segmentation_type")
        .order_by("id")
    )
    area = ", ".join(mask.display_name or mask.segmentation_type.long_name for mask in masks)
    identity = {
        **image_identity(segmentation.asset),
        "organelle": segmentation.segmentation_type.internal_name,
        "structure": segmentation.display_name
        or segmentation.segmentation_type.short_name
        or segmentation.segmentation_type.long_name,
        "segmentation_id": str(segmentation.id),
        "measurement_area": area or "Whole image",
        "review_complete": segmentation.status_stage == "COMPLETED",
        "provenance": segmentation.final_result_provenance,
    }
    return PreparedAnalysis(params, identity, input_fingerprint(segmentation, params))


def enqueue_run(run: AnalysisRun, *, batch_id: str = "") -> Job:
    defaults = JOB_DEFAULTS[JOB_TYPE_RUN_ANALYSIS]
    return Job.enqueue(
        job_type=JOB_TYPE_RUN_ANALYSIS,
        payload={
            "analysis_run_id": str(run.id),
            "segmentation_id": str(run.segmentation_id),
            "asset_id": str(run.segmentation.asset_id),
        },
        priority=defaults["priority"],
        resource_class=defaults["resource_class"],
        queue_name=defaults["queue_name"],
        max_attempts=1,
        batch_id=batch_id,
        tags=[f"analysis_run:{run.id}", f"segmentation:{run.segmentation_id}"],
    )


@transaction.atomic
def create_analysis_run(
    segmentation: ImageSegmentation,
    prepared: PreparedAnalysis,
    *,
    batch_id: str = "",
    reuse: bool = False,
) -> tuple[AnalysisRun, Job | None, bool]:
    """Return (run, job, reused), committing a run and its job together."""
    if reuse:
        run = (
            AnalysisRun.objects.filter(
                segmentation=segmentation,
                status="SUCCESS",
                input_fingerprint=prepared.fingerprint,
                input_identity=prepared.identity,
            )
            .order_by("-created_at")
            .first()
        )
        if run and run.export_path and run.export_path.is_dir():
            return run, None, True
    run = AnalysisRun.objects.create(
        segmentation=segmentation,
        params=prepared.params,
        group=prepared.params.get("group", ""),
        input_fingerprint=prepared.fingerprint,
        input_identity=prepared.identity,
    )
    return run, enqueue_run(run, batch_id=batch_id), False
