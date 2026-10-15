"""Build, validate and enqueue experiment measurement batches."""

from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import transaction
from django.db.models import Q
from rest_framework.exceptions import ValidationError

from quantem.analysis.creation import (
    PreparedAnalysis,
    create_analysis_run,
    image_identity,
    prepare_analysis,
)
from quantem.analysis.models import AnalysisBatch, AnalysisBatchItem
from quantem.assets.models import Asset
from quantem.jobs.constants import (
    ACTIVE_SEGMENTATION_JOB_TYPES,
)
from quantem.segmentation.api_views.shared import active_segmentation_job


def preview_batch(experiment, data):
    selected = data.get("asset_ids", [])
    datasets = data.get("dataset_ids", [])
    types = data.get("segmentation_types", [])
    if not all(isinstance(v, list) for v in (selected, datasets, types)):
        raise ValidationError("Image, dataset and organelle selections must be lists.")
    if not selected and not datasets:
        raise ValidationError("Select images or datasets to measure.")
    if not types or not all(isinstance(value, str) for value in types):
        raise ValidationError("Choose at least one organelle.")
    if len(types) > 30:
        raise ValidationError("Select no more than 30 segmentation types.")
    try:
        assets = list(
            Asset.objects.filter(
                Q(id__in=selected) | Q(datasets__id__in=datasets),
                experiment=experiment,
                lifecycle_status="ACTIVE",
            )
            .select_related("workflow_metadata", "experiment")
            .prefetch_related("datasets")
            .distinct()
            .order_by("id")
        )
    except (ValueError, TypeError, DjangoValidationError) as exc:
        raise ValidationError("Invalid image or dataset selection.") from exc
    valid_ids = {str(a.id) for a in assets}
    if set(selected) - valid_ids or set(datasets) - {
        str(v) for v in experiment.datasets.values_list("id", flat=True)
    }:
        raise ValidationError("Every selected image and dataset must belong to this experiment.")
    if len(assets) > 5000:
        raise ValidationError("Select at most 5,000 images per batch.")
    mask_name = str(data.get("mask_name") or "").strip()
    require_complete = data.get("require_complete", True)
    if not isinstance(require_complete, bool) or not isinstance(data.get("reuse", True), bool):
        raise ValidationError("Review and reuse options must be true or false.")
    advanced = data.get("advanced") or {}
    if not isinstance(advanced, dict) or set(advanced) - {"band_edges_nm", "replicates", "seed"}:
        raise ValidationError("Unsupported advanced measurement settings.")
    rows = []
    for asset in assets:
        identity = image_identity(asset)
        segs = list(asset.segmentations.select_related("segmentation_type"))
        masks = [
            s
            for s in segs
            if s.segmentation_type.internal_name == "quantem_internal_analysis_mask"
            and (s.display_name or s.segmentation_type.long_name) == mask_name
        ]
        for internal in dict.fromkeys(types):
            matches = [s for s in segs if s.segmentation_type.internal_name == internal]
            seg = matches[0] if len(matches) == 1 else None
            reason = ""
            if not seg:
                reason = (
                    "Missing segmentation"
                    if not matches
                    else "Multiple segmentations; select an unambiguous type."
                )
            elif asset.preprocess_stage not in ("DONE", "SKIPPED"):
                reason = "Image preparation is incomplete."
            elif seg.status_stage in (
                "RUNNING_INFERENCE",
                "EXTRACTING_CANDIDATES",
                "UPDATING",
                "COMPUTING_FEATURES",
            ):
                reason = "Segmentation is still processing."
            elif require_complete and seg.status_stage != "COMPLETED":
                reason = "Review is not marked complete."
            elif active_segmentation_job(seg, job_types=ACTIVE_SEGMENTATION_JOB_TYPES):
                reason = "A segmentation task is queued or running."
            if mask_name and len(masks) != 1:
                reason = "The selected analysis mask is missing or ambiguous."
            mask = masks[0] if len(masks) == 1 else None
            params = {"group": identity["group"]}
            prepared = None
            if seg and not reason:
                if seg.segmentation_type.measurement_mode == "global":
                    raw = {
                        "analysis_mask_ids": [str(mask.id)] if mask else [],
                        "group": identity["group"],
                    }
                else:
                    raw = {
                        "objects_in_measurement_area": True,
                        "tissue_segmentation_id": str(mask.id) if mask else None,
                        "group": identity["group"],
                    }
                    raw.update(advanced)
                try:
                    prepared = prepare_analysis(seg, raw)
                    params = prepared.params
                except ValidationError as exc:
                    reason = str(exc.detail)
            key = prepared.fingerprint if prepared else ""
            rows.append(
                {
                    "segmentation": seg,
                    "params": params,
                    "fingerprint": key,
                    "reason": reason,
                    "identity": prepared.identity
                    if prepared
                    else {
                        **identity,
                        "organelle": internal,
                        "structure": (
                            seg.display_name
                            or seg.segmentation_type.short_name
                            or seg.segmentation_type.long_name
                        )
                        if seg
                        else internal,
                        "segmentation_id": str(seg.id) if seg else None,
                        "measurement_area": mask_name or "Whole image",
                        "review_complete": bool(seg and seg.status_stage == "COMPLETED"),
                        "provenance": seg.final_result_provenance if seg else {},
                    },
                }
            )
    return rows


def start_batch(experiment, data):
    rows = preview_batch(experiment, data)
    if not any(not row["reason"] for row in rows):
        raise ValidationError("No selected image is ready. Check the readiness table.")
    with transaction.atomic():
        batch = AnalysisBatch.objects.create(
            experiment=experiment,
            experiment_name=experiment.name,
            name=str(data.get("name") or "Experiment measurements")[:255],
            settings=data,
        )
        for row in rows:
            run = job = None
            reused = False
            if not row["reason"]:
                run, job, reused = create_analysis_run(
                    row["segmentation"],
                    PreparedAnalysis(row["params"], row["identity"], row["fingerprint"]),
                    batch_id=f"analysis-batch:{batch.id}",
                    reuse=data.get("reuse", True),
                )
            AnalysisBatchItem.objects.create(
                batch=batch,
                segmentation=row["segmentation"],
                run=run,
                job=job,
                identity=row["identity"],
                params=row["params"],
                fingerprint=row["fingerprint"],
                skip_reason=row["reason"],
                reused=reused,
            )
    return batch
