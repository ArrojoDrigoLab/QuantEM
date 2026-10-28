"""Build, validate and enqueue experiment measurement batches."""

from uuid import UUID

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
from quantem.segmentation.models import SegmentationType


def _has_saved_area(segmentation) -> bool:
    if segmentation.analysis_mask_objects.exclude(geometry_wkb__isnull=True).exists():
        return True
    try:
        return int(segmentation.global_mask.foreground_pixels) > 0
    except Exception:
        return False


def preview_batch(experiment, data):
    selected = data.get("asset_ids", [])
    datasets = data.get("dataset_ids", [])
    definition_ids = data.get("segmentation_type_ids") or []
    legacy_names = data.get("segmentation_types") or []
    using_definition_ids = bool(definition_ids)
    types = definition_ids if using_definition_ids else legacy_names
    if not all(isinstance(v, list) for v in (selected, datasets, types)):
        raise ValidationError("Image, dataset and organelle selections must be lists.")
    if not selected and not datasets:
        raise ValidationError("Select images or datasets to measure.")
    allowed_selector_types = (str, UUID) if using_definition_ids else (str,)
    if not types or not all(isinstance(value, allowed_selector_types) for value in types):
        raise ValidationError("Choose at least one organelle.")
    types = [str(value) for value in types]
    if using_definition_ids:
        try:
            for value in types:
                UUID(value)
        except (TypeError, ValueError):
            raise ValidationError("Choose reusable organelle definitions by their IDs.") from None
        valid_type_ids = {
            str(value)
            for value in SegmentationType.objects.filter(
                id__in=types, category=SegmentationType.CATEGORY_ORGANELLE
            ).values_list("id", flat=True)
        }
        if set(types) - valid_type_ids:
            raise ValidationError("Choose reusable organelle definitions by their IDs.")
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
    mask_definition_id = str(data.get("mask_definition_id") or "").strip()
    if mask_definition_id:
        try:
            UUID(mask_definition_id)
        except (TypeError, ValueError):
            raise ValidationError("Choose the reusable Sample Area definition by its ID.") from None
        if not SegmentationType.objects.filter(
            id=mask_definition_id,
            category=SegmentationType.CATEGORY_AREA,
            internal_name="quantem_internal_tissue",
        ).exists():
            raise ValidationError("Choose the reusable Sample Area definition by its ID.")
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
        segs = list(
            asset.segmentations.filter(superseded_by__isnull=True).select_related(
                "segmentation_type"
            )
        )
        masks = [
            s
            for s in segs
            if (mask_definition_id and str(s.segmentation_type_id) == mask_definition_id)
            or (
                not mask_definition_id
                and mask_name
                and s.segmentation_type.internal_name == "quantem_internal_analysis_mask"
                and (s.display_name or s.segmentation_type.long_name) == mask_name
            )
        ]
        cells = next(
            (
                s
                for s in segs
                if s.segmentation_type.internal_name == "quantem_internal_cells"
                and (s.status_stage == "COMPLETED" or _has_saved_area(s))
            ),
            None,
        )
        for selector in dict.fromkeys(types):
            matches = [
                s
                for s in segs
                if (
                    str(s.segmentation_type_id) == selector
                    if using_definition_ids
                    else s.segmentation_type.internal_name == selector
                )
            ]
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
            if (mask_definition_id or mask_name) and len(masks) != 1:
                reason = "The selected Sample Area definition is missing or ambiguous."
            mask = masks[0] if len(masks) == 1 else None
            params = {"group": identity["group"]}
            prepared = None
            if seg and not reason:
                if not using_definition_ids and seg.segmentation_type.measurement_mode == "global":
                    raw = {
                        "analysis_mask_ids": [str(mask.id)] if mask else [],
                        "group": identity["group"],
                    }
                elif not using_definition_ids:
                    raw = {
                        "objects_in_measurement_area": True,
                        "tissue_segmentation_id": str(mask.id) if mask else None,
                        "group": identity["group"],
                    }
                else:
                    raw = {
                        "objects_in_measurement_area": True,
                        "tissue_segmentation_id": str(mask.id) if mask else None,
                        "cell_segmentation_id": str(cells.id) if cells else None,
                        "outputs": {
                            "whole_image": True,
                            "sample_area": bool(mask),
                            "all_cells": bool(cells),
                            "per_cell": bool(cells),
                            "individual_objects": True,
                        },
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
                        "organelle": seg.segmentation_type.internal_name if seg else selector,
                        "segmentation_type_id": (
                            str(seg.segmentation_type_id)
                            if seg
                            else selector
                            if using_definition_ids
                            else None
                        ),
                        "structure": (
                            seg.display_name
                            or seg.segmentation_type.short_name
                            or seg.segmentation_type.long_name
                        )
                        if seg
                        else selector,
                        "segmentation_id": str(seg.id) if seg else None,
                        "measurement_area": (
                            (mask.display_name or mask.segmentation_type.long_name)
                            if mask
                            else "Whole image"
                        ),
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
