"""Measurement input identity, independent of UI sorting and background progress."""

import hashlib
import json

from quantem.assets.models import Asset
from quantem.segmentation.models import (
    AnalysisMaskObject,
    CompletedROI,
    GlobalMask,
    ImageSegmentation,
    SegmentObject,
)
from quantem.segmentation.run_identity import read_run_identity


def input_fingerprint(segmentation, params):
    ids = {str(segmentation.id)}
    ids.update(str(v) for v in params.get("compartments", {}).values())
    ids.update(str(v) for v in params.get("analysis_mask_ids", []))
    if params.get("tissue_segmentation_id"):
        ids.add(str(params["tissue_segmentation_id"]))
    if params.get("cell_segmentation_id"):
        ids.add(str(params["cell_segmentation_id"]))
    # Consolidated legacy area masks remain as superseded source rows so their
    # raster files can be unioned without rewriting user data during a schema
    # migration. They are therefore measurement inputs and must invalidate a
    # saved run if one changes.
    ids.update(
        str(value)
        for value in ImageSegmentation.objects.filter(
            superseded_by_id__in=ids
        ).values_list("id", flat=True)
    )
    digest = hashlib.sha256()
    asset = Asset.objects.select_related("workflow_metadata").get(id=segmentation.asset_id)
    metadata = getattr(asset, "workflow_metadata", None)
    digest.update(
        json.dumps(
            {
                "params": params,
                "pixel_size_nm": asset.pixel_size_nm,
                "width": asset.logical_width,
                "height": asset.logical_height,
                "image_hash": asset.source_sha256,
                "sample_id": metadata.sample_id if metadata else "",
                "group": metadata.group if metadata else "",
                "experiment_id": str(asset.experiment_id),
            },
            sort_keys=True,
            default=str,
        ).encode()
    )
    for seg in ImageSegmentation.objects.filter(id__in=ids).order_by("id"):
        digest.update(
            json.dumps(
                {
                    "id": str(seg.id),
                    "level": seg.include_level,
                    "final": seg.final_result_provenance,
                    "complete": seg.status_stage == "COMPLETED",
                },
                sort_keys=True,
                default=str,
            ).encode()
        )
    sources = [
        (SegmentObject, ("id", "geometry_wkb", "label_state", "source_model", "features")),
        (CompletedROI, ("id", "geometry_wkb")),
        (AnalysisMaskObject, ("id", "geometry_wkb", "name")),
        (GlobalMask, ("id", "file_path", "updated_at", "foreground_pixels")),
    ]
    for model, fields in sources:
        queryset = model.objects.filter(segmentation_id__in=ids)
        if model is SegmentObject:
            queryset = queryset.filter(superseded_at__isnull=True)
        for row in queryset.order_by("id").values_list(*fields).iterator(chunk_size=1000):
            for field, value in zip(fields, row, strict=True):
                if field == "features":
                    value = json.dumps(read_run_identity(value), sort_keys=True, default=str)
                raw = (
                    bytes(value) if isinstance(value, (bytes, memoryview)) else str(value).encode()
                )
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
    return digest.hexdigest()


def run_is_current(run):
    if not run.segmentation_id or not run.input_fingerprint:
        return False
    try:
        return input_fingerprint(run.segmentation, run.params) == run.input_fingerprint
    except (ValueError, OSError, Asset.DoesNotExist):
        return False
