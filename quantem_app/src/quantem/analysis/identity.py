"""Measurement input identity, independent of UI sorting and background progress."""

import hashlib
import json

from quantem.assets.models import Asset
from quantem.segmentation.models import (
    AnalysisMaskObject,
    AnalysisMaskObjectAnnotation,
    CompletedROI,
    GlobalMask,
    ImageSegmentation,
    RoiSegmentationStatus,
    SavedSegmentationResult,
    SegmentObject,
    SegmentObjectAnnotation,
)
from quantem.segmentation.run_identity import read_run_identity

# Bump when measurement semantics or the input contract changes. Old runs
# remain downloadable, but cannot silently serve as current measurements.
MEASUREMENT_SCHEMA_VERSION = 2


def input_fingerprint(segmentation, params):
    from .input_contract import dependency_ids

    ids = dependency_ids(segmentation.id, params)
    digest = hashlib.sha256()
    asset = Asset.objects.select_related("workflow_metadata").get(id=segmentation.asset_id)
    metadata = getattr(asset, "workflow_metadata", None)
    digest.update(
        json.dumps(
            {
                "measurement_schema": MEASUREMENT_SCHEMA_VERSION,
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
                    "manual_full_image": seg.manual_full_image,
                    "scientific_revision": seg.scientific_revision,
                    "area_vectors_authoritative": seg.area_vectors_authoritative,
                },
                sort_keys=True,
                default=str,
            ).encode()
        )
    sources = [
        (SavedSegmentationResult, ("id", "is_canonical", "full_image", "threshold", "deleted_at")),
        (SegmentObject, ("id", "geometry_wkb", "label_state", "source_model", "features")),
        (CompletedROI, ("id", "geometry_wkb")),
        (
            RoiSegmentationStatus,
            (
                "id",
                "is_complete",
                "image_roi__x",
                "image_roi__y",
                "image_roi__width",
                "image_roi__height",
            ),
        ),
        (AnalysisMaskObject, ("id", "geometry_wkb", "name")),
        (GlobalMask, ("id", "file_path", "updated_at", "foreground_pixels")),
    ]
    for model, fields in sources:
        queryset = model.objects.filter(segmentation_id__in=ids)
        if model is SegmentObject:
            queryset = queryset.filter(superseded_at__isnull=True)
        if model is SavedSegmentationResult:
            queryset = queryset.filter(is_canonical=True, deleted_at=None)
        for row in queryset.order_by("id").values_list(*fields).iterator(chunk_size=1000):
            for field, value in zip(fields, row, strict=True):
                if field == "features":
                    value = json.dumps(read_run_identity(value), sort_keys=True, default=str)
                raw = (
                    bytes(value) if isinstance(value, (bytes, memoryview)) else str(value).encode()
                )
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
    for model in (SegmentObjectAnnotation, AnalysisMaskObjectAnnotation):
        annotations = model.objects.filter(object__segmentation_id__in=ids)
        if model is SegmentObjectAnnotation:
            annotations = annotations.filter(object__superseded_at__isnull=True)
        for row in annotations.order_by("id").values():
            digest.update(json.dumps(row, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def run_is_current(run, *, fingerprints=None):
    if run.status != "SUCCESS":
        return False
    if run.superseded_at:
        return False
    if not run.segmentation_id or not run.input_fingerprint:
        return False
    try:
        key = (str(run.segmentation_id), json.dumps(run.params, sort_keys=True, default=str))
        if fingerprints is None:
            current = input_fingerprint(run.segmentation, run.params)
        else:
            # Request-local only: identical recipes in a history list share one
            # geometry scan. The next request always observes committed edits.
            if key not in fingerprints:
                fingerprints[key] = input_fingerprint(run.segmentation, run.params)
            current = fingerprints[key]
        return current == run.input_fingerprint
    except (ValueError, OSError, Asset.DoesNotExist):
        return False
