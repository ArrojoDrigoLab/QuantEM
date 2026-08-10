from __future__ import annotations

from pathlib import Path

from django.apps import apps

# TODO(quantem): ``path_value_is_absolute_like`` lived in the dropped
# ``core.local_state`` package and is re-exported here by ``core.local_storage``,
# which is its only other consumer. Import it from wherever core settles it.
from quantem.core.local_storage import (
    StoragePath,
    path_value_is_absolute_like,
    storage_relpath_for_path,
    validate_storage_relpath,
)
from quantem.jobs.constants import (
    JOB_TYPE_ENSURE_IMAGE_NGFF,
    JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
    JOB_TYPE_RUN_ANALYSIS,
    JOB_TYPE_RUN_SEGMENTATION_FULL,
    JOB_TYPE_RUN_SEGMENTATION_ROI,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
)


def _model(model_label: str):
    app_label, model_name = model_label.split(".", 1)
    return apps.get_model(app_label, model_name)


def _rendition_file_path(rendition, *, required: bool = True) -> StoragePath | None:
    raw_stored = str(getattr(rendition, "stored_path", "") or "").strip()
    if not raw_stored:
        return None
    if path_value_is_absolute_like(raw_stored):
        return None
    storage_root = str(getattr(rendition, "storage_root", "") or "")
    if storage_root == "DATA_DIR":
        relpath = validate_storage_relpath(Path("data") / raw_stored)
    elif storage_root == "NGFF_TMP_DIR":
        relpath = validate_storage_relpath(Path("data/tmp/ngff") / raw_stored)
    elif storage_root == "STORAGE_DIR":
        relpath = validate_storage_relpath(raw_stored)
    else:
        try:
            relpath = storage_relpath_for_path(raw_stored)
        except Exception:
            return None
    return StoragePath(
        relpath,
        path_type="dir" if getattr(rendition, "is_directory", False) else "file",
        required=required,
    )


def _asset_file_path(asset_id: str, *, required: bool = True) -> StoragePath | None:
    Rendition = _model("assets.Rendition")
    rendition = (
        Rendition.objects.filter(
            asset_id=asset_id,
            type__in=["FULL", "SUBSET"],
        )
        .exclude(stored_path="")
        .order_by("-path_exists", "type", "created_at")
        .first()
    )
    return _rendition_file_path(rendition, required=required) if rendition else None


def _segmentation_image_path(segmentation_id: str) -> StoragePath | None:
    ImageSegmentation = _model("segmentation.ImageSegmentation")
    segmentation = (
        ImageSegmentation.objects.select_related("asset")
        .filter(id=segmentation_id)
        .first()
    )
    if segmentation is None or not getattr(segmentation, "asset_id", None):
        return None
    return _asset_file_path(str(segmentation.asset_id))


def _asset_ngff_path(asset_id: str, *, required: bool = False) -> StoragePath:
    Rendition = _model("assets.Rendition")
    rendition = (
        Rendition.objects.filter(asset_id=asset_id, type="NGFF")
        .exclude(stored_path="")
        .order_by("-path_exists", "created_at")
        .first()
    )
    if rendition is not None:
        path = _rendition_file_path(rendition, required=required)
        if path is not None:
            return StoragePath(
                path.relpath,
                path_type="dir",
                required=required,
                lease_required=True,
            )
    return StoragePath(
        f"data/tmp/ngff/{asset_id}.zarr",
        path_type="dir",
        required=required,
        lease_required=True,
    )


def _segmentation_ngff_path(segmentation_id: str, *, required: bool = False) -> StoragePath | None:
    ImageSegmentation = _model("segmentation.ImageSegmentation")
    segmentation = (
        ImageSegmentation.objects.filter(id=segmentation_id).only("asset_id").first()
    )
    if segmentation is None or not getattr(segmentation, "asset_id", None):
        return None
    return _asset_ngff_path(str(segmentation.asset_id), required=required)


def _overlay_root_path(segmentation_id: str, *, required: bool = False) -> StoragePath:
    return StoragePath(
        f"data/tmp/segmentation_overlays/{segmentation_id}",
        path_type="dir",
        required=required,
        lease_required=True,
    )


def _adapted_model_dir(adapter_id: str, *, required: bool = False) -> StoragePath:
    return StoragePath(
        f"models/adapted/{adapter_id}",
        path_type="dir",
        required=required,
        lease_required=True,
    )


def _analysis_export_dir(analysis_run_id: str, *, required: bool = False) -> StoragePath:
    return StoragePath(
        f"exports/{analysis_run_id}",
        path_type="dir",
        required=required,
        lease_required=True,
    )


def _probability_map_paths(segmentation_id: str) -> list[StoragePath]:
    ProbabilityMap = _model("segmentation.ProbabilityMap")
    paths: list[StoragePath] = []
    for raw_value in ProbabilityMap.objects.filter(
        segmentation_id=segmentation_id
    ).values_list("file_path", flat=True):
        raw_text = str(raw_value or "").strip()
        if raw_text:
            paths.append(StoragePath(validate_storage_relpath(raw_text), required=False))
    return paths


def _dedupe(paths: list[StoragePath]) -> list[StoragePath]:
    by_key: dict[tuple[str, str], StoragePath] = {}
    for path in paths:
        key = (path.relpath, path.path_type)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = path
            continue
        by_key[key] = StoragePath(
            path.relpath,
            path_type=path.path_type,
            required=existing.required or path.required,
            lease_required=existing.lease_required or path.lease_required,
        )
    return sorted(by_key.values(), key=lambda item: item.relpath)


def input_paths_for_job(job) -> list[StoragePath]:
    payload = job.payload_json or {}
    paths: list[StoragePath] = []
    if job.type == JOB_TYPE_ENSURE_IMAGE_NGFF or job.type == JOB_TYPE_UPLOAD_IMAGE_PIPELINE:
        asset_id = str(payload.get("asset_id") or "").strip()
        if asset_id:
            image_path = _asset_file_path(asset_id)
            if image_path is not None:
                paths.append(image_path)
    elif job.type in {
        JOB_TYPE_RUN_SEGMENTATION_ROI,
        JOB_TYPE_RUN_SEGMENTATION_FULL,
        JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY,
        JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
    }:
        segmentation_id = str(payload.get("segmentation_id") or "").strip()
        if segmentation_id:
            image_path = _segmentation_image_path(segmentation_id)
            if image_path is not None:
                paths.append(image_path)
            ngff_path = _segmentation_ngff_path(segmentation_id)
            if ngff_path is not None:
                paths.append(ngff_path)
    return _dedupe(paths)


def output_paths_for_job(job) -> list[StoragePath]:
    payload = job.payload_json or {}
    paths: list[StoragePath] = []
    if job.type == JOB_TYPE_ENSURE_IMAGE_NGFF:
        asset_id = str(payload.get("asset_id") or "").strip()
        if asset_id:
            paths.append(_asset_ngff_path(asset_id, required=True))
    elif job.type == JOB_TYPE_UPLOAD_IMAGE_PIPELINE:
        asset_id = str(payload.get("asset_id") or "").strip()
        if asset_id:
            image_path = _asset_file_path(asset_id, required=False)
            if image_path is not None:
                paths.append(image_path)
            paths.append(_asset_ngff_path(asset_id, required=False))
    elif job.type == JOB_TYPE_REBUILD_SEGMENTATION_OVERLAY:
        segmentation_id = str(payload.get("segmentation_id") or "").strip()
        if segmentation_id:
            paths.append(_overlay_root_path(segmentation_id, required=True))
    elif job.type in {JOB_TYPE_RUN_SEGMENTATION_ROI, JOB_TYPE_RUN_SEGMENTATION_FULL}:
        segmentation_id = str(payload.get("segmentation_id") or "").strip()
        if segmentation_id:
            paths.extend(_probability_map_paths(segmentation_id))
            paths.append(
                StoragePath(
                    f"data/prob_maps/{segmentation_id}",
                    path_type="dir",
                    required=False,
                    lease_required=True,
                )
            )
            paths.append(
                StoragePath(
                    f"data/tmp/prob_maps/{segmentation_id}",
                    path_type="dir",
                    required=False,
                    lease_required=True,
                )
            )
    elif job.type == JOB_TYPE_TRAIN_ORGANELLE_ADAPTER:
        # TODO(quantem): the adapter id is minted by the fine-tuning job, so the
        # caller must put it in the payload for the write to be leased. Without
        # it the run is unleased, which is safe only because the pool is one
        # slot wide.
        adapter_id = str(payload.get("adapter_id") or "").strip()
        if adapter_id:
            paths.append(_adapted_model_dir(adapter_id, required=True))
    elif job.type == JOB_TYPE_RUN_ANALYSIS:
        analysis_run_id = str(payload.get("analysis_run_id") or "").strip()
        if analysis_run_id:
            paths.append(_analysis_export_dir(analysis_run_id, required=True))
    return _dedupe(paths)


def lease_paths_for_job(job) -> list[StoragePath]:
    return [path for path in output_paths_for_job(job) if path.lease_required]
