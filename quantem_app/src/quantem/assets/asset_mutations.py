from __future__ import annotations

import uuid

from django.core.files.uploadedfile import UploadedFile
from django.db import transaction
from django.utils import timezone

from quantem.assets.models import Asset, Rendition
from quantem.assets.serializers import serialize_asset_detail
from quantem.assets.utils import (
    extract_image_metadata,
    save_uploaded_file_to_path,
    validate_upload_file,
)
from quantem.core.config import DATA_DIR, UPLOADS_DIR
from quantem.core.local_storage import normalize_stored_path_value
from quantem.jobs.constants import (
    JOB_TYPE_ENSURE_IMAGE_NGFF,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
    QUEUE_P2_UPLOAD,
)
from quantem.jobs.models import Job

PIXEL_SIZE_FIELDS = ("pixel_size_nm", "pixel_size_nm_z")


def parse_pixel_size_nm(value) -> float | None:
    """Coerce a user-supplied pixel size to a positive float (or ``None``).

    Blank input means "unknown"; a non-numeric or non-positive value is a hard
    error, because a wrong pixel size silently corrupts every analysis number.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"Pixel size must be a number, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError("Pixel size must be greater than zero.")
    return parsed


def create_uploaded_asset(
    *,
    uploaded_file: UploadedFile,
    display_name: str | None = None,
    pixel_size_nm=None,
    notes: str | None = None,
    segment_mito: bool = False,
    segment_er: bool = False,
    segment_nucleus: bool = False,
    segment_ld: bool = False,
    swallow_enqueue_errors: bool = False,
) -> dict:
    is_valid, error_message = validate_upload_file(uploaded_file)
    if not is_valid:
        raise ValueError(error_message)

    pixel_size = parse_pixel_size_nm(pixel_size_nm)
    asset_id = uuid.uuid4()
    original_filename = uploaded_file.name
    display_name = display_name or original_filename
    file_ext = (uploaded_file.name.rsplit(".", 1)[-1] if "." in uploaded_file.name else "tif").lower()
    staged_path = UPLOADS_DIR / f"{asset_id}.{file_ext}"
    save_uploaded_file_to_path(uploaded_file, staged_path)
    metadata = extract_image_metadata(staged_path)

    # A pixel size the user typed always wins; otherwise take whatever the file
    # declares. Leaving this null means no resampling and no calibrated
    # measurement, so it is worth reading even though many EM TIFFs omit it.
    if pixel_size is None:
        pixel_size = metadata.get("pixel_size_nm")

    # Free text the importer typed, stored on the column the library's search
    # already covers (``_filtered_asset_queryset`` matches display name, filename
    # and notes). The import form had a "Tags" box posting ``tag_names``, which
    # nothing here read: ``Asset`` has no tag field and there is no tag model in
    # the tree, so the text was accepted and dropped. ``notes`` is the field that
    # exists, and it is already patchable through :func:`update_asset` -- upload
    # was simply the one door that could not set it.
    notes_text = "" if notes is None else str(notes).strip()

    with transaction.atomic():
        asset = Asset.objects.create(
            id=asset_id,
            display_name=display_name,
            original_filename=original_filename,
            notes=notes_text,
            logical_width=int(metadata["width"]),
            logical_height=int(metadata["height"]),
            channels=int(metadata["channels"]),
            bit_depth=int(metadata["bit_depth"]),
            pixel_size_nm=pixel_size,
            preprocess_stage="ENCODING",
            preprocess_progress=0.0,
            preprocess_error="",
        )
        Rendition.objects.create(
            asset=asset,
            type=Rendition.TYPE_FULL,
            storage_root="DATA_DIR",
            stored_path=normalize_stored_path_value(staged_path, relative_to=DATA_DIR),
            path_exists=staged_path.exists(),
            is_directory=False,
            stored_width=int(metadata["width"]),
            stored_height=int(metadata["height"]),
            stored_channels=int(metadata["channels"]),
            stored_bit_depth=int(metadata["bit_depth"]),
            metadata={
                "upload_state": "staged",
                "original_filename": original_filename,
                # Asset.raw_metadata/normalized_metadata were corpus-curation
                # fields and are gone; the source file's own metadata belongs to
                # the rendition that was read from.
                "source_metadata": _json_safe_metadata(metadata),
            },
        )

    try:
        Job.enqueue(
            job_type=JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
            payload={
                "asset_id": str(asset.id),
                "segment_mito": segment_mito,
                "segment_er": segment_er,
                "segment_nucleus": segment_nucleus,
                "segment_ld": segment_ld,
            },
            priority="high",
            resource_class="cpu",
            queue_name=QUEUE_P2_UPLOAD,
            tags=[f"asset:{asset.id}"],
        )
    except Exception:
        if not swallow_enqueue_errors:
            raise

    asset.refresh_from_db()
    return serialize_asset_detail(asset)


def _json_safe_metadata(metadata: dict) -> dict:
    payload = dict(metadata)
    dtype = payload.get("dtype")
    if dtype is not None:
        payload["dtype"] = str(dtype)
    shape = payload.get("shape")
    if shape is not None:
        payload["shape"] = [int(value) for value in shape]
    return payload


def update_asset(asset: Asset, payload: dict) -> dict:
    allowed = {"display_name", "notes", *PIXEL_SIZE_FIELDS}
    updates = {key: payload[key] for key in allowed if key in payload}
    for field in PIXEL_SIZE_FIELDS:
        if field in updates:
            updates[field] = parse_pixel_size_nm(updates[field])
    if not updates:
        return serialize_asset_detail(asset)
    with transaction.atomic():
        for field, value in updates.items():
            setattr(asset, field, value)
        asset.save(update_fields=[*updates.keys(), "updated_at"])

    asset.refresh_from_db()
    return serialize_asset_detail(asset)


def enqueue_ngff_for_asset(asset: Asset) -> Job:
    active_job = (
        Job.objects.filter(
            type=JOB_TYPE_ENSURE_IMAGE_NGFF,
            status__in={"PENDING", "RUNNING", "RETRY"},
            payload_json__asset_id=str(asset.id),
        )
        .order_by("-created_at")
        .first()
    )
    if active_job is not None:
        return active_job
    return Job.enqueue(
        job_type=JOB_TYPE_ENSURE_IMAGE_NGFF,
        payload={"asset_id": str(asset.id)},
        priority="high",
        resource_class="cpu",
        queue_name=QUEUE_P2_UPLOAD,
        tags=[f"asset:{asset.id}"],
    )


def tombstone_asset(asset: Asset) -> None:
    if asset.lifecycle_status == Asset.LIFECYCLE_DELETED:
        return

    active_jobs = Job.objects.filter(
        status__in={"PENDING", "RUNNING", "RETRY"},
        payload_json__asset_id=str(asset.id),
    )
    for job in active_jobs:
        job.cancel_requested = True
        if job.status in {"PENDING", "RETRY"}:
            job.status = "CANCELLED"
            job.finished_at = timezone.now()
            job.message = "cancelled"
        job.save(
            update_fields=[
                "cancel_requested",
                "status",
                "finished_at",
                "message",
                "updated_at",
            ]
        )

    asset.lifecycle_status = Asset.LIFECYCLE_DELETED
    asset.deleted_at = timezone.now()
    asset.save(update_fields=["lifecycle_status", "deleted_at", "updated_at"])
