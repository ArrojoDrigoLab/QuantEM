"""
Background tasks for image preprocessing.

These are plain Python helpers invoked by the DB job handlers.
"""

import logging
import os
import shutil
import time
from pathlib import Path

from quantem.core.config import IMAGES_DIR
from quantem.core.local_storage import normalize_stored_path_value
from quantem.segmentation.roi_selection import select_roi_for_image

from .asset_openable import get_asset_openable, upsert_ngff_rendition
from .models import Asset, ImageROI, Rendition
from .ngff import ensure_ngff_for_image
from .preprocess_status import set_stage
from .roi_state import get_active_roi_for_asset
from .utils import (
    convert_png_to_8bit_grayscale,
    convert_tiff_to_png,
    create_roi_image_from_image,
)

logger = logging.getLogger(__name__)

ROI_SIZE_DEFAULT = int(os.environ.get("ROI_SIZE", "3000"))
ROI_MIN_IMAGE_SIZE = int(os.environ.get("ROI_MIN_IMAGE_SIZE", "6000"))


def _get_asset_or_none(asset_id: str) -> Asset | None:
    try:
        asset = Asset.objects.get(id=asset_id)
    except Asset.DoesNotExist:
        logger.warning(
            "Asset %s not found - likely deleted after task was queued. Skipping.",
            asset_id,
        )
        return None
    if asset.preprocess_stage == "CANCELLED":
        logger.info("Preprocessing for asset %s was cancelled, skipping", asset_id)
        return None
    return asset


def encode_asset_full_to_png_task(asset_id: str) -> None:
    encode_asset_full_to_png(asset_id)


def _is_canonical_image_file(path: Path) -> bool:
    """True when ``path`` already lives in the canonical image store.

    The encode step is re-entrant: on a job retry the FULL rendition already
    points at ``IMAGES_DIR/<asset>/<stem>.png``, and re-encoding it would be
    pointless work. A *staged* upload (``TMP_DIR/uploads``) is never canonical,
    even when it is itself a PNG.
    """

    try:
        Path(path).resolve().relative_to(IMAGES_DIR.resolve())
    except (ValueError, OSError):
        return False
    return True


def encode_asset_full_to_png(asset_id: str) -> None:
    """
    Convert a canonical Asset FULL rendition to the compressed PNG working form.

    Sources are TIFF or PNG (the only formats the upload API accepts): a TIFF is
    decoded and windowed to 8-bit grayscale, a PNG is re-saved into the same
    canonical 8-bit grayscale form. This updates the existing FULL rendition in
    place.
    """

    asset = _get_asset_or_none(asset_id)
    if asset is None:
        return
    set_stage(asset, "ENCODING", progress=0.0, error="")

    try:
        openable = get_asset_openable(asset)
    except Exception as exc:
        set_stage(asset, "FAILED", progress=0.0, error=f"Missing full image rendition: {exc}")
        return

    if openable.rendition.type != Rendition.TYPE_FULL:
        set_stage(
            asset,
            "FAILED",
            progress=0.0,
            error="Upload preprocessing requires a FULL rendition.",
        )
        return

    file_path = openable.path
    if not file_path.exists():
        set_stage(asset, "FAILED", progress=0.0, error="Missing source file for upload.")
        return

    is_png_source = file_path.suffix.lower() == ".png"
    if is_png_source and _is_canonical_image_file(file_path):
        set_stage(asset, "ENCODING", progress=55.0, error="")
        return

    disk_usage = shutil.disk_usage(str(IMAGES_DIR))
    estimated_bytes = int(openable.width * openable.height)
    min_free = int(os.environ.get("ENCODE_MIN_FREE_BYTES", str(512 * 1024 * 1024)))
    required = int(estimated_bytes * 1.2)
    if disk_usage.free < max(min_free, required):
        raise ValueError("Insufficient free disk space")

    last_update = 0.0

    def progress_callback(progress: float, message: str) -> None:
        del message
        nonlocal last_update
        now = time.time()
        if now - last_update < 1.0:
            return
        last_update = now
        set_stage(asset, "ENCODING", progress=progress, error="")

    metadata = {
        "width": int(openable.width),
        "height": int(openable.height),
        "channels": int(openable.channels),
        "bit_depth": int(openable.bit_depth),
    }
    original_stem = (asset.original_filename or asset.display_name or str(asset.id)).split(".")[0]
    target_png_path = IMAGES_DIR / str(asset.id) / f"{original_stem}.png"
    target_png_path = target_png_path.parent / f"{target_png_path.stem}.png"

    if is_png_source:
        # A staged PNG upload: no decode/window pass, just canonicalize it to
        # 8-bit grayscale in the image store so every downstream reader sees the
        # same single-channel form it gets from a TIFF.
        convert_png_to_8bit_grayscale(file_path, target_png_path)
    else:
        convert_tiff_to_png(
            file_path,
            metadata,
            target_file_path=target_png_path,
            progress_callback=progress_callback,
        )

    if file_path != target_png_path and file_path.exists():
        file_path.unlink()

    relative_path = normalize_stored_path_value(target_png_path, relative_to=IMAGES_DIR.parent)
    asset.channels = 1
    asset.bit_depth = 8
    asset.save(update_fields=["channels", "bit_depth", "updated_at"])
    Rendition.objects.filter(id=openable.rendition.id).update(
        storage_root="DATA_DIR",
        stored_path=relative_path,
        path_exists=target_png_path.exists(),
        is_directory=False,
        stored_width=openable.width,
        stored_height=openable.height,
        stored_channels=1,
        stored_bit_depth=8,
    )
    set_stage(asset, "ENCODING", progress=55.0, error="")


def ensure_ngff_for_asset_task(asset_id: str) -> None:
    asset = _get_asset_or_none(asset_id)
    if asset is None:
        return
    logger.info("Asset %s: NGFF stage started", asset_id)
    openable = get_asset_openable(asset)
    ngff_root = ensure_ngff_for_image(openable)
    upsert_ngff_rendition(asset, ngff_root, openable)
    logger.info("Asset %s: NGFF generation completed", asset_id)


def ensure_roi_for_asset_task(asset_id: str) -> ImageROI | None:
    asset = _get_asset_or_none(asset_id)
    if asset is None:
        return None

    existing_roi = get_active_roi_for_asset(asset)
    if existing_roi:
        return existing_roi

    openable = get_asset_openable(asset)
    roi_size = ROI_SIZE_DEFAULT
    if openable.width * openable.height >= ROI_MIN_IMAGE_SIZE**2:
        roi_result = select_roi_for_image(
            image=openable,
            roi_size=roi_size,
        )
        return create_roi_image_from_image(
            openable,
            x=roi_result.x,
            y=roi_result.y,
            width=roi_result.width,
            height=roi_result.height,
            source="AUTO",
        )
    return None
