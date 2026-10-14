"""
Generic Probability Map Persistence
=====================================

Save, load, and check probability maps using Django models.
Parameterized by prefix (e.g. "er", "mito") and generated_flag.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import zarr
from numcodecs import Blosc
from PIL import Image

from quantem.assets.asset_openable import get_asset_openable
from quantem.assets.models import ImageROI
from quantem.core.config import PROB_MAPS_DIR, STORAGE_DIR, TMP_DIR
from quantem.core.local_storage import StorageError, storage_path
from quantem.inference.resample import quantize_probability
from quantem.jobs.constants import (
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
)
from quantem.jobs.models import Job
from quantem.segmentation.models import ImageSegmentation, ProbabilityMap
from quantem.segmentation.prob_maps.io import resolve_probability_map_path
from quantem.segmentation.prob_maps.preview import (
    probability_preview_path,
    save_probability_preview,
)

logger = logging.getLogger(__name__)

# Above this size, PNG's whole-image decode is not an acceptable working
# representation. The Zarr store contains the identical uint8 levels in
# independently readable chunks; this is a storage change, not a quantisation
# or algorithm change.
PROBABILITY_MAP_ZARR_MIN_PIXELS = 256 * 1024 * 1024
PROBABILITY_MAP_CHUNK_SIZE = 2048
_PROBABILITY_MAP_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.NOSHUFFLE)

_ACTIVE_MAP_READER_STATUSES = ("PENDING", "RETRY", "RUNNING")
_MAP_READER_JOB_TYPES = (
    JOB_TYPE_REEXTRACT_AT_INCLUDE_LEVEL,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
)


def probability_map_reader_active(segmentation: ImageSegmentation) -> bool:
    """Whether queued/running work can still be reading this result map."""
    jobs = Job.objects.filter(
        type__in=_MAP_READER_JOB_TYPES,
        status__in=_ACTIVE_MAP_READER_STATUSES,
    ).values_list("payload_json", flat=True)
    segmentation_id = str(segmentation.id)
    asset_id = str(segmentation.asset_id or "")
    for payload in jobs.iterator():
        if not isinstance(payload, dict):
            continue
        if str(payload.get("segmentation_id") or "") == segmentation_id:
            return True
        asset_ids = payload.get("asset_ids")
        if (
            asset_id
            and isinstance(asset_ids, list)
            and asset_id in {str(value) for value in asset_ids}
        ):
            return True
    return False


#: ROI maps are canonical on their own. The composited full-image raster is a
#: convenience for small images only; allocating a gigapixel canvas after an
#: otherwise small ROI run defeats the point of an ROI run.
MAX_ROI_COMPOSITE_MEGAPIXELS = 512.0


def _probability_map_name(prefix: str, model_name: str) -> str:
    return f"{prefix.upper()}_{model_name}"


def _prob_map_storage_dir(
    segmentation: ImageSegmentation,
    roi_id: str | None,
) -> Path:
    if roi_id:
        return TMP_DIR / "prob_maps" / str(segmentation.id) / str(roi_id)
    return PROB_MAPS_DIR / str(segmentation.id)


def _prob_map_png_path(
    segmentation: ImageSegmentation,
    model_name: str,
    prefix: str,
    roi_id: str | None,
) -> Path:
    storage_dir = _prob_map_storage_dir(segmentation, roi_id)
    return storage_dir / f"{prefix}_{model_name.lower()}_prob.png"


def get_prob_map_file_path(
    segmentation: ImageSegmentation,
    model_name: str,
    prefix: str,
    roi_id: str | None = None,
) -> Path:
    """Get the file path for a probability map.

    Args:
        segmentation: The ImageSegmentation instance.
        model_name: Name of the model (e.g. "ResNet34", "MitoNet").
        prefix: Filename prefix (e.g. "er", "mito").
        roi_id: Optional ROI ID for ROI-scoped maps.
    """
    storage_dir = _prob_map_storage_dir(segmentation, roi_id)
    stem = f"{prefix}_{model_name.lower()}_prob"
    chunked = storage_dir / f"{stem}.zarr"
    png = storage_dir / f"{stem}.png"
    if chunked.exists() and png.exists():
        # A cloned record may still own the previous storage format.
        paths = {
            str(path.relative_to(STORAGE_DIR)).replace("\\", "/"): path for path in (chunked, png)
        }
        latest = (
            ProbabilityMap.objects.filter(
                segmentation=segmentation,
                name=_probability_map_name(prefix, model_name),
                file_path__in=paths,
            )
            .order_by("-updated_at", "-created_at")
            .values_list("file_path", flat=True)
            .first()
        )
        if latest is not None:
            return paths[latest]
    if chunked.exists():
        return chunked
    return png


def get_prob_map_chunked_path(
    segmentation: ImageSegmentation,
    model_name: str,
    prefix: str,
    roi_id: str | None = None,
) -> Path:
    """Canonical chunked path for a large probability map."""
    storage_dir = _prob_map_storage_dir(segmentation, roi_id)
    return storage_dir / f"{prefix}_{model_name.lower()}_prob.zarr"


def get_composite_prob_map_file_path(
    segmentation: ImageSegmentation,
    model_name: str,
    prefix: str,
) -> Path:
    """Get the path for ROI-composited full-view probability maps.

    Composite maps are visualization artifacts and must not be treated as
    canonical full-image inference caches.
    """
    storage_dir = PROB_MAPS_DIR / str(segmentation.id) / "composite"
    filename = f"{prefix}_{model_name.lower()}_prob.png"
    return storage_dir / filename


def _metadata_describes_roi(metadata: object, roi: ImageROI) -> bool:
    """Whether a legacy or uploaded map records this ROI's rectangle.

    Run-created maps now have both an ROI-specific path and ``roi_id`` metadata.
    Older and uploaded ROI maps only recorded their rectangle, so retain that
    comparison when reclaiming a completed ROI's cache.
    """
    if not isinstance(metadata, dict):
        return False
    if str(metadata.get("roi_id") or "") == str(roi.id):
        return True
    window = metadata.get("roi")
    if not isinstance(window, dict):
        return False
    try:
        return (
            int(window["x"]) == int(roi.x)
            and int(window["y"]) == int(roi.y)
            and int(window["width"]) == int(roi.width)
            and int(window["height"]) == int(roi.height)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _unlink_unshared_probability_map_file(prob_map: ProbabilityMap) -> None:
    """Remove one file unless another surviving map record still uses it."""
    if ProbabilityMap.objects.filter(file_path=prob_map.file_path).exists():
        return
    try:
        source_path = storage_path(prob_map.file_path)
        if source_path.is_dir():
            shutil.rmtree(source_path, ignore_errors=True)
        else:
            source_path.unlink(missing_ok=True)
        probability_preview_path(source_path).unlink(missing_ok=True)
    except (OSError, StorageError):
        logger.warning(
            "Could not remove probability-map file %s after reclaiming its record.",
            prob_map.file_path,
            exc_info=True,
        )


def _remove_superseded_storage_format(
    segmentation: ImageSegmentation,
    model_name: str,
    prefix: str,
    roi_id: str | None,
    *,
    kept_path: Path,
) -> None:
    """Reclaim the PNG/Zarr counterpart after the replacement is durable."""
    png_path = _prob_map_png_path(segmentation, model_name, prefix, roi_id)
    chunked_path = get_prob_map_chunked_path(segmentation, model_name, prefix, roi_id)
    counterpart = png_path if kept_path == chunked_path else chunked_path
    if not counterpart.exists():
        return
    relative = str(counterpart.relative_to(STORAGE_DIR)).replace("\\", "/")
    ProbabilityMap.objects.filter(
        segmentation=segmentation,
        name=_probability_map_name(prefix, model_name),
        file_path=relative,
    ).delete()
    if ProbabilityMap.objects.filter(file_path=relative).exists():
        return
    if counterpart.is_dir():
        shutil.rmtree(counterpart, ignore_errors=True)
    else:
        counterpart.unlink(missing_ok=True)
    probability_preview_path(counterpart).unlink(missing_ok=True)


def _remove_directory_if_unreferenced(path: Path) -> None:
    """Remove an owned map directory unless a surviving DB row still uses it."""
    try:
        relative_prefix = str(path.relative_to(STORAGE_DIR)).replace("\\", "/")
    except ValueError:
        logger.warning("Refusing to remove probability-map directory outside storage: %s", path)
        return
    if ProbabilityMap.objects.filter(file_path__startswith=f"{relative_prefix}/").exists():
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:  # pragma: no cover - rmtree(ignore_errors) rarely raises
        logger.warning("Could not remove probability-map directory %s.", path, exc_info=True)


def delete_probability_maps_for_roi(
    segmentation: ImageSegmentation,
    roi: ImageROI,
) -> int:
    """Reclaim maps scoped to one ROI after that ROI is marked done.

    A full-image map remains available for any unfinished parts of the image.
    The composited full-view cache is discarded for matching models: it cannot
    remove just one ROI's pixels without allocating another full-image canvas,
    and it is never a canonical inference cache.
    """
    if probability_map_reader_active(segmentation):
        logger.info(
            "Deferred ROI probability-map cleanup for segmentation %s while a "
            "threshold reader is active.",
            segmentation.id,
        )
        return 0
    roi_dir = TMP_DIR / "prob_maps" / str(segmentation.id) / str(roi.id)
    roi_prefix = str(roi_dir.relative_to(STORAGE_DIR)).replace("\\", "/")
    maps = list(ProbabilityMap.objects.filter(segmentation=segmentation))
    roi_maps = [
        prob_map
        for prob_map in maps
        if prob_map.file_path.replace("\\", "/").startswith(f"{roi_prefix}/")
        or _metadata_describes_roi(prob_map.metadata, roi)
    ]
    model_names = {prob_map.name for prob_map in roi_maps}
    composite_maps = [
        prob_map
        for prob_map in maps
        if prob_map.name in model_names
        and isinstance(prob_map.metadata, dict)
        and prob_map.metadata.get("composite") is True
    ]
    to_delete = {prob_map.id: prob_map for prob_map in [*roi_maps, *composite_maps]}
    if not to_delete:
        _remove_directory_if_unreferenced(roi_dir)
        return 0

    records = list(to_delete.values())
    ProbabilityMap.objects.filter(id__in=list(to_delete)).delete()
    for record in records:
        _unlink_unshared_probability_map_file(record)
    _remove_directory_if_unreferenced(roi_dir)
    composite_dir = PROB_MAPS_DIR / str(segmentation.id) / "composite"
    _remove_directory_if_unreferenced(composite_dir)
    return len(records)


def delete_probability_maps_for_segmentation(segmentation: ImageSegmentation) -> int:
    """Reclaim every reusable map for a completed segmentation.

    This removes only probability-map records and their cache files. Confirmed
    objects, stored masks, overlays, and analysis outputs are deliberately not
    part of this cleanup.
    """
    if probability_map_reader_active(segmentation):
        logger.info(
            "Deferred probability-map cleanup for segmentation %s while a "
            "threshold reader is active.",
            segmentation.id,
        )
        return 0
    records = list(ProbabilityMap.objects.filter(segmentation=segmentation))
    if records:
        ProbabilityMap.objects.filter(id__in=[record.id for record in records]).delete()
        for record in records:
            _unlink_unshared_probability_map_file(record)
    segmentation_id = str(segmentation.id)
    _remove_directory_if_unreferenced(PROB_MAPS_DIR / segmentation_id)
    _remove_directory_if_unreferenced(TMP_DIR / "prob_maps" / segmentation_id)
    return len(records)


def _latest_map_for_file_path(
    segmentation: ImageSegmentation,
    model_name: str,
    prefix: str,
    relative_path: str,
) -> ProbabilityMap | None:
    return (
        ProbabilityMap.objects.filter(
            segmentation=segmentation,
            name=_probability_map_name(prefix, model_name),
            file_path=relative_path,
        )
        .order_by("-updated_at")
        .first()
    )


def _is_stale_composite_full_cache(
    segmentation: ImageSegmentation,
    model_name: str,
    prefix: str,
    file_path: Path,
) -> bool:
    """True when the canonical full-image path currently points to a composite map."""
    relative_path = str(file_path.relative_to(STORAGE_DIR)).replace("\\", "/")
    latest = _latest_map_for_file_path(segmentation, model_name, prefix, relative_path)
    if latest is None:
        return False
    metadata = latest.metadata if isinstance(latest.metadata, dict) else {}
    if metadata.get("composite") is True:
        logger.info(
            "Ignoring stale composite probability-map cache for full-image inference "
            "(segmentation=%s, model=%s, path=%s)",
            segmentation.id,
            model_name,
            relative_path,
        )
        return True
    return False


def prob_map_file_exists(
    segmentation: ImageSegmentation,
    model_name: str,
    prefix: str,
    roi_id: str | None = None,
) -> bool:
    """Check if a probability map file exists on disk."""
    file_path = get_prob_map_file_path(segmentation, model_name, prefix, roi_id)
    if not file_path.exists():
        return False
    return not (
        roi_id is None
        and _is_stale_composite_full_cache(segmentation, model_name, prefix, file_path)
    )


@contextmanager
def _publish_probability_map(
    file_path: Path,
    source: np.ndarray | zarr.Array,
    *,
    chunked: bool,
) -> Iterator[np.ndarray | zarr.Array]:
    """Restore the previous map if publishing its bytes or record fails."""
    token = uuid4().hex
    temporary = file_path.with_name(f".{file_path.stem}-{token}.tmp{file_path.suffix}")
    backup = file_path.with_name(f".{file_path.stem}-{token}.bak{file_path.suffix}")
    published = False
    succeeded = False
    try:
        if chunked:
            height, width = source.shape
            stored = zarr.open_array(
                str(temporary),
                mode="w",
                shape=(height, width),
                chunks=(
                    min(PROBABILITY_MAP_CHUNK_SIZE, height),
                    min(PROBABILITY_MAP_CHUNK_SIZE, width),
                ),
                dtype=np.uint8,
                compressor=_PROBABILITY_MAP_COMPRESSOR,
                zarr_format=2,
            )
            for y0 in range(0, height, PROBABILITY_MAP_CHUNK_SIZE):
                y1 = min(height, y0 + PROBABILITY_MAP_CHUNK_SIZE)
                for x0 in range(0, width, PROBABILITY_MAP_CHUNK_SIZE):
                    x1 = min(width, x0 + PROBABILITY_MAP_CHUNK_SIZE)
                    block = np.asarray(source[y0:y1, x0:x1])
                    stored[y0:y1, x0:x1] = (
                        block if block.dtype == np.uint8 else quantize_probability(block)
                    )
        else:
            array = np.asarray(source)
            stored = array if array.dtype == np.uint8 else quantize_probability(array)
            Image.fromarray(stored, mode="L").save(temporary)

        if file_path.exists():
            os.replace(file_path, backup)
        os.replace(temporary, file_path)
        published = True
        yield zarr.open_array(str(file_path), mode="r") if chunked else stored
        succeeded = True
    except BaseException:
        if published:
            if file_path.is_dir():
                shutil.rmtree(file_path)
            else:
                file_path.unlink(missing_ok=True)
        if backup.exists():
            os.replace(backup, file_path)
        raise
    finally:
        for obsolete in (temporary, backup) if succeeded else (temporary,):
            if obsolete.is_dir():
                shutil.rmtree(obsolete, ignore_errors=True)
            else:
                obsolete.unlink(missing_ok=True)


def save_probability_map(
    segmentation: ImageSegmentation,
    model_name: str,
    prob_data: np.ndarray | zarr.Array,
    prefix: str,
    generated_flag: str,
    roi_id: str | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> ProbabilityMap:
    """Save a probability map to disk and create a ProbabilityMap record.

    Args:
        segmentation: The ImageSegmentation instance.
        model_name: Name of the model (e.g. "ResNet34", "Ensemble").
        prob_data: Probability map array, values in [0, 1].
        prefix: Filename prefix (e.g. "er", "mito").
        generated_flag: Metadata flag key (e.g. "er_generated").
        roi_id: Optional ROI ID.

    Returns:
        Created ProbabilityMap instance.
    """
    shape = getattr(prob_data, "shape", ())
    if len(shape) != 2 or any(int(dimension) < 1 for dimension in shape):
        raise ValueError(f"Probability map must be 2D, got shape {shape}")
    height, width = int(shape[0]), int(shape[1])
    use_chunked = height * width >= PROBABILITY_MAP_ZARR_MIN_PIXELS
    file_path = (
        get_prob_map_chunked_path(segmentation, model_name, prefix, roi_id)
        if use_chunked
        else _prob_map_png_path(segmentation, model_name, prefix, roi_id)
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)

    relative_path = str(file_path.relative_to(STORAGE_DIR)).replace("\\", "/")
    metadata: dict[str, object] = {"model_type": model_name, generated_flag: True}
    if extra_metadata:
        metadata.update(extra_metadata)

    with _publish_probability_map(file_path, prob_data, chunked=use_chunked) as prob_uint8:
        prob_map = ProbabilityMap.objects.create(
            segmentation=segmentation,
            name=_probability_map_name(prefix, model_name),
            file_path=relative_path,
            metadata=metadata,
        )
    try:
        save_probability_preview(file_path, prob_uint8)
    except Exception:  # noqa: BLE001 - the authoritative map is already durable
        # The authoritative full-resolution map is already safe. A preview can
        # be retried lazily without turning successful inference into failure.
        logger.warning(
            "Could not write the browser probability preview for %s.",
            file_path,
            exc_info=True,
        )

    _remove_superseded_storage_format(
        segmentation,
        model_name,
        prefix,
        roi_id,
        kept_path=file_path,
    )

    # Composite ROI prob map into a full-image version for viewing
    if roi_id:
        _composite_roi_into_full_image_prob_map(
            segmentation=segmentation,
            model_name=model_name,
            prob_uint8=prob_uint8,
            prefix=prefix,
            generated_flag=generated_flag,
            roi_id=roi_id,
        )

    return prob_map


def load_prob_map_from_file(prob_map: ProbabilityMap) -> np.ndarray:
    """Load a probability map from its file path.

    Returns:
        Probability map as float32 array in [0, 1].
    """
    file_path = resolve_probability_map_path(prob_map)
    if not file_path.exists():
        raise FileNotFoundError(f"Probability map file not found: {file_path}")

    if file_path.is_dir():
        return np.asarray(zarr.open_array(str(file_path), mode="r"), dtype=np.float32) / 255.0
    img = Image.open(file_path)
    return np.array(img, dtype=np.float32) / 255.0


def _composite_roi_into_full_image_prob_map(
    segmentation: ImageSegmentation,
    model_name: str,
    prob_uint8: np.ndarray,
    prefix: str,
    generated_flag: str,
    roi_id: str,
) -> None:
    """Composite an ROI probability map into a full-image version.

    Creates or updates a full-image probability map by inserting the ROI
    crop at its correct position within a canvas of the parent image size.
    """
    try:
        roi = ImageROI.objects.select_related("asset").get(id=roi_id)
    except ImageROI.DoesNotExist:
        logger.warning("Cannot composite prob map: ImageROI %s not found", roi_id)
        return

    if roi.asset_id is None:
        logger.warning("Cannot composite prob map: ImageROI %s has no asset", roi_id)
        return
    parent = get_asset_openable(roi.asset)
    full_h, full_w = parent.height, parent.width
    full_megapixels = (full_h * full_w) / 1e6
    if full_megapixels > MAX_ROI_COMPOSITE_MEGAPIXELS:
        logger.info(
            "Skipping ROI probability-map composite for segmentation %s: parent image is %.0f MP "
            "(ROI map remains available at its native window).",
            segmentation.id,
            full_megapixels,
        )
        return

    # Composite path is separate from canonical full-image cache path.
    composite_path = get_composite_prob_map_file_path(segmentation, model_name, prefix)
    composite_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing full-image canvas or create black
    if composite_path.exists():
        canvas = np.array(Image.open(composite_path), dtype=np.uint8)
    else:
        canvas = np.zeros((full_h, full_w), dtype=np.uint8)

    # Insert ROI data
    y1, y2 = roi.y, roi.y + roi.height
    x1, x2 = roi.x, roi.x + roi.width
    # Clip to canvas bounds
    cy1 = max(y1, 0)
    cy2 = min(y2, full_h)
    cx1 = max(x1, 0)
    cx2 = min(x2, full_w)
    ry1 = cy1 - roi.y
    ry2 = ry1 + (cy2 - cy1)
    rx1 = cx1 - roi.x
    rx2 = rx1 + (cx2 - cx1)
    canvas[cy1:cy2, cx1:cx2] = prob_uint8[ry1:ry2, rx1:rx2]

    # Save composited image
    Image.fromarray(canvas, mode="L").save(composite_path)

    # Create or update DB record for the full-image prob map
    full_name = _probability_map_name(prefix, model_name)
    relative_path = str(composite_path.relative_to(STORAGE_DIR)).replace("\\", "/")

    # SQLite does not support JSONField `contains` lookups; use key transforms.
    lookup_kwargs = {
        "segmentation": segmentation,
        "name": full_name,
        "metadata__composite": True,
        f"metadata__{generated_flag}": True,
    }
    ProbabilityMap.objects.update_or_create(
        **lookup_kwargs,
        defaults={
            "file_path": relative_path,
            "metadata": {
                "model_type": model_name,
                generated_flag: True,
                "composite": True,
            },
        },
    )
    logger.info(
        "Composited ROI prob map into full image: %s for segmentation %s",
        full_name,
        segmentation.id,
    )


def load_prob_map_from_path(
    segmentation: ImageSegmentation,
    model_name: str,
    prefix: str,
    roi_id: str | None = None,
) -> np.ndarray | None:
    """Load a probability map from the expected file path.

    Returns None if the file does not exist.
    """
    file_path = get_prob_map_file_path(segmentation, model_name, prefix, roi_id)
    if not file_path.exists():
        return None
    if roi_id is None and _is_stale_composite_full_cache(
        segmentation, model_name, prefix, file_path
    ):
        return None

    data = load_prob_map_uint8_from_path(segmentation, model_name, prefix, roi_id)
    return None if data is None else np.asarray(data, dtype=np.float32) / 255.0


def load_prob_map_uint8_from_path(
    segmentation: ImageSegmentation,
    model_name: str,
    prefix: str,
    roi_id: str | None = None,
) -> object | None:
    """Load the stored native bytes without allocating a full float copy."""
    file_path = get_prob_map_file_path(segmentation, model_name, prefix, roi_id)
    if not file_path.exists():
        return None
    if roi_id is None and _is_stale_composite_full_cache(
        segmentation, model_name, prefix, file_path
    ):
        return None
    if file_path.is_dir():
        array = zarr.open_array(str(file_path), mode="r")
        if len(array.shape) != 2 or np.dtype(array.dtype) != np.dtype(np.uint8):
            raise ValueError(
                f"Stored probability map must be 2D uint8, got {array.shape} {array.dtype}"
            )
        return array
    with Image.open(file_path) as img:
        return np.array(img, dtype=np.uint8)
