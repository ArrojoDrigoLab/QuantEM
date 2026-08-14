"""
Helper functions for image processing and asset-backed ROI creation.

These functions are designed to be reusable across different views and tasks.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
from django.core.files.uploadedfile import UploadedFile
from django.db import transaction
from PIL import Image

try:
    import pyvips
except ImportError:
    pyvips = None

try:
    import psutil
except ImportError:
    psutil = None

# Disable decompression bomb check for large SEM/TEM images
# This is a legitimate use case for very large scientific images
Image.MAX_IMAGE_PIXELS = None

from quantem.assets.file_paths import get_file_absolute_path
from quantem.core.config import IMAGES_DIR, ROIS_DIR, UPLOADS_DIR

from .models import ImageROI
from .volume_readers import PNG_SUFFIXES, TIFF_SUFFIXES

PNG_COMPRESS_LEVEL = int(os.environ.get("IMAGE_PNG_COMPRESS_LEVEL", "3"))


def _upload_suffixes(reader_suffixes: set[str]) -> tuple[str, ...]:
    """Reader suffixes reduced to what ``Path.suffix`` can actually match.

    ``.ome.tif`` is dropped because ``Path("x.ome.tif").suffix`` is ``.tif``,
    which is already in the set; keeping it would only pad the user-facing
    "accepted formats" message with an extension nothing would ever compare
    equal to.
    """

    return tuple(sorted(s for s in reader_suffixes if s.count(".") == 1))


# Formats accepted by the upload API. v1 is TIFF + PNG only (owner ruling
# 2026-08-06). Derived from the reader suffix sets in ``volume_readers`` so the
# API can never advertise a format the readers would then refuse.
TIFF_UPLOAD_SUFFIXES = _upload_suffixes(TIFF_SUFFIXES)  # (".tif", ".tiff")
PNG_UPLOAD_SUFFIXES = _upload_suffixes(PNG_SUFFIXES)  # (".png",)
UPLOAD_SUFFIXES = TIFF_UPLOAD_SUFFIXES + PNG_UPLOAD_SUFFIXES


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _run_with_progress(
    operation: Callable[[], None],
    *,
    progress_callback: Callable[[float, str], None] | None,
    start_progress: float,
    message: str,
) -> None:
    errors: list[Exception] = []

    def _run() -> None:
        try:
            operation()
        except Exception as exc:  # pragma: no cover - exercised via caller
            errors.append(exc)

    if progress_callback is None:
        _run()
        if errors:
            raise errors[0]
        return

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    last_progress = start_progress
    while thread.is_alive():
        thread.join(timeout=1.0)
        last_progress = min(last_progress + 1.0, 95.0)
        progress_callback(last_progress, message)
    if errors:
        raise errors[0]


def _maybe_sync_png_save(target_file_path: Path) -> None:
    if not _env_flag("IMAGE_PNG_FORCE_OS_SYNC", default=False):
        return
    if not hasattr(os, "sync"):
        return
    try:
        logging.getLogger(__name__).info(
            "Forcing os.sync() after PNG save for %s because IMAGE_PNG_FORCE_OS_SYNC is enabled",
            target_file_path,
        )
        os.sync()
    except (OSError, AttributeError):
        return


def _native_max_for_bit_depth(bit_depth: int) -> float:
    return float((2**bit_depth) - 1)


def _normalize_array_to_uint8(values: np.ndarray, *, bit_depth: int) -> np.ndarray:
    """
    Reduce an integer microscopy plane to 8-bit by plain native bit-depth
    conversion: scale the native value range ``[0, 2**bit_depth - 1]`` linearly
    onto ``[0, 255]``.

    The data is never windowed/clipped nor stretched to its own min/max, so the
    8-bit output occupies whatever sub-range of 0..255 the input occupied (the
    full range if the input used the full native range, less otherwise). Any
    contrast adjustment happens later (e.g. tile generation), not here.
    """
    native_max = _native_max_for_bit_depth(bit_depth)
    scale = 255.0 / native_max if native_max > 0 else 1.0
    float_values = np.asarray(values, dtype=np.float32)
    scaled = float_values * scale
    return np.nan_to_num(np.clip(scaled, 0, 255), nan=0).astype(np.uint8)


def _vips_native_window(image_vips, *, bit_depth: int) -> tuple[float, float]:
    """Native-range window [0, 2**bit_depth - 1] for plain 8-bit bit-depth conversion."""
    return 0.0, _native_max_for_bit_depth(bit_depth)


def _prepare_vips_grayscale_uchar(
    image_vips,
    *,
    channels: int,
    bit_depth: int,
    logger: logging.Logger,
):
    """
    Convert a TIFF-backed vips image to the canonical 8-bit grayscale PNG view.

    This mirrors the NumPy path:
    - if the TIFF has multiple channels, keep only the first channel
    - if the TIFF uses >8-bit integer storage, scale the native range
      [0, 2**bit_depth - 1] into [0, 255] before writing PNG (no windowing or
      min/max stretching)
    """
    if image_vips.bands > 1:
        logger.info(
            "Streaming TIFF conversion: extracting first band from %s-band image",
            image_vips.bands,
        )
        image_vips = image_vips.extract_band(0)
    elif channels > 1:
        logger.info(
            "Streaming TIFF conversion: metadata reported %s channels, libvips exposed %s bands; using the available single band",
            channels,
            image_vips.bands,
        )

    if image_vips.format != "uchar":
        if bit_depth > 8:
            low, high = _vips_native_window(image_vips, bit_depth=bit_depth)
            scale = 255.0 / float(high - low)
            logger.info(
                "Streaming TIFF conversion: scaling %s-bit pixels into 8-bit grayscale "
                "by native bit-depth with low=%.3f high=%.3f scale=%.8f",
                bit_depth,
                low,
                high,
                scale,
            )
            image_vips = (
                image_vips.cast("float")
                .linear(scale, -low * scale)
                .clamp(min=0, max=255)
                .cast("uchar")
            )
        else:
            logger.info(
                "Streaming TIFF conversion: casting %s pixels to 8-bit grayscale",
                image_vips.format,
            )
            image_vips = image_vips.cast("uchar")
    return image_vips


def _vips_format_bit_depth(format_name: str) -> int:
    if format_name in {"uchar", "char"}:
        return 8
    if format_name in {"ushort", "short"}:
        return 16
    if format_name in {"uint", "int", "float"}:
        return 32
    if format_name == "double":
        return 64
    return 8


def _prepare_vips_png_grayscale_uchar(image_vips, *, logger: logging.Logger):
    """
    Convert a PNG-backed vips image to the canonical 8-bit grayscale PNG view.
    """
    if image_vips.bands > 1:
        logger.info(
            "PNG staging: extracting first band from %s-band image",
            image_vips.bands,
        )
        image_vips = image_vips.extract_band(0)

    if image_vips.format != "uchar":
        bit_depth = _vips_format_bit_depth(image_vips.format)
        if bit_depth > 8:
            low, high = _vips_native_window(image_vips, bit_depth=bit_depth)
            scale = 255.0 / float(high - low)
            logger.info(
                "PNG staging: scaling %s-bit pixels into 8-bit grayscale by native "
                "bit-depth with low=%.3f high=%.3f scale=%.8f",
                bit_depth,
                low,
                high,
                scale,
            )
            image_vips = (
                image_vips.cast("float")
                .linear(scale, -low * scale)
                .clamp(min=0, max=255)
                .cast("uchar")
            )
        else:
            image_vips = image_vips.cast("uchar")
    return image_vips


def _convert_tiff_to_png_via_vips(
    tiff_path: Path,
    metadata: dict,
    target_file_path: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    logger = logging.getLogger(__name__)
    step_start = time.time()

    if pyvips is None:
        raise RuntimeError("pyvips is unavailable")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    target_file_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(5.0, "opening tiff")
    open_start = time.time()
    source_image = pyvips.Image.new_from_file(str(tiff_path), access="random")
    open_elapsed = time.time() - open_start
    logger.info(
        "Opened TIFF via pyvips in %.2fs: %s (width=%s height=%s bands=%s format=%s)",
        open_elapsed,
        tiff_path,
        source_image.width,
        source_image.height,
        source_image.bands,
        source_image.format,
    )

    if progress_callback:
        progress_callback(25.0, "preparing grayscale pipeline")
    pipeline_start = time.time()
    png_ready_image = _prepare_vips_grayscale_uchar(
        source_image,
        channels=int(metadata.get("channels", 1) or 1),
        bit_depth=int(metadata.get("bit_depth", 8) or 8),
        logger=logger,
    )
    pipeline_elapsed = time.time() - pipeline_start
    logger.info(
        "Prepared streaming PNG pipeline in %.2fs (bands=%s format=%s)",
        pipeline_elapsed,
        png_ready_image.bands,
        png_ready_image.format,
    )

    if progress_callback:
        progress_callback(70.0, "saving png")
    save_start = time.time()

    def _save_png() -> None:
        png_ready_image.pngsave(
            str(target_file_path),
            compression=PNG_COMPRESS_LEVEL,
            bitdepth=8,
            interlace=False,
        )

    _run_with_progress(
        _save_png,
        progress_callback=progress_callback,
        start_progress=70.0,
        message="saving png",
    )
    save_elapsed = time.time() - save_start
    logger.info(
        "PNG save completed in %.2fs using libvips streaming decode + normalize + encode",
        save_elapsed,
    )

    _maybe_sync_png_save(target_file_path)

    verify_start = time.time()
    if not target_file_path.exists():
        raise OSError(f"PNG file was not created at {target_file_path}")
    file_size = target_file_path.stat().st_size
    if file_size == 0:
        raise OSError(f"PNG file is empty at {target_file_path}")
    verify_elapsed = time.time() - verify_start

    if progress_callback:
        progress_callback(98.0, "verifying png")
    total_time = time.time() - step_start
    logger.info(
        "PNG verified in %.2fs (%s bytes); total TIFF->PNG time %.2fs",
        verify_elapsed,
        file_size,
        total_time,
    )
    if progress_callback:
        progress_callback(100.0, "png saved")
    return target_file_path


def _convert_png_to_8bit_via_vips(source_path: Path, target_path: Path) -> None:
    logger = logging.getLogger(__name__)
    source_image = pyvips.Image.new_from_file(str(source_path), access="random")
    png_ready_image = _prepare_vips_png_grayscale_uchar(source_image, logger=logger)
    png_ready_image.pngsave(
        str(target_path),
        compression=PNG_COMPRESS_LEVEL,
        bitdepth=8,
        interlace=False,
    )


def _convert_png_to_8bit_via_pillow(source_path: Path, target_path: Path) -> None:
    with Image.open(source_path) as source_image:
        if source_image.mode in {"I;16", "I;16B", "I;16L"}:
            values = np.asarray(source_image, dtype=np.uint16)
            scaled = _normalize_array_to_uint8(values, bit_depth=16)
            canonical = Image.fromarray(scaled, mode="L")
        elif source_image.mode == "L":
            canonical = source_image.copy()
        else:
            canonical = source_image.convert("L")
        canonical.save(
            str(target_path),
            "PNG",
            compress_level=PNG_COMPRESS_LEVEL,
            optimize=False,
        )


def convert_png_to_8bit_grayscale(source_path: Path, target_path: Path) -> Path:
    """
    Save a source PNG as the app's canonical 8-bit grayscale PNG.
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if pyvips is not None:
        try:
            _convert_png_to_8bit_via_vips(source_path, target_path)
            _maybe_sync_png_save(target_path)
            return target_path
        except Exception:
            logging.getLogger(__name__).warning(
                "Streaming PNG canonicalization failed for %s; falling back to Pillow",
                source_path,
                exc_info=True,
            )

    _convert_png_to_8bit_via_pillow(source_path, target_path)
    _maybe_sync_png_save(target_path)
    return target_path


def _select_png_grayscale_plane(tiff_data: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Return a 2D array for the canonical grayscale PNG.

    Some importer paths pass lightweight metadata without the original TIFF
    shape. The fallback conversion must rely on the loaded array in that case
    instead of failing with KeyError("shape").
    """
    shape = tuple(metadata.get("shape") or tiff_data.shape)
    channels = int(metadata.get("channels", 1) or 1)

    if tiff_data.ndim == 2:
        return tiff_data

    if tiff_data.ndim == 3:
        if channels > 1:
            if len(shape) == 3 and shape[0] == channels and shape[-1] != channels:
                return tiff_data[0]
            return tiff_data[:, :, 0]
        return tiff_data[0]

    squeezed = np.squeeze(tiff_data)
    if squeezed.ndim == 2:
        return squeezed
    if squeezed.ndim == 3:
        return squeezed[0] if channels <= 1 else squeezed[:, :, 0]
    raise ValueError(f"Unsupported TIFF shape for PNG conversion: {tiff_data.shape}")


def _convert_tiff_to_png_via_numpy(
    tiff_path: Path,
    metadata: dict,
    target_file_path: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    logger = logging.getLogger(__name__)
    step_start = time.time()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Load TIFF data
    if progress_callback:
        progress_callback(5.0, "loading tiff")
    logger.info(f"Loading TIFF file into memory: {tiff_path}")
    if psutil:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        logger.info(f"Memory before load: {mem_before:.2f} MB")

    load_start = time.time()
    tiff_data = tifffile.imread(str(tiff_path))
    load_time = time.time() - load_start

    try:
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_used = mem_after - mem_before
        logger.info(
            f"TIFF loaded in {load_time:.2f}s, shape: {tiff_data.shape}, dtype: {tiff_data.dtype}, memory used: {mem_used:.2f} MB"
        )
    except NameError:
        logger.info(
            f"TIFF loaded in {load_time:.2f}s, shape: {tiff_data.shape}, dtype: {tiff_data.dtype}"
        )

    # Use actual dtype from loaded array (should match metadata, but safer)
    dtype = tiff_data.dtype
    channels = int(metadata.get("channels", 1) or 1)
    tiff_data = _select_png_grayscale_plane(tiff_data, metadata)

    # Convert to PIL Image for PNG conversion
    if progress_callback:
        progress_callback(35.0, "converting to png")
    logger.info(f"Converting array to PIL Image (channels={channels}, dtype={dtype})")
    convert_start = time.time()

    if channels > 1:
        logger.info("Extracted first channel from multi-channel image")
    dtype = tiff_data.dtype
    if dtype != np.uint8:
        # Canonical PNGs are 8-bit grayscale via native bit-depth conversion
        # (no windowing or min/max stretching) at save time.
        bit_depth = int(metadata.get("bit_depth", 8) or 8)
        logger.info(f"Normalizing {bit_depth}-bit to 8-bit")
        tiff_data = _normalize_array_to_uint8(tiff_data, bit_depth=bit_depth)
    pil_image = Image.fromarray(tiff_data, mode="L")

    convert_time = time.time() - convert_start
    logger.info(
        f"Array conversion completed in {convert_time:.2f}s, PIL image size: {pil_image.size}"
    )

    target_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as compressed PNG
    if progress_callback:
        progress_callback(70.0, "saving png")
    logger.info(
        "Saving PNG with compression level %s via Pillow (this may take a while for large images)",
        PNG_COMPRESS_LEVEL,
    )
    save_start = time.time()

    def _save_png() -> None:
        pil_image.save(
            str(target_file_path),
            "PNG",
            compress_level=PNG_COMPRESS_LEVEL,
            optimize=False,
        )

    _run_with_progress(
        _save_png,
        progress_callback=progress_callback,
        start_progress=70.0,
        message="saving png",
    )

    _maybe_sync_png_save(target_file_path)
    save_time = time.time() - save_start

    # Verify file exists and is readable
    if not target_file_path.exists():
        raise OSError(f"PNG file was not created at {target_file_path}")

    file_size = target_file_path.stat().st_size
    if file_size == 0:
        raise OSError(f"PNG file is empty at {target_file_path}")

    if progress_callback:
        progress_callback(98.0, "verifying png")
    logger.info(f"PNG saved in {save_time:.2f}s, verified: {file_size} bytes")

    total_time = time.time() - step_start
    logger.info(f"Total PNG conversion time: {total_time:.2f}s")
    if progress_callback:
        progress_callback(100.0, "png saved")
    return target_file_path


def _vips_plane_to_numpy(vips_image) -> np.ndarray:
    memory = vips_image.write_to_memory()
    array = np.frombuffer(memory, dtype=np.uint8)
    return array.reshape(vips_image.height, vips_image.width)


def _load_tiff_plane_uint8(tiff_path: Path, metadata: dict) -> np.ndarray:
    logger = logging.getLogger(__name__)

    if pyvips is not None:
        try:
            source_image = pyvips.Image.new_from_file(str(tiff_path), access="sequential")
            prepared = _prepare_vips_grayscale_uchar(
                source_image,
                channels=int(metadata.get("channels", 1) or 1),
                bit_depth=int(metadata.get("bit_depth", 8) or 8),
                logger=logger,
            )
            return _vips_plane_to_numpy(prepared)
        except Exception:  # pragma: no cover - exercised only with libvips installed
            logger.warning(
                "pyvips could not decode %s; falling back to tifffile",
                tiff_path,
                exc_info=True,
            )

    load_start = time.time()
    tiff_data = tifffile.imread(str(tiff_path))
    logger.info(
        "TIFF decoded in %.2fs, shape: %s, dtype: %s",
        time.time() - load_start,
        tiff_data.shape,
        tiff_data.dtype,
    )
    tiff_data = _select_png_grayscale_plane(tiff_data, metadata)
    if tiff_data.dtype != np.uint8:
        bit_depth = int(metadata.get("bit_depth", 8) or 8)
        logger.info("Normalizing %s-bit to 8-bit", bit_depth)
        tiff_data = _normalize_array_to_uint8(tiff_data, bit_depth=bit_depth)
    return tiff_data


def _load_png_plane_uint8(png_path: Path) -> np.ndarray:
    with Image.open(png_path) as source_image:
        if source_image.mode in {"I;16", "I;16B", "I;16L"}:
            values = np.asarray(source_image, dtype=np.uint16)
            return _normalize_array_to_uint8(values, bit_depth=16)
        if source_image.mode == "L":
            return np.asarray(source_image, dtype=np.uint8)
        return np.asarray(source_image.convert("L"), dtype=np.uint8)


def load_source_plane_uint8(source_path: Path, metadata: dict) -> np.ndarray:
    """Decode a staged upload to the canonical 8-bit grayscale plane.

    This is the single decode the import pipeline is allowed: the array it
    returns feeds both the canonical PNG and the NGFF pyramid. The
    transformation is exactly the one ``convert_tiff_to_png`` /
    ``convert_png_to_8bit_grayscale`` apply (first band, then native
    bit-depth scaling for >8-bit integer data, never a min/max stretch), so
    every consumer sees the same pixels it saw when the PNG was the only
    canonical form.

    Every failure is reported as a ``ValueError`` that **names this stage**.
    ``convert_tiff_to_png`` used to do that ("Error converting TIFF to PNG:
    failed to read 1050000 bytes, got 419846"); when the pipeline stopped going
    through it, a truncated upload started surfacing as a bare
    "failed to read 1050000 bytes, got 419846" -- a byte count with nothing to
    say it was the user's image that would not decode. The decoder's own
    sentence is kept verbatim inside the message and chained as ``__cause__``.
    """

    source_path = Path(source_path)
    is_png = source_path.suffix.lower() in PNG_UPLOAD_SUFFIXES
    stage = "PNG" if is_png else "TIFF"
    try:
        plane = (
            _load_png_plane_uint8(source_path)
            if is_png
            else _load_tiff_plane_uint8(source_path, metadata)
        )
        if plane.ndim != 2:
            raise ValueError(f"unsupported image shape {plane.shape}")
        # zarr and Pillow both want a contiguous, writable-strided buffer; a
        # tifffile view or a Pillow-backed asarray can be neither.
        return np.ascontiguousarray(plane, dtype=np.uint8)
    except MemoryError as exc:
        raise ValueError(f"Out of memory: Image is too large to process. {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Error decoding {stage} to 8-bit grayscale: {exc}") from exc


def save_plane_as_canonical_png(plane: np.ndarray, target_file_path: Path) -> Path:
    """Write the canonical 8-bit grayscale PNG from an already-decoded plane.

    Byte-for-byte what the Pillow branch of ``convert_tiff_to_png`` wrote --
    same mode, same compress level, same ``optimize=False`` -- just without
    decoding the source a second time to get there.
    """

    logger = logging.getLogger(__name__)
    target_file_path = Path(target_file_path)
    target_file_path.parent.mkdir(parents=True, exist_ok=True)

    save_start = time.time()
    Image.fromarray(plane, mode="L").save(
        str(target_file_path),
        "PNG",
        compress_level=PNG_COMPRESS_LEVEL,
        optimize=False,
    )
    _maybe_sync_png_save(target_file_path)

    if not target_file_path.exists():
        raise OSError(f"PNG file was not created at {target_file_path}")
    file_size = target_file_path.stat().st_size
    if file_size == 0:
        raise OSError(f"PNG file is empty at {target_file_path}")
    logger.info(
        "Canonical PNG written in %.2fs (%s bytes) to %s",
        time.time() - save_start,
        file_size,
        target_file_path,
    )
    return target_file_path


def validate_upload_file(uploaded_file: UploadedFile) -> tuple[bool, str | None]:
    """
    Validate that the uploaded file is one QuantEM can read.

    v1 accepts TIFF and PNG only (owner ruling 2026-08-06); the readers in
    ``assets/volume_readers.py`` accept exactly the same set, and the rejection
    message names the accepted formats so the user is never left guessing.

    Args:
        uploaded_file: The uploaded file object

    Returns:
        Tuple of (is_valid, error_message)
        If valid, returns (True, None)
        If invalid, returns (False, error_message)
    """
    if not uploaded_file:
        return False, "No file provided"

    original_filename = uploaded_file.name
    file_ext = Path(original_filename).suffix.lower()

    if file_ext not in UPLOAD_SUFFIXES:
        accepted = ", ".join(UPLOAD_SUFFIXES)
        got = file_ext or "no extension"
        return False, (f"Unsupported file type '{got}'. QuantEM accepts {accepted} files.")

    return True, None


def save_uploaded_file_to_temp(uploaded_file: UploadedFile) -> Path:
    """
    Save uploaded file to a temporary location.

    Args:
        uploaded_file: The uploaded file object

    Returns:
        Path to the temporary file

    Raises:
        IOError: If file cannot be saved
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    original_filename = uploaded_file.name
    file_ext = Path(original_filename).suffix.lower()
    temp_file_path = UPLOADS_DIR / f"{uuid.uuid4()}{file_ext}"

    with open(temp_file_path, "wb") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    return temp_file_path


#: Block size for staging an upload. Django's ``UploadedFile.chunks()`` defaults
#: to 64 kB, which costs ~16k Python-level iterations per GiB for no benefit
#: when the source is a file on the same volume as the destination.
UPLOAD_COPY_BLOCK_BYTES = 8 * 1024 * 1024


def save_uploaded_file_to_path(uploaded_file: UploadedFile, target_path: Path) -> None:
    """
    Save an uploaded file to a specific target path.

    An upload streamed by :class:`quantem.assets.upload_staging
    .StagedFileUploadHandler` is already in this directory, so it is claimed
    with a rename and no bytes move at all. Anything else -- a small in-memory
    upload, a ``SimpleUploadedFile`` from a test, a caller that did not install
    the handler -- is copied.

    Note what is deliberately *not* done: renaming Django's own
    ``TemporaryUploadedFile``. MEASURED on Windows 11 with Python 3.13, its
    ``tempfile.NamedTemporaryFile`` opens with ``O_TEMPORARY``, so
    ``os.replace`` on that path either raises ``PermissionError`` or succeeds
    and is then undone when the handle closes -- ``FILE_FLAG_DELETE_ON_CLOSE``
    deletes by the file's current name. That is why the handler exists instead.

    Args:
        uploaded_file: The uploaded file object
        target_path: Destination path to write to
    """
    claim = getattr(uploaded_file, "claim", None)
    if callable(claim):
        claim(target_path)
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        for chunk in uploaded_file.chunks(chunk_size=UPLOAD_COPY_BLOCK_BYTES):
            f.write(chunk)


def extract_tiff_metadata(tiff_path: Path) -> dict:
    """
    Extract geometry metadata from a TIFF file.

    Args:
        tiff_path: Path to the TIFF file

    Returns:
        Dictionary with keys: width, height, channels, bit_depth, dtype, shape

    Raises:
        ValueError: If TIFF cannot be read or has unsupported format

    Also reads the physical pixel size when the file declares one, so
    ``Asset.pixel_size_nm`` is filled on upload rather than left for the user.
    Nothing measurable works without it: it drives per-organelle resampling and
    every number the analysis suite produces. The same precedence the volume
    reader uses -- OME ``PhysicalSizeX`` first, then the ImageJ/Fiji ``unit``
    from the ImageDescription block, then the Zeiss/Fibics ATLAS record in TIFF
    tag 51023, then the bare ``XResolution``/``ResolutionUnit`` tags (the rule,
    and what happens when they disagree, is documented on
    ``volume_readers.in_plane_pixel_size_nm``) -- so 2D and 3D imports cannot
    disagree about the same file. Returns ``None`` when the file is silent; a
    guess would be worse than asking. ``pixel_size_caveat`` carries the conflict
    note when the file contradicts itself and ``pixel_size_source`` names the
    tag that supplied the number, so both land in the rendition's
    ``source_metadata`` and from there on the API payload.
    """
    try:
        with tifffile.TiffFile(str(tiff_path)) as tif:
            series = tif.series[0]
            shape = series.shape
            axes = series.axes
            dtype = series.dtype

        if "Y" not in axes or "X" not in axes:
            raise ValueError(f"Unsupported TIFF axes: {axes}")

        height = shape[axes.index("Y")]
        width = shape[axes.index("X")]
        if "C" in axes:
            channels = shape[axes.index("C")]
        elif "S" in axes:
            channels = shape[axes.index("S")]
        else:
            channels = 1

        if dtype == np.uint8:
            bit_depth = 8
        elif dtype == np.uint16:
            bit_depth = 16
        elif dtype == np.uint32:
            bit_depth = 32
        else:
            bit_depth = 8

        pixel_size_nm, pixel_size_caveat, pixel_size_source = _tiff_pixel_size_nm(tiff_path)

        return {
            "width": int(width),
            "height": int(height),
            "channels": int(channels),
            "bit_depth": int(bit_depth),
            "dtype": dtype,
            "shape": shape,
            "pixel_size_nm": pixel_size_nm,
            "pixel_size_caveat": pixel_size_caveat,
            "pixel_size_source": pixel_size_source,
        }
    except Exception as e:
        raise ValueError(f"Error reading TIFF file: {str(e)}") from e


def _tiff_pixel_size_nm(
    tiff_path: Path,
) -> tuple[float | None, str | None, str | None]:
    """``(nm, caveat, source)``; ``(None, None, None)`` when the file is silent.

    Delegates to the same helpers the volume reader uses so a file imported as a
    2D image and the same file imported as a volume report the same scale. The
    second element is the calibration caveat from
    ``volume_readers.in_plane_pixel_size_nm`` -- a file that declares its scale
    twice and disagrees with itself, or a vendor record that no longer matches
    the raster -- and is ``None`` in the normal case. The third names the tag
    or block the number came from, so a reader can tell a pixel size the
    microscope wrote from one a person typed.

    Header reads only; no pixels are decoded, which is what makes this cheap
    enough to run inside the upload request on a 2 GB file.
    """
    from .volume_readers import (
        PIXEL_SIZE_SOURCE_OME,
        _imagej_calibration,
        _ome_physical_size,
        in_plane_pixel_size_nm,
    )

    try:
        with tifffile.TiffFile(str(tiff_path)) as tif:
            if tif.ome_metadata:
                value = _ome_physical_size(tif.ome_metadata, "X")
                if value:
                    return value, None, PIXEL_SIZE_SOURCE_OME
            ij = _imagej_calibration(tif)
            nm, caveat, source = in_plane_pixel_size_nm(tif.pages[0], "XResolution", ij.get("unit"))
            return nm, caveat, source
    except Exception:  # pragma: no cover - a malformed tag must not block upload
        logging.getLogger(__name__).debug(
            "Could not read a pixel size from %s", tiff_path, exc_info=True
        )
        return None, None, None


# Pillow mode -> (channels, bit depth) for the accepted PNG variants.
_PNG_MODE_GEOMETRY = {
    "1": (1, 8),
    "L": (1, 8),
    "LA": (2, 8),
    "P": (1, 8),
    "PA": (2, 8),
    "RGB": (3, 8),
    "RGBA": (4, 8),
    "I;16": (1, 16),
    "I;16B": (1, 16),
    "I;16L": (1, 16),
    "I": (1, 32),
    "F": (1, 32),
}


def extract_png_metadata(png_path: Path) -> dict:
    """
    Extract geometry metadata from a PNG file, mirroring ``extract_tiff_metadata``.

    Only the header is parsed: ``Image.open`` is lazy, so a 40k x 40k PNG is not
    decoded just to learn its shape.
    """
    try:
        with Image.open(png_path) as img:
            width, height = int(img.size[0]), int(img.size[1])
            mode = str(img.mode)
    except Exception as e:
        raise ValueError(f"Error reading PNG file: {str(e)}") from e

    channels, bit_depth = _PNG_MODE_GEOMETRY.get(mode, (1, 8))
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "bit_depth": bit_depth,
        "dtype": "uint16" if bit_depth == 16 else "uint8",
        "shape": (height, width) if channels == 1 else (height, width, channels),
    }


def extract_image_metadata(image_path: Path) -> dict:
    """
    Extract geometry metadata from an uploaded TIFF or PNG.

    Dispatches on suffix over exactly the set the upload API accepts
    (:data:`UPLOAD_SUFFIXES`); anything else is a caller bug, and is reported as
    a user-facing ``ValueError`` rather than a KeyError.
    """
    suffix = Path(image_path).suffix.lower()
    if suffix in TIFF_UPLOAD_SUFFIXES:
        return extract_tiff_metadata(Path(image_path))
    if suffix in PNG_UPLOAD_SUFFIXES:
        return extract_png_metadata(Path(image_path))
    raise ValueError(
        f"Unsupported file type '{suffix or image_path}'. "
        f"QuantEM accepts {', '.join(UPLOAD_SUFFIXES)} files."
    )


def convert_tiff_to_png(
    tiff_path: Path,
    metadata: dict,
    target_file_path: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    """
    Convert TIFF file to compressed PNG.

    Args:
        tiff_path: Path to the source TIFF file
        metadata: Dictionary with image metadata (from extract_tiff_metadata)

    Returns:
        Path to the created PNG file

    Raises:
        ValueError: If conversion fails
    """
    logger = logging.getLogger(__name__)
    try:
        return _convert_tiff_to_png_via_vips(
            tiff_path,
            metadata,
            target_file_path,
            progress_callback=progress_callback,
        )
    except MemoryError as e:
        logger.error(f"Out of memory during TIFF to PNG conversion: {str(e)}", exc_info=True)
        raise ValueError(f"Out of memory: Image is too large to process. {str(e)}") from e
    except Exception as e:
        # pyvips is an optional accelerator (`pip install quantem-app[vips]`). Its
        # absence is the normal case for a plain pip install, not a fault, so it
        # is logged at debug; anything else is a real degradation and warns.
        if isinstance(e, RuntimeError) and "pyvips is unavailable" in str(e):
            logger.debug("pyvips not installed; using the NumPy/Pillow path for %s", tiff_path)
        else:
            logger.warning(
                "Streaming TIFF->PNG conversion failed for %s; "
                "falling back to NumPy/Pillow path: %s",
                tiff_path,
                e,
                exc_info=True,
            )
        try:
            if target_file_path.exists():
                target_file_path.unlink()
        except OSError:
            pass
        try:
            return _convert_tiff_to_png_via_numpy(
                tiff_path,
                metadata,
                target_file_path,
                progress_callback=progress_callback,
            )
        except MemoryError as mem_exc:
            logger.error(
                f"Out of memory during TIFF to PNG conversion: {str(mem_exc)}",
                exc_info=True,
            )
            raise ValueError(
                f"Out of memory: Image is too large to process. {str(mem_exc)}"
            ) from mem_exc
        except Exception as fallback_exc:
            logger.error(
                f"Error converting TIFF to PNG: {str(fallback_exc)}",
                exc_info=True,
            )
            raise ValueError(f"Error converting TIFF to PNG: {str(fallback_exc)}") from fallback_exc


def _save_roi_png_from_ngff(
    image,
    roi_path: Path,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> bool:
    """Crop the ROI out of NGFF level 0 instead of the canonical PNG.

    Neither Pillow nor libvips can seek into a PNG, so cropping a 9 MP window
    out of a 200 MB canonical PNG costs a full decode of the whole image. Level
    0 of the pyramid holds exactly the same pixels in 1024^2 chunks, so the
    same window is a handful of chunk reads. Returns ``False`` (and writes
    nothing) whenever the store is absent or unreadable, which is the normal
    case for an ROI requested before the pyramid exists.
    """

    from .task_utils import _ngff_level0_window

    try:
        window = _ngff_level0_window(image, x, y, width, height)
    except Exception:
        logging.getLogger(__name__).debug(
            "NGFF level-0 ROI crop unavailable for image %s; using the source file",
            getattr(image, "id", None),
            exc_info=True,
        )
        return False
    if window is None:
        # No store, a store whose geometry disagrees with the image, or a
        # window clipped by the store's bounds -- none of which is the ROI that
        # was asked for. Fall through rather than write the wrong crop.
        return False
    Image.fromarray(window, mode="L").save(str(roi_path))
    return True


def create_roi_image_from_image(
    image: object,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    source: str = "AUTO",
    display_name: str | None = None,
    is_active: bool = True,
    is_complete: bool = False,
) -> ImageROI:
    """Create an ROI image from an asset-backed openable and save its PNG.

    Cropping and encoding deliberately happen before the transaction. QuantEM
    uses ``BEGIN IMMEDIATE`` for explicit SQLite transactions, so entering an
    atomic block takes the one database writer lock even before the first
    query. A fallback crop can decode a very large source image and take longer
    than SQLite's busy timeout; holding the lock across that work made an
    unrelated upload fail with ``database is locked`` and stalled job-status
    writes for the same interval.

    Only the row count, active-ROI handoff, and row insert need to be atomic.
    If that short commit fails, remove the PNG that no row can reference.
    """

    ROIS_DIR.mkdir(parents=True, exist_ok=True)
    roi_id = uuid.uuid4()
    roi_filename = f"{roi_id}.png"
    roi_path = ROIS_DIR / roi_filename

    from .task_utils import _source_read_is_already_canonical

    file_path = get_file_absolute_path(image)
    logger = logging.getLogger(__name__)
    crop_start = time.time()
    if _save_roi_png_from_ngff(image, roi_path, x=x, y=y, width=width, height=height):
        logger.info(
            "Created ROI %s from image %s via the NGFF pyramid in %.2fs",
            roi_id,
            image.id,
            time.time() - crop_start,
        )
    elif not _source_read_is_already_canonical(image):
        # No pyramid to crop, and the source is >8-bit or multi-band, so the
        # plain grayscale reads below would saturate or luma-blend it. Decode
        # the way the importer does instead, or this ROI is a different picture
        # from the one the user will see in the viewer.
        from .task_utils import _canonical_plane_from_source_file

        plane = Image.fromarray(_canonical_plane_from_source_file(image), mode="L")
        plane.crop((x, y, x + width, y + height)).save(
            str(roi_path), "PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=False
        )
        logger.info(
            "Created ROI %s from image %s via the canonical source decode in %.2fs",
            roi_id,
            image.id,
            time.time() - crop_start,
        )
    elif pyvips is not None:
        try:
            source_image = pyvips.Image.new_from_file(str(file_path), access="sequential")
            if source_image.bands > 1:
                source_image = source_image.extract_band(0)
            roi_image = source_image.crop(x, y, width, height)
            roi_image.pngsave(
                str(roi_path),
                compression=PNG_COMPRESS_LEVEL,
                interlace=False,
            )
            logger.info(
                "Created ROI %s from image %s via pyvips in %.2fs",
                roi_id,
                image.id,
                time.time() - crop_start,
            )
        except Exception:
            logger.warning(
                "pyvips ROI crop failed for image %s; falling back to PIL",
                image.id,
                exc_info=True,
            )
            pil_image = Image.open(file_path)
            if pil_image.mode != "L":
                pil_image = pil_image.convert("L")
            roi = pil_image.crop((x, y, x + width, y + height))
            roi.save(roi_path)
    else:
        pil_image = Image.open(file_path)
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")
        roi = pil_image.crop((x, y, x + width, y + height))
        roi.save(roi_path)

    asset = getattr(image, "asset", None)
    try:
        with transaction.atomic():
            resolved_display_name = display_name
            if resolved_display_name is None:
                current_roi_count = ImageROI.objects.filter(asset=asset).count()
                resolved_display_name = f"{image.display_name} ROI {current_roi_count + 1}"

            if is_active and asset is not None:
                ImageROI.objects.filter(asset=asset).update(is_active=False)

            return ImageROI.objects.create(
                id=roi_id,
                asset=asset,
                display_name=resolved_display_name,
                x=x,
                y=y,
                width=width,
                height=height,
                source=source,
                is_active=is_active,
                is_complete=is_complete,
            )
    except BaseException:
        try:
            roi_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove unreferenced ROI image %s", roi_path)
        raise
