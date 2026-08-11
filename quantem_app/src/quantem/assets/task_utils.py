"""Reading an asset's pixels: full plane, preview, and ROI window.

**One source of truth, one decode, and no way to be handed zeros.**

Every read here resolves through
:func:`quantem.assets.pyramid_authority.resolve_pyramid`. There is no
``path.exists()``, no ``.zarray`` probe, no completion marker and no
chunk-presence heuristic in this module any more: those were four of the twelve
different predicates that between them let three rounds of guards each miss a
path. A reader asks the authority, and gets either a published generation --
which it opens through the **strict store**, so a chunk that vanishes mid-read
raises :class:`~quantem.assets.pyramid_authority.PyramidChunkMissing` instead of
silently substituting ``fill_value`` -- or an :class:`Unavailable` carrying a
reason it must answer for:

* ``NEVER_BUILT`` / ``BUILDING`` -> decode the source file. This is the normal
  state before the first import finishes, and it is honest: the source holds
  the same pixels the pyramid will.
* ``GEOMETRY_MISMATCH`` / ``STALE_DECODER`` -> decode the source file, and log.
  Serving a store that describes a different picture under this asset's name
  would be worse than a slow read.
* ``TERMINAL_FAILURE`` / ``CANCELLED`` -> **raise**. An import that failed has
  no pixels to hand out, and returning the staged upload's would be the
  half-open asset this whole change exists to remove.

The source-file fallback goes through
:func:`quantem.assets.canonical_decode.decode_canonical_array` and only that.
The three Pillow/libvips arms that used to live here -- ``Image.open(...)``,
``.convert("L")``, ``pyvips ... extract_band(0)`` -- are gone: two of them
saturated an ``I;16`` source to a white rectangle, which is the bug that has
now been found in four separate functions.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PIL import Image

# Disable decompression bomb check for large SEM/TEM images.
Image.MAX_IMAGE_PIXELS = None

from .canonical_decode import decode_canonical_array
from .file_paths import get_file_absolute_path
from .pyramid_authority import (
    Intent,
    PublishedPyramid,
    PyramidChunkMissing,
    Reason,
    Unavailable,
    resolve_pyramid,
)

logger = logging.getLogger(__name__)

def _open_generation_level_cache_clear() -> None:
    """Drop the open-array cache.

    Only a test needs this: a published generation is immutable, so nothing in
    the product invalidates the cache -- but Windows will not delete a
    directory a cached array still holds a chunk file open in, and the tests
    that delete a generation on purpose have to let go of it first.
    """

    from .pyramid_authority import _open_generation_level

    _open_generation_level.cache_clear()


__all__ = [
    "PyramidChunkMissing",
    "load_image_array",
    "load_image_ngff_level0_roi_array",
    "load_image_preview_array",
    "load_image_roi_array",
]

#: Reasons a reader may answer by decoding the source file instead. Everything
#: else is either a published pyramid or a refusal.
_SOURCE_FALLBACK_REASONS = frozenset(
    {Reason.NEVER_BUILT, Reason.BUILDING, Reason.GEOMETRY_MISMATCH, Reason.STALE_DECODER}
)


class PyramidUnavailable(FileNotFoundError):
    """This asset has no readable pixels, and saying so is the honest answer."""

    def __init__(self, unavailable: Unavailable) -> None:
        super().__init__(f"{unavailable.reason.value}: {unavailable.detail}")
        self.unavailable = unavailable


def _resolved(image) -> PublishedPyramid | Unavailable:
    return resolve_pyramid(image, intent=Intent.READ)


def _published_or_reason(image) -> tuple[PublishedPyramid | None, Unavailable | None]:
    """A published generation, or the reason there is none -- refusing if terminal."""

    resolved = _resolved(image)
    if isinstance(resolved, PublishedPyramid):
        return resolved, None
    if resolved.reason in _SOURCE_FALLBACK_REASONS:
        if resolved.reason in {Reason.GEOMETRY_MISMATCH, Reason.STALE_DECODER}:
            logger.warning(
                "Image %s: not using the published pyramid (%s: %s); decoding the source file.",
                getattr(image, "id", None),
                resolved.reason.value,
                resolved.detail,
            )
        return None, resolved
    raise PyramidUnavailable(resolved)


def _source_read_is_already_canonical(image) -> bool:
    """Always ``False``. Kept only so ``utils`` keeps compiling.

    This used to answer "is a plain ``Image.open(...).convert('L')`` of the
    source the canonical plane?", and ``utils.create_roi_image_from_image``
    still asks it before falling back to its own Pillow/libvips arms. There is
    no longer a case where the answer is yes worth acting on: one decoder
    handles every container and every dtype, it costs the same, and the
    shortcut is precisely the arm that saturated an ``I;16`` source into the
    ROI a user then labelled. Returning ``False`` routes that call site through
    :func:`_canonical_plane_from_source_file`.

    Handoff: delete this when ``utils.create_roi_image_from_image`` is folded
    into :mod:`quantem.assets.canonical_decode`.
    """

    del image
    return False


def _canonical_plane_from_source_file(image) -> np.ndarray:
    """Decode the source file the way the importer does, and only that way."""

    return decode_canonical_array(
        get_file_absolute_path(image),
        declared={
            "channels": int(getattr(image, "channels", 1) or 1),
            "bit_depth": int(getattr(image, "bit_depth", 8) or 8),
            "width": int(getattr(image, "width", 0) or 0),
            "height": int(getattr(image, "height", 0) or 0),
        },
    )


def _ngff_level0_plane(image) -> np.ndarray | None:
    published, _ = _published_or_reason(image)
    if published is None:
        return None
    return np.asarray(published.open_level(0)[0], dtype=np.uint8)


def load_image_array(image) -> tuple[np.ndarray, float]:
    """The canonical 8-bit grayscale plane, and how long it took.

    Returns:
        ``(image_array, load_time_seconds)``.

    Raises:
        PyramidUnavailable: the import failed or was cancelled.
        PyramidChunkMissing: the published generation lost a chunk mid-read.
        FileNotFoundError / ValueError: the source file is missing or undecodable.
    """

    load_start = time.time()
    image_array = _ngff_level0_plane(image)
    source = "ngff level 0"
    if image_array is None:
        source = "source file"
        image_array = _canonical_plane_from_source_file(image)
    load_elapsed = time.time() - load_start
    logger.info(
        "Loaded image array from %s: shape=%s, dtype=%s (took %.2f seconds)",
        source, image_array.shape, image_array.dtype, load_elapsed,
    )
    return image_array, load_elapsed


def _thumbnail_of(plane: np.ndarray, max_size: int) -> np.ndarray:
    pil_image = Image.fromarray(plane, mode="L")
    pil_image.thumbnail((max_size, max_size))
    return np.array(pil_image, dtype=np.uint8)


def load_image_preview_array(image, max_size: int = 1024) -> np.ndarray:
    """A downsampled preview of the image, as a grayscale array.

    The AUTO ROI heuristic scores *this* preview to choose the 3000^2 window a
    user is handed to label, so the resampling is deliberately left alone:
    level 0 holds the same pixels as the canonical PNG, and the same Pillow
    ``thumbnail`` runs over them either way.
    """

    plane = _ngff_level0_plane(image)
    if plane is None:
        plane = _canonical_plane_from_source_file(image)
    return _thumbnail_of(plane, max_size)


def load_image_roi_array(image, x: int, y: int, width: int, height: int) -> np.ndarray:
    """A ROI window in full-resolution pixels, as a ``(height, width)`` array."""

    window = _ngff_level0_window(image, x, y, width, height)
    if window is not None:
        return window
    # ``Image.fromarray(...).crop`` rather than a numpy slice: a window that
    # runs off the edge has always come back zero-padded to the requested
    # shape, and callers size their output buffers on that.
    plane = Image.fromarray(_canonical_plane_from_source_file(image), mode="L")
    return np.array(plane.crop((x, y, x + width, y + height)), dtype=np.uint8)


def _ngff_level0_window(
    image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray | None:
    """The requested window out of level 0, or ``None`` to use the source file.

    A window clipped by the store's bounds comes back ``None`` rather than
    short: the file paths zero-pad an over-the-edge crop to the requested
    shape, and returning a smaller array here would be a different answer, not
    a faster one.
    """

    published, _ = _published_or_reason(image)
    if published is None:
        return None
    if x < 0 or y < 0:
        return None
    level0 = published.open_level(0)
    window = np.asarray(level0[0, y : y + height, x : x + width], dtype=np.uint8)
    if window.shape != (height, width):
        return None
    return window


def load_image_ngff_level0_roi_array(
    image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """A ROI window out of the published pyramid, or an exception saying why not.

    Unlike :func:`load_image_roi_array` this never falls back to the source
    file: the caller asked for the pyramid.
    """

    resolved = _resolved(image)
    if not isinstance(resolved, PublishedPyramid):
        raise PyramidUnavailable(resolved)
    level0 = resolved.open_level(0)
    return np.asarray(level0[0, y : y + height, x : x + width], dtype=np.uint8)
