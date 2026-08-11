"""The one decode. Nothing else in the pyramid path may open an image for pixels.

Three verification rounds each found a *different* function turning a 16-bit
source into a white rectangle, and the design review found a fourth
(``ngff.read_source_plane``'s pyvips arm, invisible here because libvips is not
installed, which meant the pyramid's pixels depended on whether a system
library happened to be present). Adding a guard at each site was the losing
game. This module is the site.

**The rule, stated once.**

* Dispatch on **magic bytes**, never on the file name. A staged ``.png`` upload
  that is really a TIFF, and a TIFF renamed ``.png``, both decode correctly --
  and the suffix test that let round 3 rebuild a *staged* upload as though it
  were the canonical PNG has nowhere left to live.
* Take **band 0** of a multi-band container, in **every** container, and take
  it from the axis the *file itself* declares. TIFF says which axis holds its
  samples (``YXS`` vs ``SYX``); PNG through Pillow is always interleaved. The
  shape-guessing heuristic that used to decide this survives only as a last
  resort for a container that declares nothing, because it transposed a
  three-row RGB TIFF and failed the import.
* **Resolve a palette before reading pixels.** A palette image has no band, it
  has a lookup: mode ``P`` and TIFF ``PHOTOMETRIC.PALETTE`` hold *indices*, and
  reading those as intensities returns an unrelated picture with no error at
  all. The palette is applied, then band 0 of the result is taken, so a picture
  saved as a palette and the same picture saved as RGB decode identically.
* Scale the **native integer range** onto 0..255 -- the range the source's own
  dtype declares, so ``int16`` scales against 32767 and not 65535. Never a
  min/max stretch: per-image stretching would make each image's grey levels
  mean something different (owner ruling R6).
* For a **16-bit source only**, fall back to one of a small set of standard
  windows when, and only when, the stated criterion in :func:`plan_conversion`
  says the full-range map would obviously destroy the picture (owner ruling
  R8). This is not hypothetical: on a corpus of 92 real 16-bit EM TIFFs, 26 of
  them -- pancreas mosaics whose values all sit between 28481 and 30076 --
  rendered as **6 to 16 distinct greys** under the fixed full-range map. The
  criterion, the window and the observed range are all recorded, and the window
  comes from a fixed grid rather than being fitted to the image, so two images
  whose data lands in the same window remain directly comparable.
* Floats are already display-range and are clipped, not rescaled -- which is
  what the previous decoder did for them too, via the ``bit_depth = 8``
  fallback.
* **Refuse by name** what cannot be represented honestly: complex data, and
  signed integers holding negative values. Both used to import silently, one by
  discarding the imaginary part and one by clipping half the data to black.

**What the type lock does and does not cover.** Everything here returns a
:class:`CanonicalPlane`, not a bare array, so a decode that slips past the AST
chokepoint test cannot reach :func:`quantem.assets.ngff.build_pyramid`, which
takes a plane and a ticket rather than a path. It does **not** cover the reader
functions in :mod:`quantem.assets.task_utils`: ``load_image_array`` and its
siblings return bare ``ndarray``, and that is what segmentation, ROI selection
and feature measurement consume. An earlier version of this docstring claimed
otherwise; it was wrong, and the gap is real. Closing it means giving those
readers a plane-shaped return type, which is a change in a module this one does
not own.

**Memory, which is a correctness question on the target machine.** Every
mapping is applied a row-block at a time into a single uint8 output, and the
16-bit histogram is accumulated the same way, so peak memory is the output plus
one block rather than several whole-image float copies. MEASURED on a real 400
MP 16-bit EM mosaic (20000 x 20000, 800 MB on disk): **peak working set 5760 MB
before, 1188 MB after**, and 5.19 s before, 3.88 s after. The owner's floor is
an 8 GB laptop (ruling R3), so the old figure was not a tuning opportunity, it
was a machine that could not open the file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# Microscopy images are routinely far past Pillow's bomb threshold.
Image.MAX_IMAGE_PIXELS = None

#: Bumped whenever the transform above changes. Recorded in every published
#: manifest; a generation built by an older decoder resolves as ``STALE_DECODER``
#: and is rebuilt rather than served.
#:
#: ``2026-08-10.2`` fixed the signed-integer scale (``int16`` was divided by
#: 65535 and came out at half brightness), resolved palettes instead of reading
#: their indices as intensities, took band 0 from the axis the file declares,
#: and added the R8 standard-window fallback for 16-bit sources.
DECODER_VERSION = "2026-08-10.2"

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_TIFF_MAGICS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")

CONTAINER_PNG = "png"
CONTAINER_TIFF = "tiff"

#: TIFF ``PhotometricInterpretation`` value for a colour-mapped image.
_PHOTOMETRIC_PALETTE = 3

#: Rows are mapped in blocks of about this many pixels. Large enough that the
#: per-block overhead is invisible, small enough that the temporary stays in
#: cache-friendly megabytes on the 8 GB laptop floor (owner ruling R3).
_BLOCK_PIXELS = 4 * 1024 * 1024

# --- owner ruling R8: when a fixed full-range map obviously destroys the image ---

#: The robust interval used to describe where an image's data actually lives.
#: Percentiles rather than min/max, because a single hot pixel or a black
#: border (both present in the measured corpus) would otherwise claim the whole
#: range. Fixed constants, so the same file always takes the same branch.
ROBUST_LOW_FRACTION = 0.001
ROBUST_HIGH_FRACTION = 0.999

#: **The criterion.** Map the robust interval through the fixed full-range
#: conversion. If it lands on fewer than this many of the 256 output levels,
#: the full-range conversion is discarding more than three of the eight bits
#: and the standard-window fallback is used instead.
#:
#: MEASURED, not guessed, on 92 real 16-bit EM TIFFs (66 TEM plates at
#: 2448x2464 and 26 large pancreas mosaics up to 20000x20000). The corpus is
#: cleanly bimodal: 66 images put their robust interval across **256** output
#: levels, 26 put it across **8 or fewer** (the pancreas mosaics live between
#: raw values 28481 and 30076 -- 2.4% of the uint16 range -- and the fixed
#: full-range map renders them as 6 to 16 distinct greys). Nothing in the
#: corpus landed between 9 and 255 levels, so this threshold sits in an empty
#: gap 32 times wide and is not delicately placed.
MIN_FULL_RANGE_LEVELS = 32

#: The standard windows are the intervals whose two ends are multiples of this
#: step. The chosen window is the smallest of them containing the robust
#: interval. It is a fixed grid shared by every image, not a per-image fit
#: (owner ruling R8 requirement 2), so two images whose data sits in the same
#: cells get byte-identical conversions and stay directly comparable.
#:
#: MEASURED. The step trades tonal resolution against how often two images
#: agree, and both matter, so candidate steps were scored on the 26 real
#: narrow-range EM images in the corpus:
#:
#: ==========  =================  =========================
#: step        distinct windows   output levels min/median
#: ==========  =================  =========================
#: 256                      8/26                  171 / 218
#: 512                      5/26                  114 / 176
#: **1024**             **4/26**             **85 / 121**
#: 4096                     3/26                   34 /  57
#: 8192                     2/26                   21 /  30
#: ==========  =================  =========================
#:
#: 1024 is the knee: 14 of the 26 images share one window and 10 share another,
#: while every image still clears 85 of the 256 output levels -- comfortably
#: above the 32 that defines "obviously destroyed". A coarser step buys one
#: fewer window at the cost of dropping images back to 21-34 levels, which is
#: the damage the fallback exists to prevent.
#:
#: The honest cost is a discontinuity: two similar images whose robust
#: intervals straddle a grid line land on different windows. That is inherent
#: to any fixed quantisation, and it is why the window is recorded on the plane
#: and in the manifest rather than assumed away.
STANDARD_WINDOW_STEP = 1024


class UnsupportedPixelType(ValueError):
    """The file decoded, but its pixels cannot be represented as canonical 8-bit."""


class UnrecognisedContainer(ValueError):
    """The first bytes of the file are neither PNG nor TIFF."""


@dataclass(frozen=True)
class Conversion:
    """How one source plane's values were mapped onto 0..255.

    Recorded on the plane, in the published manifest and in the asset's
    provenance, because owner ruling R1 makes the reduction to 8 bits a
    scientific operation rather than a formatting detail: a reader has to be
    able to see which mapping produced the numbers they are comparing.
    """

    #: Plain words for the source's depth: ``"8-bit"``, ``"16-bit"``,
    #: ``"32-bit"``, ``"floating point"``, ``"black and white"``.
    depth: str
    #: ``"identity"``, ``"full-range"``, ``"standard-window"``, ``"clip"`` or
    #: ``"black-and-white"``.
    strategy: str
    #: Inclusive source-value window mapped onto 0..255, when the strategy is
    #: a range mapping.
    window: tuple[int, int] | None = None
    #: The source values actually observed, when they were measured.
    observed: tuple[float, float] | None = None
    #: The robust interval the criterion was evaluated on, when it ran.
    robust_interval: tuple[int, int] | None = None
    #: How many of the 256 output levels the robust interval would have
    #: occupied under the fixed full-range map. This is the number the
    #: criterion thresholds.
    full_range_levels: int | None = None
    #: Compact token for the provenance string.
    label: str = ""
    #: One plain-language sentence for the import surface, or ``""`` when the
    #: conversion has nothing a scientist needs told (owner ruling R8.4).
    notice: str = ""

    def as_dict(self) -> dict:
        return {
            "depth": self.depth,
            "strategy": self.strategy,
            "window": list(self.window) if self.window else None,
            "observed": list(self.observed) if self.observed else None,
            "robust_interval": (
                list(self.robust_interval) if self.robust_interval else None
            ),
            "full_range_levels": self.full_range_levels,
            "notice": self.notice,
        }


@dataclass(frozen=True)
class CanonicalPlane:
    """A validated 2-D uint8 plane, plus what produced it.

    Deliberately **not** an ``ndarray`` subclass: a bare array must not be
    passable where a canonical plane is required, and a subclass would be. The
    validation runs at construction, so an ill-formed plane cannot exist.
    """

    array: np.ndarray
    decoder_version: str
    provenance: str
    source_fingerprint: str
    conversion: Conversion | None = field(default=None)

    def __post_init__(self) -> None:
        array = self.array
        if not isinstance(array, np.ndarray):
            raise TypeError(f"CanonicalPlane.array must be an ndarray, got {type(array)!r}")
        if array.dtype != np.uint8:
            raise ValueError(f"CanonicalPlane.array must be uint8, got {array.dtype}")
        if array.ndim != 2:
            raise ValueError(f"CanonicalPlane.array must be 2-D, got shape {array.shape}")
        if not array.flags["C_CONTIGUOUS"]:
            raise ValueError("CanonicalPlane.array must be C-contiguous")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.array.shape[0]), int(self.array.shape[1]))

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def notice(self) -> str:
        """The one sentence to show the user about this conversion, or ``""``."""

        return self.conversion.notice if self.conversion else ""


def sniff_container(path: Path) -> str:
    """``"png"`` or ``"tiff"``, from the file's first bytes."""

    path = Path(path)
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
    except OSError as exc:
        raise UnrecognisedContainer(f"could not read {path}: {exc}") from exc
    if head.startswith(_PNG_MAGIC):
        return CONTAINER_PNG
    for magic in _TIFF_MAGICS:
        if head.startswith(magic):
            return CONTAINER_TIFF
    raise UnrecognisedContainer(
        f"{path.name} is neither a PNG nor a TIFF (first bytes {head!r})"
    )


def source_fingerprint(path: Path) -> str:
    """Cheap identity for "is the published pyramid still of this file?".

    Size and modification time, not a content hash: this runs on the read path
    for every resolve, and hashing a gigabyte of EM data to answer it would
    cost more than rebuilding the pyramid. A source that is replaced in place
    with the same size *and* the same nanosecond timestamp is not a case this
    application can produce -- the importer writes each staged upload once,
    under a fresh name.
    """

    try:
        stat = Path(path).stat()
    except OSError:
        return ""
    return hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:32]


# --------------------------------------------------------------------------
# Band selection
# --------------------------------------------------------------------------


def _drop_axis(array: np.ndarray, index: int) -> np.ndarray:
    """Index 0 of ``index``, as a **view** -- ``np.take`` would copy the image."""

    return array[(slice(None),) * index + (0,)]


def _band0_by_axes(array: np.ndarray, axes: str) -> tuple[np.ndarray, int] | None:
    """Band 0 using the axis labels the container declared, or ``None``.

    ``axes`` is tifffile's series axis string (``"YXS"``, ``"SYX"``, ``"QYX"``,
    ``"ZYX"``, ...) or the equivalent synthesised for Pillow. Samples come off
    first, then any remaining non-image axis at index 0, which is the same
    "first page, first band" answer the heuristic gave for every unambiguous
    layout -- it simply also gets the ambiguous ones right.
    """

    if not axes or len(axes) != array.ndim:
        return None
    order = list(axes)
    if order.count("Y") != 1 or order.count("X") != 1:
        return None
    bands = 1
    working = array
    while working.ndim > 2:
        if "S" in order:
            index = order.index("S")
            bands = max(bands, int(working.shape[index]))
        else:
            index = next(i for i, axis in enumerate(order) if axis not in ("Y", "X"))
        working = _drop_axis(working, index)
        order.pop(index)
    if order == ["X", "Y"]:
        working = working.T
    return working, bands


def _band0_by_shape(array: np.ndarray) -> tuple[np.ndarray, int]:
    """Last resort for a container that declares no axis labels.

    Kept because a hand-built array can reach here, but no longer the primary
    path: this heuristic read a ``(3, 300, 3)`` interleaved TIFF as planar and
    produced a ``(300, 3)`` plane, which then failed the geometry check at
    import (verification finding F4).
    """

    if array.ndim == 2:
        return array, 1
    if array.ndim == 3:
        leading, trailing = array.shape[0], array.shape[-1]
        if trailing <= 4 and leading > 4:
            return array[:, :, 0], int(trailing)
        return array[0], int(leading)
    squeezed = np.squeeze(array)
    if squeezed.ndim == 2:
        return squeezed, 1
    if squeezed.ndim == 3:
        return _band0_by_shape(squeezed)
    raise UnsupportedPixelType(f"unsupported image shape {array.shape}")


def _band0(array: np.ndarray, axes: str = "") -> tuple[np.ndarray, int]:
    if array.ndim == 2:
        return array, 1
    declared = _band0_by_axes(array, axes)
    if declared is not None and declared[0].ndim == 2:
        return declared
    return _band0_by_shape(array)


# --------------------------------------------------------------------------
# Value mapping
# --------------------------------------------------------------------------


def _blocked(plane: np.ndarray):
    """Row blocks of about ``_BLOCK_PIXELS`` pixels, as views."""

    height = int(plane.shape[0])
    width = max(1, int(plane.shape[1]) if plane.ndim > 1 else 1)
    rows = max(1, _BLOCK_PIXELS // width)
    for start in range(0, height, rows):
        yield start, min(height, start + rows)


def _histogram(plane: np.ndarray, domain: int) -> np.ndarray:
    """Exact value histogram over ``0 .. domain - 1``, one row block at a time.

    ``np.bincount`` promotes its input to ``intp``, so calling it on a whole
    475 MP plane would allocate 3.8 GB. Accumulating per block keeps the
    temporary at a few tens of megabytes and gives the identical result.
    """

    total = np.zeros(domain, dtype=np.int64)
    for start, stop in _blocked(plane):
        block = np.ascontiguousarray(plane[start:stop]).reshape(-1)
        total += np.bincount(block, minlength=domain)[:domain]
    return total


def _value_at_fraction(histogram: np.ndarray, total: int, fraction: float) -> int:
    """The smallest value at or below which ``fraction`` of the pixels lie."""

    if total <= 0:
        return 0
    cumulative = np.cumsum(histogram)
    return int(np.searchsorted(cumulative, fraction * total, side="left"))


def standard_window(low: int, high: int, declared_max: int) -> tuple[int, int]:
    """The smallest standard window containing ``[low, high]``.

    The standard windows are the intervals whose ends are multiples of
    :data:`STANDARD_WINDOW_STEP` -- a fixed grid shared by every image, never
    fitted to one (owner ruling R8 requirement 2). A source too narrow for even
    one grid cell, or a grid coarser than the whole declared range, falls
    through to the full range, which is always a valid answer.
    """

    step = STANDARD_WINDOW_STEP
    start = int((low // step) * step)
    stop = int(min(declared_max, ((high // step) + 1) * step - 1))
    if stop <= start or (start == 0 and stop >= declared_max):
        return 0, int(declared_max)
    return start, stop


def plan_conversion(plane: np.ndarray, declared_max: int) -> tuple[tuple[int, int], dict]:
    """Choose the window for a 16-bit source: full range, or a standard window.

    **The criterion, stated once and deterministic.** Take the robust interval
    ``[p0.1, p99.9]``. Map both ends through the fixed full-range conversion.
    If the interval covers at least :data:`MIN_FULL_RANGE_LEVELS` of the 256
    output levels, keep the full-range conversion -- that is the default and
    the comparable one (owner ruling R6). Only when it covers fewer does the
    conversion fall back to the smallest standard window containing the
    interval. The inputs are the image's own values and three module constants,
    so the same file always takes the same branch.
    """

    domain = declared_max + 1
    histogram = _histogram(plane, domain)
    total = int(histogram.sum())
    if total <= 0:
        return (0, declared_max), {
            "strategy": "full-range",
            "observed": None,
            "robust_interval": None,
            "full_range_levels": None,
        }
    non_zero = np.nonzero(histogram)[0]
    observed = (int(non_zero[0]), int(non_zero[-1]))
    low = _value_at_fraction(histogram, total, ROBUST_LOW_FRACTION)
    high = _value_at_fraction(histogram, total, ROBUST_HIGH_FRACTION)
    if high < low:
        low, high = high, low
    levels = (high * 255 // declared_max) - (low * 255 // declared_max) + 1
    detail = {
        "observed": observed,
        "robust_interval": (low, high),
        "full_range_levels": int(levels),
    }
    if levels >= MIN_FULL_RANGE_LEVELS:
        detail["strategy"] = "full-range"
        return (0, declared_max), detail
    detail["strategy"] = "standard-window"
    return standard_window(low, high, declared_max), detail


def _window_lut(window: tuple[int, int], domain: int, work) -> np.ndarray:
    """A uint8 lookup table for every value a 16-bit source can hold.

    ``work`` is the floating type the shipped decoder used for that branch --
    float32 for unsigned, float64 for signed. Keeping it identical is what
    makes the full-range branch bit-for-bit the arithmetic that ran before
    (``value.astype(work) * (255 / native_max)``, clipped, truncated), verified
    over all 65 536 values in the tests: the default path is not a behaviour
    change, only a cheaper one.
    """

    low, high = window
    span = max(1, high - low)
    values = np.arange(domain, dtype=work) - work(low)
    scaled = values * work(255.0 / span)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _apply_lut(plane: np.ndarray, lut: np.ndarray) -> np.ndarray:
    out = np.empty(plane.shape, dtype=np.uint8)
    for start, stop in _blocked(plane):
        out[start:stop] = lut[plane[start:stop]]
    return out


def _apply_scale(plane: np.ndarray, native_max: float, work: type) -> np.ndarray:
    out = np.empty(plane.shape, dtype=np.uint8)
    factor = 255.0 / native_max
    for start, stop in _blocked(plane):
        block = plane[start:stop].astype(work) * factor
        out[start:stop] = np.nan_to_num(np.clip(block, 0, 255), nan=0).astype(np.uint8)
    return out


def _observed_range(plane: np.ndarray) -> tuple[float, float] | None:
    if plane.size == 0:
        return None
    low = None
    high = None
    for start, stop in _blocked(plane):
        block = plane[start:stop]
        finite = block[np.isfinite(block)] if np.issubdtype(block.dtype, np.floating) else block
        if finite.size == 0:
            continue
        block_low = float(finite.min())
        block_high = float(finite.max())
        low = block_low if low is None else min(low, block_low)
        high = block_high if high is None else max(high, block_high)
    if low is None or high is None:
        return None
    return (low, high)


def _percent(fraction: float) -> str:
    if fraction <= 0:
        return "less than 0.1%"
    percent = fraction * 100.0
    if percent < 0.1:
        return "less than 0.1%"
    return f"{percent:.1f}%"


def _to_uint8(plane: np.ndarray) -> tuple[np.ndarray, Conversion]:
    dtype = plane.dtype
    if np.issubdtype(dtype, np.complexfloating):
        raise UnsupportedPixelType(
            f"complex pixel data ({dtype}) cannot be shown as a grayscale image; "
            "export the component you want to look at and import that"
        )
    if dtype == np.bool_:
        out = np.empty(plane.shape, dtype=np.uint8)
        for start, stop in _blocked(plane):
            out[start:stop] = plane[start:stop].astype(np.uint8) * 255
        return out, Conversion(
            depth="black and white",
            strategy="black-and-white",
            label="bool->0/255",
        )
    if np.issubdtype(dtype, np.floating):
        out = np.empty(plane.shape, dtype=np.uint8)
        for start, stop in _blocked(plane):
            block = np.clip(plane[start:stop].astype(np.float32), 0, 255)
            out[start:stop] = np.nan_to_num(block, nan=0).astype(np.uint8)
        observed = _observed_range(plane)
        return out, Conversion(
            depth="floating point",
            strategy="clip",
            observed=observed,
            label=f"{dtype}:clip0-255",
            notice=(
                "This image stores floating-point pixel values. Values below 0 and "
                "above 255 were clipped to that range when it was converted to 8-bit."
            ),
        )
    if np.issubdtype(dtype, np.signedinteger):
        if plane.size and int(plane.min()) < 0:
            raise UnsupportedPixelType(
                f"signed integer pixel data ({dtype}) holding negative values cannot be "
                "shown as a grayscale image; the negative half would be clipped to black. "
                "Offset or rescale the data before importing it"
            )
        # The largest value the dtype can hold, not the largest an *unsigned*
        # one of the same width could: dividing int16 by 65535 halved every
        # picture (verification finding F3).
        width = dtype.itemsize * 8
        return _scale_integer(
            plane, declared_max=2 ** (width - 1) - 1, width=width, work=np.float64
        )
    if np.issubdtype(dtype, np.unsignedinteger):
        width = dtype.itemsize * 8
        if width == 8:
            return np.ascontiguousarray(plane, dtype=np.uint8), Conversion(
                depth="8-bit",
                strategy="identity",
                label="uint8:identity",
            )
        return _scale_integer(
            plane, declared_max=2**width - 1, width=width, work=np.float32
        )
    raise UnsupportedPixelType(f"unsupported pixel dtype {dtype}")


def _scale_integer(
    plane: np.ndarray, *, declared_max: int, width: int, work
) -> tuple[np.ndarray, Conversion]:
    """Map an integer source onto 0..255 over a fixed, recorded window.

    ``work`` is the floating type this dtype's branch has always used. It is
    carried rather than chosen because the source matrix compares this decoder
    against an independent oracle at max-abs-diff 0, and float32 versus float64
    moves a uint32 image by one grey level.
    """

    depth = f"{width}-bit"
    if width <= 16 and plane.size:
        # Owner ruling R8 applies to 16-bit sources. A lookup table over the
        # whole value domain makes both branches exact and cheap; anything
        # wider would need a 4-billion-entry table, so it keeps the arithmetic
        # path and the fixed full-range map (see the module report for the
        # 32-bit gap this leaves).
        window, detail = plan_conversion(plane, declared_max)
        lut = _window_lut(window, declared_max + 1, work)
        out = _apply_lut(plane, lut)
        strategy = detail["strategy"]
        observed = detail["observed"]
        levels = detail["full_range_levels"]
        if strategy == "standard-window":
            interval = detail["robust_interval"]
            covered = (interval[1] - interval[0] + 1) / float(declared_max + 1)
            notice = (
                f"This {depth} image was converted to 8-bit. Its pixel values cover only "
                f"{_percent(covered)} of the {depth} range, so the standard full-range "
                f"conversion would have squeezed almost the whole image into "
                f"{levels} of the 256 grey levels. A narrow-range conversion covering "
                f"values {window[0]} to {window[1]} was used instead. Images converted "
                "over different ranges are not directly comparable on grey level."
            )
            label = f"{plane.dtype}:window/{window[0]}-{window[1]}"
        else:
            notice = (
                f"This {depth} image was converted to 8-bit. Every image gets the same "
                "fixed full-range conversion, so grey levels stay comparable between "
                "images."
            )
            label = f"{plane.dtype}:full-range/0-{declared_max}"
        return out, Conversion(
            depth=depth,
            strategy=strategy,
            window=window,
            observed=(float(observed[0]), float(observed[1])) if observed else None,
            robust_interval=detail["robust_interval"],
            full_range_levels=levels,
            label=label,
            notice=notice,
        )

    out = _apply_scale(plane, float(declared_max), work)
    return out, Conversion(
        depth=depth,
        strategy="full-range",
        window=(0, declared_max),
        observed=_observed_range(plane),
        label=f"{plane.dtype}:full-range/0-{declared_max}",
        notice=(
            f"This {depth} image was converted to 8-bit using the same fixed "
            "full-range conversion every image gets, so grey levels stay comparable "
            "between images."
        ),
    )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Read:
    array: np.ndarray
    #: Axis labels the container declared, or ``""`` when it declared none.
    axes: str
    #: ``"palette"`` when a lookup was resolved, ``""`` otherwise.
    note: str


def _read_tiff(path: Path) -> _Read:
    with tifffile.TiffFile(str(path)) as handle:
        series = handle.series[0]
        axes = str(series.axes or "")
        array = np.asarray(handle.asarray())
        page = handle.pages[0]
        photometric = int(getattr(page, "photometric", 0) or 0)
        colormap = getattr(page, "colormap", None)
    if photometric == _PHOTOMETRIC_PALETTE:
        if colormap is None:
            # A palette photometric with no lookup is a broken file. Reading the
            # indices is the only thing left to do, and saying so in provenance
            # is the difference between a known limitation and a silent lie.
            return _Read(array=array, axes=axes, note="palette-unresolved")
        lookup = np.asarray(colormap)
        # Band 0 of the resolved colour, for the same reason band 0 is taken
        # from every other multi-band container.
        resolved = lookup[0][array]
        return _Read(array=resolved, axes=axes, note="palette")
    return _Read(array=array, axes=axes, note="")


def _read_png(path: Path) -> _Read:
    with Image.open(path) as handle:
        handle.load()
        mode = handle.mode
        note = ""
        if mode in {"P", "PA"}:
            # Mode P is an index array, not pixels. Resolving the palette to
            # RGB and then taking band 0 keeps a palette PNG and the same
            # picture saved as RGB decoding to the same plane. The resolved
            # image is bound to its own name: rebinding ``handle`` would leave
            # the ``with`` block closing the copy and the source file open,
            # which on Windows blocks the importer from moving it afterwards.
            resolved = handle.convert("RGBA" if mode == "PA" else "RGB")
            note = "palette"
            array = np.array(resolved)
            resolved.close()
        elif mode in {"I;16", "I;16B", "I;16L"}:
            array = np.asarray(handle, dtype=np.uint16)
        elif mode == "I":
            array = np.asarray(handle, dtype=np.int32)
        elif mode == "F":
            array = np.asarray(handle, dtype=np.float32)
        else:
            array = np.asarray(handle)
    # Pillow hands back interleaved data or a bare plane; it is never planar.
    axes = "YX" if array.ndim == 2 else ("YXS" if array.ndim == 3 else "")
    return _Read(array=array, axes=axes, note=note)


def _read(path: Path, container: str) -> _Read:
    if container == CONTAINER_TIFF:
        return _read_tiff(path)
    return _read_png(path)


def decode_canonical_plane(path, *, declared: dict | None = None) -> CanonicalPlane:
    """Decode ``path`` to the canonical grayscale plane, or refuse by name.

    ``declared`` is the geometry the importer recorded (``channels``,
    ``bit_depth``, ``width``, ``height``). It is **never** used to choose a
    decode -- that comes from the bytes -- but a disagreement is recorded in the
    provenance string, because a store built from a file the database describes
    wrongly is worth being able to see afterwards.
    """

    path = Path(path)
    container = sniff_container(path)
    try:
        read = _read(path, container)
    except UnsupportedPixelType:
        raise
    except MemoryError as exc:
        raise ValueError(f"Out of memory: Image is too large to process. {exc}") from exc
    except Exception as exc:
        raise ValueError(
            f"Error decoding {container.upper()} to 8-bit grayscale: {exc}"
        ) from exc

    band, bands = _band0(read.array, read.axes)
    plane, conversion = _to_uint8(band)
    if plane.ndim != 2:
        raise UnsupportedPixelType(f"unsupported image shape {read.array.shape}")

    parts = [container]
    if read.note:
        parts.append(read.note)
    parts.append(f"band0/{bands}")
    parts.append(conversion.label)
    if conversion.observed is not None:
        low, high = conversion.observed
        low_text = f"{low:g}"
        high_text = f"{high:g}"
        parts.append(f"range/{low_text}-{high_text}")
    provenance = ":".join(parts)
    if declared:
        declared_shape = (
            int(declared.get("height") or 0),
            int(declared.get("width") or 0),
        )
        if declared_shape != (0, 0) and declared_shape != plane.shape:
            provenance += f":declared{declared_shape}"

    return CanonicalPlane(
        array=np.ascontiguousarray(plane, dtype=np.uint8),
        decoder_version=DECODER_VERSION,
        provenance=provenance,
        source_fingerprint=source_fingerprint(path),
        conversion=conversion,
    )


def decode_canonical_array(path, *, declared: dict | None = None) -> np.ndarray:
    """``decode_canonical_plane(...).array``, for the readers that want pixels."""

    return decode_canonical_plane(path, declared=declared).array
