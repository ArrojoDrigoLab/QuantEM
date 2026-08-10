"""Pluggable readers for image volumes — TIFF and PNG only.

The 2D pipeline canonicalizes a single uploaded TIFF or PNG into an 8-bit
grayscale PNG (see ``assets/tasks.py`` / ``assets/utils.py``). For 3D volumes we
mirror that design: a source volume is decoded here into a uniform,
lazily-readable plane interface, and ``assets/volume_tasks.py`` turns the
selected planes into the canonical multi-page OME-TIFF.

Supported sources:
  * ``.tif`` / ``.tiff`` / OME-TIFF     -> tifffile (lazy, per-plane page reads)
  * ``.png``                            -> Pillow (single-plane volume)
  * a directory of ``.tif``/``.png``    -> tifffile/Pillow (one file per z-plane)

Everything else is refused with :class:`UnsupportedVolumeSource`, whose message
names the formats that *are* accepted (see :data:`SUPPORTED_SOURCE_DESCRIPTION`).
``.mrc``/``.nd2``/``.dm3``/``.dm4`` and video containers are not supported in
v1: those readers and their dependencies (``mrcfile``, ``nd2``,
``rosettasciio``, OpenCV) were dropped per the owner ruling of 2026-08-06. Restoring one means re-adding its suffix set, its reader class and
its dispatch branch here.

Only ``numpy`` and ``tifffile`` are imported at module load; Pillow is imported
inside the readers that need it so importing this module stays cheap (it is
pulled in from the Django URLconf path).

Each reader exposes a single 2D ``(height, width)`` plane at a time in the
source's native dtype. Channel reduction (to the first channel) and 8-bit
windowing happen later in the encoder, exactly as for the 2D path.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

# Extensions handled by single-file readers.
TIFF_SUFFIXES = {".tif", ".tiff", ".ome.tif", ".ome.tiff"}
PNG_SUFFIXES = {".png"}
# Per-slice image-sequence members when reading a directory. Anything else in
# the directory (READMEs, sidecar JSON, other image formats) is ignored.
IMAGE_SEQUENCE_SUFFIXES = {".tif", ".tiff", ".png"}

# Single user-facing sentence naming what QuantEM can read; it is what an
# unsupported file is rejected with. The upload validator in ``assets/utils.py``
# derives its accepted-extension list from the suffix sets above, so the API and
# the readers cannot drift apart.
SUPPORTED_SOURCE_DESCRIPTION = (
    "TIFF (.tif, .tiff, .ome.tif, .ome.tiff) and PNG (.png), "
    "or a directory of .tif/.tiff/.png slices"
)


class UnsupportedVolumeSource(ValueError):
    """Raised when a path cannot be matched to any volume reader."""


@dataclass
class VolumeMetadata:
    """Normalized, JSON-serializable description of a source volume.

    Voxel sizes are stored in nanometres with ``None`` for unknown axes. The
    ordering convention everywhere in this module is ``(z, y, x)``.
    """

    source_format: str
    depth: int
    height: int
    width: int
    channels: int = 1
    dtype: str = "uint8"
    bit_depth: int = 8
    # (z, y, x) voxel size in nm; any component may be None when unknown.
    voxel_size_nm: tuple[float | None, float | None, float | None] = (None, None, None)
    axes: str = "ZYX"
    # Preserved, bounded, source-specific acquisition metadata.
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "depth": self.depth,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "dtype": self.dtype,
            "bit_depth": self.bit_depth,
            "voxel_size_nm": list(self.voxel_size_nm),
            "axes": self.axes,
            "extra": _json_safe(self.extra),
        }


class VolumeSource:
    """Base class: lazy random access to 2D planes of a 3D volume.

    Subclasses implement :meth:`read_plane`. Instances are context managers and
    should be closed to release file handles / memmaps.
    """

    metadata: VolumeMetadata

    def read_plane(self, z: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def iter_planes(self, indices: Iterable[int]) -> Iterator[np.ndarray]:
        for z in indices:
            yield self.read_plane(z)

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self) -> VolumeSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def read_volume_source(path: str | Path) -> VolumeSource:
    """Open ``path`` as a volume, choosing a reader by kind/extension.

    A directory is read as a per-slice image sequence. A file is matched by
    suffix. Raises :class:`UnsupportedVolumeSource` if nothing matches.
    """

    source = Path(path)
    if source.is_dir():
        return _ImageSequenceVolumeSource(source)

    suffix = _full_suffix(source)
    if suffix in TIFF_SUFFIXES:
        return _TiffVolumeSource(source)
    if suffix in PNG_SUFFIXES:
        return _PngVolumeSource(source)
    raise UnsupportedVolumeSource(
        f"Unsupported image format '{suffix or source.name}'. "
        f"QuantEM reads {SUPPORTED_SOURCE_DESCRIPTION}."
    )


def probe_volume_source(path: str | Path) -> VolumeMetadata:
    """Open ``path``, read its metadata, and close it again."""

    with read_volume_source(path) as source:
        return source.metadata


# --------------------------------------------------------------------------- #
# TIFF / OME-TIFF
# --------------------------------------------------------------------------- #
class _TiffVolumeSource(VolumeSource):
    """Reads (OME-)TIFF stacks one z-plane at a time.

    Primary path reads a single page per plane via ``tifffile.imread(key=...)``
    which only touches the bytes for that plane. When the page count does not
    match the series geometry (e.g. ImageJ contiguous hyperstacks stored in one
    IFD), it falls back to a lazily-decoded, cached series array.
    """

    def __init__(self, path: Path):
        self._path = path
        self._tif = tifffile.TiffFile(str(path))
        series = self._tif.series[0]
        axes = (series.axes or "").upper()
        shape = tuple(int(s) for s in series.shape)
        if not axes or len(axes) != len(shape):
            axes = _fallback_axes(len(shape))

        self._axes = axes
        self._shape = shape
        self._depth_axis, depth = _depth_axis_and_count(axes, shape)
        height, width = _spatial_hw(axes, shape)
        channels = _channel_count(axes, shape)
        dtype = np.dtype(series.dtype)

        # Decide whether per-page reads line up with the series geometry.
        non_spatial = [ax for ax in axes if ax not in ("Y", "X")]
        self._non_spatial_axes = non_spatial
        self._non_spatial_shape = [shape[axes.index(ax)] for ax in non_spatial]
        expected_pages = int(np.prod(self._non_spatial_shape)) if non_spatial else 1
        self._use_pages = len(self._tif.pages) == expected_pages and expected_pages > 0
        self._cached: np.ndarray | None = None

        self.metadata = VolumeMetadata(
            source_format="ome-tiff" if self._tif.ome_metadata else "tiff",
            depth=depth,
            height=height,
            width=width,
            channels=channels,
            dtype=dtype.name,
            bit_depth=_bit_depth_for_dtype(dtype),
            voxel_size_nm=self._read_voxel_size(),
            axes=axes,
            extra=self._read_extra(),
        )

    def _read_voxel_size(self) -> tuple[float | None, float | None, float | None]:
        z = y = x = None
        self._calibration_conflict: str | None = None
        ome = self._tif.ome_metadata
        if ome:
            x = _ome_physical_size(ome, "X")
            y = _ome_physical_size(ome, "Y")
            z = _ome_physical_size(ome, "Z")
        if (x is None or y is None) or z is None:
            ij = _imagej_calibration(self._tif)
            page = self._tif.pages[0]
            if z is None and ij.get("spacing"):
                try:
                    z = _to_nm(float(ij["spacing"]), ij.get("unit"))
                except (TypeError, ValueError):
                    z = None
            res_unit = ij.get("unit")
            conflicts: list[str] = []
            if x is None:
                x, conflict = _resolution_tag_nm_and_conflict(
                    page, "XResolution", res_unit
                )
                if conflict:
                    conflicts.append(conflict)
            if y is None:
                y, conflict = _resolution_tag_nm_and_conflict(
                    page, "YResolution", res_unit
                )
                if conflict:
                    conflicts.append(conflict)
            if conflicts:
                # X and Y almost always produce the same sentence; keep one.
                self._calibration_conflict = " ".join(dict.fromkeys(conflicts))
        return (z, y, x)

    def _read_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self._tif.ome_metadata:
            extra["ome_xml"] = _truncate(self._tif.ome_metadata)
        if self._tif.imagej_metadata:
            extra["imagej"] = _json_safe(self._tif.imagej_metadata)
        # Recorded by _read_voxel_size (always evaluated first: it is the
        # earlier keyword in the VolumeMetadata construction above); persists
        # through volume_metadata["source"]["extra"] so the serializer can
        # surface it as file_declared_pixel_size_caveat.
        if getattr(self, "_calibration_conflict", None):
            extra["calibration_conflict"] = self._calibration_conflict
        return extra

    def _page_index(self, z: int) -> int:
        idx = [
            z if ax == self._depth_axis else 0 for ax in self._non_spatial_axes
        ]
        if not idx:
            return 0
        return int(np.ravel_multi_index(idx, self._non_spatial_shape))

    def read_plane(self, z: int) -> np.ndarray:
        if self._use_pages:
            plane = tifffile.imread(str(self._path), key=self._page_index(z))
            return _as_2d_plane(np.asarray(plane))
        if self._cached is None:
            self._cached = self._tif.series[0].asarray(out="memmap")
        return _plane_from_array(self._cached, self._axes, self._depth_axis, z)

    def close(self) -> None:
        self._cached = None
        self._tif.close()


# --------------------------------------------------------------------------- #
# Per-slice image sequence (a directory of files, one per z-plane)
# --------------------------------------------------------------------------- #
class _ImageSequenceVolumeSource(VolumeSource):
    """Reads a directory of per-slice images, one file per z-plane.

    Members are TIFF or PNG (:data:`IMAGE_SEQUENCE_SUFFIXES`); files of any
    other type in the directory are ignored, and a directory with no usable
    member is refused rather than silently read as an empty volume. Slices are
    ordered by natural (human) sort so ``slice_2`` precedes ``slice_10``.
    """

    def __init__(self, directory: Path):
        files = sorted(
            (
                p
                for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SEQUENCE_SUFFIXES
            ),
            key=lambda p: _natural_key(p.name),
        )
        if not files:
            accepted = ", ".join(sorted(IMAGE_SEQUENCE_SUFFIXES))
            raise UnsupportedVolumeSource(
                f"No readable image slices in directory: {directory}. "
                f"An image sequence must contain files ending in {accepted}."
            )
        self._files = files
        first = self._read_file(0)
        height, width = first.shape[:2]
        channels = 1 if first.ndim == 2 else int(first.shape[-1])
        dtype = first.dtype
        self.metadata = VolumeMetadata(
            source_format="image-sequence",
            depth=len(files),
            height=int(height),
            width=int(width),
            channels=channels,
            dtype=dtype.name,
            bit_depth=_bit_depth_for_dtype(dtype),
            axes="ZYX",
            extra={"member_count": len(files), "first_member": files[0].name},
        )

    def _read_file(self, index: int) -> np.ndarray:
        path = self._files[index]
        if path.suffix.lower() in (".tif", ".tiff"):
            return np.asarray(tifffile.imread(str(path)))
        return _read_png_array(path)

    def read_plane(self, z: int) -> np.ndarray:
        return _as_2d_plane(self._read_file(z))


# --------------------------------------------------------------------------- #
# Single PNG (a one-plane "volume")
# --------------------------------------------------------------------------- #
class _PngVolumeSource(VolumeSource):
    """Reads a single PNG as a depth-1 volume via Pillow.

    Geometry comes from the PNG header (``Image.open`` is lazy), so opening the
    source to probe it never decodes the pixels; the plane is decoded on the
    first :meth:`read_plane`. PNG carries no physical-scale metadata, so
    ``voxel_size_nm`` stays unknown and the user's ``pixel_size_nm`` on the
    asset is the only source of scale.
    """

    def __init__(self, path: Path):
        self._path = path
        try:
            with _open_pillow(path) as img:
                width, height = int(img.size[0]), int(img.size[1])
                mode = str(img.mode)
        except Exception as exc:
            raise UnsupportedVolumeSource(f"Could not read PNG image {path}: {exc}") from exc

        dtype_name, channels = _PNG_MODE_INFO.get(mode, ("uint8", 1))
        dtype = np.dtype(dtype_name)
        self.metadata = VolumeMetadata(
            source_format="png",
            depth=1,
            height=height,
            width=width,
            channels=channels,
            dtype=dtype.name,
            bit_depth=_bit_depth_for_dtype(dtype),
            axes="ZYX",
            extra={"pil_mode": mode},
        )

    def read_plane(self, z: int) -> np.ndarray:
        if int(z) != 0:
            raise IndexError(
                f"PNG source {self._path} has a single plane; requested plane {z}"
            )
        return _as_2d_plane(_read_png_array(self._path))


# --------------------------------------------------------------------------- #
# Axis / shape helpers
# --------------------------------------------------------------------------- #
def _full_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".ome.tif"):
        return ".ome.tif"
    if name.endswith(".ome.tiff"):
        return ".ome.tiff"
    return path.suffix.lower()


def _fallback_axes(ndim: int) -> str:
    return {2: "YX", 3: "ZYX", 4: "CZYX", 5: "TCZYX"}.get(ndim, "Q" * (ndim - 2) + "YX")


def _depth_axis_and_count(axes: str, shape: tuple[int, ...]) -> tuple[str | None, int]:
    axes = axes.upper()
    for candidate in ("Z", "I", "Q", "T"):
        if candidate in axes:
            return candidate, int(shape[axes.index(candidate)])
    return None, 1


def _channel_count(axes: str, shape: tuple[int, ...]) -> int:
    axes = axes.upper()
    for candidate in ("C", "S"):
        if candidate in axes:
            return int(shape[axes.index(candidate)])
    return 1


def _spatial_hw(axes: str, shape: tuple[int, ...]) -> tuple[int, int]:
    axes = axes.upper()
    height = int(shape[axes.index("Y")]) if "Y" in axes else int(shape[-2])
    width = int(shape[axes.index("X")]) if "X" in axes else int(shape[-1])
    return height, width


def _plane_from_array(
    array: np.ndarray, axes: str, depth_axis: str | None, z: int
) -> np.ndarray:
    """Index a multi-dimensional array down to a single 2D (Y, X) plane."""

    axes = axes.upper()
    slicer: list[Any] = []
    for ax in axes:
        if ax in ("Y", "X"):
            slicer.append(slice(None))
        elif ax == depth_axis:
            slicer.append(int(z))
        else:
            slicer.append(0)  # first channel / time / generic index
    return _as_2d_plane(np.asarray(array[tuple(slicer)]))


def _as_2d_plane(plane: np.ndarray) -> np.ndarray:
    """Collapse a possibly-multichannel plane to a single 2D channel."""

    plane = np.asarray(plane)
    if plane.ndim == 2:
        return plane
    if plane.ndim == 3:
        # Channel-last (e.g. RGB) or channel-first; take the first channel.
        if plane.shape[-1] <= 4:
            return plane[..., 0]
        return plane[0]
    # Higher-dim: take leading indices until 2D remains.
    while plane.ndim > 2:
        plane = plane[0]
    return plane


def _bit_depth_for_dtype(dtype: np.dtype) -> int:
    dtype = np.dtype(dtype)
    if dtype.kind in ("u", "i", "f"):
        return int(dtype.itemsize * 8)
    return 8


# --------------------------------------------------------------------------- #
# Voxel-size / metadata parsing helpers
# --------------------------------------------------------------------------- #
_UNIT_TO_NM = {
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometre": 1.0,
    "um": 1000.0,
    "µm": 1000.0,  # micro sign
    "μm": 1000.0,  # greek small mu
    "micron": 1000.0,
    "microns": 1000.0,
    "micrometer": 1000.0,
    "micrometre": 1000.0,
    "mm": 1_000_000.0,
    "millimeter": 1_000_000.0,
    "millimetre": 1_000_000.0,
    "cm": 10_000_000.0,
    "centimeter": 10_000_000.0,
    "centimetre": 10_000_000.0,
    "a": 0.1,
    "angstrom": 0.1,
    "å": 0.1,
    "pm": 0.001,
}


def _to_nm(value: float, unit: str | None) -> float | None:
    if value is None or value <= 0:
        return None
    factor = _UNIT_TO_NM.get((unit or "nm").strip().lower(), 1.0)
    return float(value) * factor


def _length_unit_to_nm(unit: str | None) -> float | None:
    """Nanometres per ``unit`` for recognised length units; ``None`` otherwise.

    Unlike :func:`_to_nm` this never defaults to nanometres: an ImageJ
    ``unit=pixel`` (or any unrecognised string) is *not* a physical length, and
    assuming nm would fabricate a calibration from an uncalibrated file.
    """
    if not unit:
        return None
    return _UNIT_TO_NM.get(str(unit).strip().lower())


def _ome_physical_size(ome_xml: str, axis: str) -> float | None:
    size = _xml_attr(ome_xml, f"PhysicalSize{axis}")
    if size is None:
        return None
    unit = _xml_attr(ome_xml, f"PhysicalSize{axis}Unit") or "um"
    return _to_nm(size, unit)


def _xml_attr(xml: str, name: str):
    match = re.search(rf'{name}="([^"]+)"', xml)
    if not match:
        return None
    raw = match.group(1)
    try:
        return float(raw)
    except ValueError:
        return raw


#: Baseline TIFF ``ResolutionUnit`` tag values (TIFF 6.0 §Image Description).
#: 1 means the resolution is a unitless aspect ratio and carries no physical
#: scale, so it must not be converted to nanometres at all.
_TIFF_RESOLUTION_UNIT_NM: dict[int, float | None] = {
    1: None,            # no absolute unit
    2: 25_400_000.0,    # inch
    3: 10_000_000.0,    # centimetre
}

_TIFF_RESOLUTION_UNIT_NAME = {2: "inch", 3: "centimetre"}

#: An ImageJ-style ImageDescription block: a first line that is either
#: ``ImageJ=<version>`` (what Fiji itself writes, and what tifffile parses) or
#: a bare ``ImageJ`` (what some vendor writers emit — tifffile does *not*
#: recognise those, which is why this module parses the block itself).
_IMAGEJ_DESCRIPTION_RE = re.compile(r"\AImageJ(?:=[^\r\n]*)?(?:\r?\n|\Z)")


def _imagej_description_meta(page) -> dict[str, str]:
    """``key=value`` lines of an ImageJ-style ImageDescription, or ``{}``.

    Vendor blocks append their own ``key=value`` lines after the ImageJ ones
    (the pancreas TEM that motivated this carries an ``AppFive`` section with
    ``PixelScaleX=...``); the first occurrence of a key wins, so the ImageJ
    lines at the top cannot be overridden by a later vendor key.
    """
    tag = page.tags.get("ImageDescription")
    text = getattr(tag, "value", None)
    if not isinstance(text, str) or _IMAGEJ_DESCRIPTION_RE.match(text) is None:
        return {}
    meta: dict[str, str] = {}
    for line in text.splitlines()[1:]:
        key, sep, value = line.partition("=")
        if sep and key.strip():
            meta.setdefault(key.strip(), value.strip())
    return meta


def _imagej_calibration(tif) -> dict[str, Any]:
    """ImageJ calibration keys (``unit``, ``spacing``, ...) for a TiffFile.

    Prefers tifffile's parsed ``imagej_metadata`` (typed values, and it also
    handles the metadata stored in the separate IJMetadata tag); falls back to
    parsing the raw ImageDescription for the vendor variants tifffile rejects
    (a bare ``ImageJ`` first line instead of ``ImageJ=<version>``).
    """
    parsed = tif.imagej_metadata
    if parsed:
        return dict(parsed)
    try:
        page = tif.pages[0]
    except (IndexError, AttributeError):
        return {}
    return _imagej_description_meta(page)


def _resolution_tag_nm(page, tag_name: str, unit: str | None) -> float | None:
    """Physical pixel size in nm; see :func:`_resolution_tag_nm_and_conflict`."""
    return _resolution_tag_nm_and_conflict(page, tag_name, unit)[0]


def _resolution_tag_nm_and_conflict(
    page, tag_name: str, unit: str | None
) -> tuple[float | None, str | None]:
    """Physical size of one pixel in nm from ``XResolution``/``YResolution``.

    The resolution tags record *pixels per unit*. Two different conventions
    say what that unit is, and this is where the precedence between them is
    decided:

    1. **ImageJ/Fiji**: ``ResolutionUnit=NONE`` and a ``unit=<u>`` line in the
       ImageDescription (passed in as ``unit``; see :func:`_imagej_calibration`).
       This is the most common calibrated-EM layout in the wild.
    2. **Baseline TIFF**: the ``ResolutionUnit`` tag (inch/centimetre).

    Rule: a recognised ImageJ length unit wins, because it is the writer's
    deliberate calibration statement — Fiji sets ``ResolutionUnit=NONE``
    precisely because inch/cm cannot express its unit, and writers that leave a
    real ``ResolutionUnit`` behind alongside an ImageJ unit are keeping a
    library default, not making a second claim. But when the two *both* yield a
    physical size and disagree (beyond 0.1%), the loser is not silently
    dropped: the disagreement is returned as a conflict note, which travels
    with the asset's calibration provenance (``pixel_size_caveat`` on the
    source metadata, ``file_declared_pixel_size_caveat`` on the API payload).
    ``ResolutionUnit=NONE``/absent makes no physical claim, so it never
    conflicts. An unrecognised ImageJ unit (``pixel``, ...) is not a length
    and falls through to the baseline tag (:func:`_length_unit_to_nm`).

    (Historically this consulted *only* the ImageJ unit string tifffile had
    parsed, so a plain TIFF written in pixels-per-centimetre came back off by
    1e7, and a Fiji-convention file whose block tifffile did not recognise
    imported uncalibrated.)
    """
    tag = page.tags.get(tag_name)
    if tag is None:
        return None, None
    try:
        numerator, denominator = tag.value
        if numerator == 0:
            return None, None
        pixels_per_unit = numerator / denominator
        if pixels_per_unit <= 0:
            return None, None
        unit_size = 1.0 / pixels_per_unit  # source units per pixel
    except Exception:
        return None, None

    imagej_factor = _length_unit_to_nm(unit)
    imagej_nm = unit_size * imagej_factor if imagej_factor else None

    res_unit_tag = page.tags.get("ResolutionUnit")
    raw = getattr(res_unit_tag, "value", None)
    try:
        res_unit = int(raw)
    except (TypeError, ValueError):
        res_unit = None
    factor = _TIFF_RESOLUTION_UNIT_NM.get(res_unit) if res_unit is not None else None
    tag_nm = unit_size * factor if factor is not None else None

    if imagej_nm is not None:
        conflict = None
        if tag_nm is not None and not math.isclose(imagej_nm, tag_nm, rel_tol=1e-3):
            tag_unit = _TIFF_RESOLUTION_UNIT_NAME.get(res_unit, f"unit {res_unit}")
            conflict = (
                f"The pixel size was read from the file's ImageJ calibration "
                f"(unit '{unit}': {imagej_nm:.4g} nm/pixel), but its baseline "
                f"TIFF ResolutionUnit tag disagrees ({tag_unit}: "
                f"{tag_nm:.4g} nm/pixel). The ImageJ calibration was used."
            )
        return imagej_nm, conflict
    return tag_nm, None


# --------------------------------------------------------------------------- #
# PNG helpers (Pillow)
# --------------------------------------------------------------------------- #
# Pillow mode -> (numpy dtype name, channel count). Anything unlisted is treated
# as 8-bit single-channel, which is what ``_as_2d_plane`` reduces it to anyway.
_PNG_MODE_INFO: dict[str, tuple[str, int]] = {
    "1": ("uint8", 1),
    "L": ("uint8", 1),
    "LA": ("uint8", 2),
    "P": ("uint8", 1),
    "PA": ("uint8", 2),
    "RGB": ("uint8", 3),
    "RGBA": ("uint8", 4),
    "I": ("int32", 1),
    "I;16": ("uint16", 1),
    "I;16B": ("uint16", 1),
    "I;16L": ("uint16", 1),
    "F": ("float32", 1),
}


def _open_pillow(path: Path):
    """``Image.open`` with the decompression-bomb guard lifted.

    Electron-microscopy PNGs routinely exceed Pillow's default 89M-pixel limit;
    the same override is applied in ``assets/utils.py`` for the 2D path.
    """

    from PIL import Image  # core dep; local import keeps module import cheap

    Image.MAX_IMAGE_PIXELS = None
    return Image.open(path)


def _read_png_array(path: Path) -> np.ndarray:
    with _open_pillow(path) as img:
        return np.asarray(img)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _natural_key(name: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def _truncate(text: str, limit: int = 20000) -> str:
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of arbitrary metadata into JSON-serializable form."""

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else f"<ndarray shape={value.shape}>"
    return str(value)
