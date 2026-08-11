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
from typing import Any, NamedTuple

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
        self._calibration_source: str | None = None
        ome = self._tif.ome_metadata
        if ome:
            x = _ome_physical_size(ome, "X")
            y = _ome_physical_size(ome, "Y")
            z = _ome_physical_size(ome, "Z")
            if x is not None or y is not None:
                self._calibration_source = PIXEL_SIZE_SOURCE_OME
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
            # Goes through the same composed reader as the 2D import
            # (``utils._tiff_pixel_size_nm``) so the vendor tag, the ImageJ
            # unit and the baseline tag are ranked identically here: the same
            # file must not calibrate differently depending on whether it was
            # imported as an image or as a volume.
            if x is None:
                x, conflict, source = in_plane_pixel_size_nm(
                    page, "XResolution", res_unit
                )
                if conflict:
                    conflicts.append(conflict)
                if x is not None and self._calibration_source is None:
                    self._calibration_source = source
            if y is None:
                y, conflict, source = in_plane_pixel_size_nm(
                    page, "YResolution", res_unit
                )
                if conflict:
                    conflicts.append(conflict)
                if y is not None and self._calibration_source is None:
                    self._calibration_source = source
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
        # Which tag or block the voxel size came from. Same purpose and same
        # vocabulary as ``source_metadata.pixel_size_source`` on a 2D import;
        # surfaced as file_declared_pixel_size_source.
        if getattr(self, "_calibration_source", None):
            extra["calibration_source"] = self._calibration_source
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


#: TIFF tag 51023, ``FibicsXML``: the whole acquisition record Zeiss/Fibics
#: ATLAS writes into a private tag as an XML string. MEASURED over 139 TIFFs in
#: this lab's corpus: 97 carry it, every one of them with
#: ``Software = 'Fibics ATLAS Export'``, ``ResolutionUnit = NONE`` and either no
#: ``XResolution`` or ``XResolution = (1, 1)`` -- so before this tag was read,
#: every calibrated Zeiss Atlas export in the lab imported uncalibrated.
FIBICS_XML_TAG = 51023

#: Provenance labels for :class:`PixelSizeReading`. Each one names the tag or
#: block that supplied the number, because "5.229 nm/pixel" on a library card is
#: a different claim depending on whether the microscope wrote it, Fiji wrote
#: it, or a person typed it.
PIXEL_SIZE_SOURCE_OME = "OME-XML PhysicalSize"
PIXEL_SIZE_SOURCE_FIBICS = "TIFF tag 51023 (FibicsXML)"
PIXEL_SIZE_SOURCE_IMAGEJ = "ImageJ ImageDescription unit (TIFF tag 270)"
PIXEL_SIZE_SOURCE_RESOLUTION_UNIT = "TIFF XResolution/ResolutionUnit (tags 282/296)"

#: Above this a "pixel size" is not an electron-microscopy scale, it is a
#: parse accident. One metre per pixel is already absurd by nine orders of
#: magnitude; the bound only exists so a corrupt tag cannot produce a number.
_MAX_PLAUSIBLE_PIXEL_NM = 1e9


class PixelSizeReading(NamedTuple):
    """What a file said its in-plane pixel size is, and who said it.

    ``caveat`` is the sentence shown to the user when two declarations in the
    same file disagree, or when one of them was rejected. ``source`` is one of
    the ``PIXEL_SIZE_SOURCE_*`` labels, or ``None`` when the file is silent.
    """

    nm: float | None
    caveat: str | None
    source: str | None


def _fibics_xml(page) -> str | None:
    """The ``FibicsXML`` string of TIFF tag 51023, or ``None``.

    Deliberately *not* handed to an XML parser: the block arrives inside an
    uploaded file, and ``xml.etree`` is documented as vulnerable to entity
    expansion. Every field this module wants is a leaf element holding a
    number, so a bounded regex over the string is both sufficient and safe --
    the same choice :func:`_xml_attr` already makes for OME-XML.
    """
    value = getattr(page.tags.get(FIBICS_XML_TAG), "value", None)
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str) or "<Fibics" not in value:
        return None
    return value


def _fibics_element(xml: str, name: str) -> tuple[float | None, str | None]:
    """``(number, units attribute)`` of a leaf ``<name ...>number</name>``."""
    match = re.search(rf"<{name}((?:\s[^>]*)?)>([^<]*)</{name}>", xml)
    if match is None:
        return None, None
    units = re.search(r'units\s*=\s*"([^"]*)"', match.group(1))
    try:
        return float(match.group(2).strip()), (units.group(1) if units else None)
    except ValueError:
        return None, None


def _fibics_axis_nm(xml: str, axis: str) -> float | None:
    """In-plane pixel size in nm along ``axis`` from the Fibics ``<Scan>`` block.

    ``<Ux>/<Uy>`` and ``<Vx>/<Vy>`` are the two scan step *vectors* in
    micrometres -- U from one pixel to the next along the image x axis, V along
    y -- so the pixel size is each vector's length, not its named component.
    That distinction is not academic: in every Atlas file in this lab's corpus
    ``<Uy>`` is exactly ``0`` and the y pixel size lives in ``<Vy>`` (negative,
    because the raster runs down), so reading ``<Ux>``/``<Uy>`` as the x and y
    sizes -- the obvious reading -- yields ``0`` for y. With ``<ScanRot>``
    non-zero the components mix and only the vector length is right.

    Falls back to ``FOV_X / Width``, which is the same number to 15 digits in
    every corpus file (checked) and is what survives if a future writer emits
    the field of view but not the step vectors.
    """
    first, second = ("Ux", "Uy") if axis == "X" else ("Vx", "Vy")
    a, _ = _fibics_element(xml, first)
    b, _ = _fibics_element(xml, second)
    if a is not None or b is not None:
        step_um = math.hypot(a or 0.0, b or 0.0)
        if step_um > 0:
            return step_um * 1000.0

    fov, fov_unit = _fibics_element(xml, "FOV_X" if axis == "X" else "FOV_Y")
    extent, _ = _fibics_element(xml, "Width" if axis == "X" else "Height")
    if fov and extent and extent > 0:
        factor = _length_unit_to_nm(fov_unit or "um")
        if factor:
            return fov / extent * factor
    return None


def _fibics_reading(page, axis: str) -> tuple[float | None, str | None]:
    """``(nm, note)`` from tag 51023 for one axis; ``(None, None)`` if absent.

    The block records the geometry of the *acquisition*. If someone has since
    binned or cropped the raster with a tool that preserves unknown TIFF tags,
    the scan step no longer describes these pixels -- and the block says so
    itself, because it carries its own ``<Width>``/``<Height>``. When those
    disagree with the IFD's dimensions the value is refused and the reason is
    returned as a note, rather than calibrating the file to a scale it no
    longer has.
    """
    xml = _fibics_xml(page)
    if xml is None:
        return None, None

    nm = _fibics_axis_nm(xml, axis)
    if nm is None or not math.isfinite(nm) or not 0 < nm <= _MAX_PLAUSIBLE_PIXEL_NM:
        return None, None

    declared_w, _ = _fibics_element(xml, "Width")
    declared_h, _ = _fibics_element(xml, "Height")
    actual_w = getattr(page, "imagewidth", None)
    actual_h = getattr(page, "imagelength", None)
    if (
        declared_w
        and declared_h
        and actual_w
        and actual_h
        and (int(declared_w) != int(actual_w) or int(declared_h) != int(actual_h))
    ):
        return None, (
            f"This file carries the microscope's calibration in "
            f"{PIXEL_SIZE_SOURCE_FIBICS} ({nm:.4g} nm/pixel), but that record "
            f"describes a {int(declared_w)} x {int(declared_h)} image and this "
            f"one is {int(actual_w)} x {int(actual_h)} pixels. The image has "
            f"been resized since acquisition, so the recorded scale was not "
            f"used."
        )
    return nm, None


def _join_notes(notes: Iterable[str | None]) -> str | None:
    kept = dict.fromkeys(note for note in notes if note)
    return " ".join(kept) or None


def in_plane_pixel_size_nm(page, tag_name: str, unit: str | None) -> PixelSizeReading:
    """The file's own in-plane pixel size for one axis, with its provenance.

    Composes the vendor tag with the resolution tags under one precedence, so
    that a 2D import and a 3D import of the same file cannot disagree:

    1. a recognised **ImageJ/Fiji unit** in the ImageDescription
       (:func:`_resolution_tag_reading`),
    2. the **Zeiss/Fibics ATLAS block** in TIFF tag 51023,
    3. the **baseline TIFF ``ResolutionUnit``** tag.

    ImageJ stays on top because that is the precedence this reader already had
    and the argument for it has not changed: a ``unit=`` line is a statement
    somebody made about *this* raster, after acquisition, and a re-save through
    Fiji is exactly the operation that can change the pixel size while leaving
    an acquisition-time vendor block behind. The vendor block goes above the
    baseline tag for the mirror-image reason given on
    :func:`_resolution_tag_reading`: a writer that sets ``ResolutionUnit`` to
    inches while recording nanometres in a private tag is keeping a library
    default, not making a second claim. (No file in this lab's corpus exercises
    that ordering: all 97 Fibics files have ``ResolutionUnit = NONE``, which
    makes no physical claim at all.)

    Whichever loses, the user is told. Any disagreement beyond 0.1% comes back
    as a caveat that travels with the asset (``pixel_size_caveat`` on the
    source metadata, ``file_declared_pixel_size_caveat`` on the API payload).
    """
    axis = "Y" if str(tag_name).upper().startswith("Y") else "X"
    vendor_nm, vendor_note = _fibics_reading(page, axis)
    tag_nm, tag_caveat, tag_source = _resolution_tag_reading(page, tag_name, unit)
    notes = [vendor_note, tag_caveat]

    if tag_source == PIXEL_SIZE_SOURCE_IMAGEJ and tag_nm is not None:
        if vendor_nm is not None and not math.isclose(vendor_nm, tag_nm, rel_tol=1e-3):
            notes.append(
                f"The pixel size was read from the file's ImageJ calibration "
                f"(unit '{unit}': {tag_nm:.4g} nm/pixel), but the microscope's "
                f"own record in {PIXEL_SIZE_SOURCE_FIBICS} disagrees "
                f"({vendor_nm:.4g} nm/pixel). The ImageJ calibration was used."
            )
        return PixelSizeReading(tag_nm, _join_notes(notes), PIXEL_SIZE_SOURCE_IMAGEJ)

    if vendor_nm is not None:
        if tag_nm is not None and not math.isclose(vendor_nm, tag_nm, rel_tol=1e-3):
            notes.append(
                f"The pixel size was read from the microscope's own record in "
                f"{PIXEL_SIZE_SOURCE_FIBICS} ({vendor_nm:.4g} nm/pixel), but "
                f"the file's baseline TIFF ResolutionUnit tag disagrees "
                f"({tag_nm:.4g} nm/pixel). The microscope's record was used."
            )
        return PixelSizeReading(vendor_nm, _join_notes(notes), PIXEL_SIZE_SOURCE_FIBICS)

    return PixelSizeReading(
        tag_nm, _join_notes(notes), tag_source if tag_nm is not None else None
    )


def _resolution_tag_nm(page, tag_name: str, unit: str | None) -> float | None:
    """Physical pixel size in nm; see :func:`_resolution_tag_nm_and_conflict`."""
    return _resolution_tag_nm_and_conflict(page, tag_name, unit)[0]


def _resolution_tag_nm_and_conflict(
    page, tag_name: str, unit: str | None
) -> tuple[float | None, str | None]:
    """``(nm, conflict)`` from the resolution tags; see
    :func:`_resolution_tag_reading`, which also names the source."""
    nm, caveat, _source = _resolution_tag_reading(page, tag_name, unit)
    return nm, caveat


def _resolution_tag_reading(
    page, tag_name: str, unit: str | None
) -> tuple[float | None, str | None, str | None]:
    """Physical size of one pixel in nm from ``XResolution``/``YResolution``.

    The resolution tags record *pixels per unit*. Two different conventions
    say what that unit is, and this is where the precedence between them is
    decided:

    1. **ImageJ/Fiji**: ``ResolutionUnit=NONE`` and a ``unit=<u>`` line in the
       ImageDescription (passed in as ``unit``; see :func:`_imagej_calibration`).
       This is the most common calibrated-EM layout in the wild.
    2. **Baseline TIFF**: the ``ResolutionUnit`` tag (inch/centimetre).

    A third convention -- a vendor's own private tag -- is folded in one level
    up, by :func:`in_plane_pixel_size_nm`; this function stays about the
    resolution tags alone. The third element of its return says which of the
    two conventions above produced the number, which is what lets the caller
    apply a precedence between all three and record the winner as provenance.

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
        return None, None, None
    try:
        numerator, denominator = tag.value
        if numerator == 0:
            return None, None, None
        pixels_per_unit = numerator / denominator
        if pixels_per_unit <= 0:
            return None, None, None
        unit_size = 1.0 / pixels_per_unit  # source units per pixel
    except Exception:
        return None, None, None

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
        return imagej_nm, conflict, PIXEL_SIZE_SOURCE_IMAGEJ
    if tag_nm is not None:
        return tag_nm, None, PIXEL_SIZE_SOURCE_RESOLUTION_UNIT
    return None, None, None


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
