"""The oracle. **This module must never import anything from ``quantem``.**

Every pixel assertion in ``test_ngff_source_matrix`` and in the race harnesses
is compared against this file, so the suite cannot agree with itself by
construction. It is a second implementation of the *rule*, written from the
rule -- magic bytes, band 0, native-range scaling, named refusals -- and not
from ``assets/canonical_decode.py``.

Three rounds of hand-written tests passed against a product that was serving a
white rectangle, because every one of them compared the product with itself.
Do not "refactor this to share code". If it ever imports ``quantem``, the
chokepoint test below fails and it should.

The self-tests at the bottom exercise the oracle itself: an oracle nobody
checks is just a second place to be wrong.
"""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
TIFF_MAGICS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


class ReferenceRefusal(ValueError):
    """The reference decoder will not represent these pixels as 8-bit gray."""


def sniff(path) -> str:
    head = Path(path).read_bytes()[:8]
    if head.startswith(PNG_MAGIC):
        return "png"
    for magic in TIFF_MAGICS:
        if head.startswith(magic):
            return "tiff"
    raise ReferenceRefusal(f"unrecognised container: {head!r}")


def _band0(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if array.shape[-1] <= 4 and array.shape[0] > 4:
            return array[:, :, 0]
        return array[0]
    squeezed = np.squeeze(array)
    if squeezed.ndim == 2:
        return squeezed
    if squeezed.ndim == 3:
        return _band0(squeezed)
    raise ReferenceRefusal(f"unsupported shape {array.shape}")


def _to_uint8(plane: np.ndarray) -> np.ndarray:
    dtype = plane.dtype
    if np.issubdtype(dtype, np.complexfloating):
        raise ReferenceRefusal(f"complex pixel data ({dtype})")
    if dtype == np.bool_:
        return (plane.astype(np.uint8) * 255).astype(np.uint8)
    if np.issubdtype(dtype, np.floating):
        return np.nan_to_num(np.clip(plane.astype(np.float32), 0, 255), nan=0).astype(np.uint8)
    if np.issubdtype(dtype, np.signedinteger):
        if plane.size and int(plane.min()) < 0:
            raise ReferenceRefusal(f"signed integer data with negative values ({dtype})")
        native_max = float(2 ** (dtype.itemsize * 8) - 1)
        return np.clip(plane.astype(np.float64) * (255.0 / native_max), 0, 255).astype(np.uint8)
    if np.issubdtype(dtype, np.unsignedinteger):
        if dtype.itemsize == 1:
            return plane.astype(np.uint8)
        native_max = float(2 ** (dtype.itemsize * 8) - 1)
        scaled = plane.astype(np.float32) * (255.0 / native_max)
        return np.nan_to_num(np.clip(scaled, 0, 255), nan=0).astype(np.uint8)
    raise ReferenceRefusal(f"unsupported dtype {dtype}")


def read_array(path) -> np.ndarray:
    if sniff(path) == "tiff":
        return np.asarray(tifffile.imread(str(path)))
    with Image.open(path) as handle:
        handle.load()
        if handle.mode in {"I;16", "I;16B", "I;16L"}:
            return np.asarray(handle, dtype=np.uint16)
        if handle.mode == "I":
            return np.asarray(handle, dtype=np.int32)
        if handle.mode == "F":
            return np.asarray(handle, dtype=np.float32)
        return np.asarray(handle)


def decode(path) -> np.ndarray:
    """The canonical 8-bit grayscale plane for ``path``, or a named refusal."""

    return np.ascontiguousarray(_to_uint8(_band0(read_array(path))))


def downsample(plane: np.ndarray) -> np.ndarray:
    """One pyramid step: 2x2 box mean over an edge-padded plane, rint to uint8."""

    height = max(1, -(-plane.shape[0] // 2))
    width = max(1, -(-plane.shape[1] // 2))
    padded = np.pad(
        plane,
        ((0, max(0, height * 2 - plane.shape[0])), (0, max(0, width * 2 - plane.shape[1]))),
        mode="edge",
    )[: height * 2, : width * 2]
    return np.rint(padded.reshape(height, 2, width, 2).astype(np.float64).mean(axis=(1, 3))).astype(
        np.uint8
    )


def pyramid(plane: np.ndarray) -> list[np.ndarray]:
    levels = [plane]
    while min(levels[-1].shape) > 1:
        nxt = downsample(levels[-1])
        if nxt.shape == levels[-1].shape:
            break
        levels.append(nxt)
    return levels


def canonical_png_bytes(plane: np.ndarray, *, compress_level: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(plane, mode="L").save(
        buffer, "PNG", compress_level=compress_level, optimize=False
    )
    return buffer.getvalue()


def thumbnail(plane: np.ndarray, max_size: int) -> np.ndarray:
    image = Image.fromarray(plane, mode="L")
    image.thumbnail((max_size, max_size))
    return np.array(image, dtype=np.uint8)


# ---------------------------------------------------------------------------
# The oracle's own tests
# ---------------------------------------------------------------------------


def test_the_oracle_never_imports_the_thing_it_is_checking():
    """A comparand that imports the app is not a comparand."""

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.ImportFrom) and node.level:
            imported.append("." * node.level)
    offenders = [name for name in imported if name.split(".")[0] in {"quantem", ""}]
    assert offenders == [], (
        f"test_ngff_reference.py imports {offenders} from the application it is "
        "supposed to be an independent check on"
    )


def test_the_oracle_scales_by_native_range_rather_than_saturating():
    plane = np.array([[0, 32768, 65535]], dtype=np.uint16)
    assert _to_uint8(plane).tolist() == [[0, 127, 255]]


def test_the_oracle_takes_band_zero_from_both_layouts():
    interleaved = np.zeros((6, 6, 3), dtype=np.uint8)
    interleaved[..., 0] = 7
    interleaved[..., 1] = 200
    planar = np.zeros((3, 6, 6), dtype=np.uint8)
    planar[0] = 7
    planar[1] = 200
    assert _band0(interleaved).max() == 7
    assert _band0(planar).max() == 7


def test_the_oracle_refuses_complex_and_negative_signed_data():
    for array, fragment in (
        (np.zeros((2, 2), dtype=np.complex64), "complex"),
        (np.array([[-1, 2]], dtype=np.int32), "negative"),
    ):
        try:
            _to_uint8(array)
        except ReferenceRefusal as exc:
            assert fragment in str(exc)
        else:  # pragma: no cover - a silent accept is the bug
            raise AssertionError(f"expected a refusal naming {fragment!r}")
