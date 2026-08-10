"""TIFF/PNG read-write helpers for data-prep.

Reading uses ``tifffile``, which handles the deflate-compressed and samples-packed OpenOrganelle
tiles that PIL mis-reads as RGB. Writing uses PIL single-channel ``"L"`` PNGs, matching the
single-channel PNG encoding the pretraining tile shards use (``em_ssl.data.shard_writer``). numpy is
used for indexing and elementwise work only, never BLAS matmul, which some environments abort on.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

def _write_with_retry(path, fn, tries: int = 6, delay: float = 1.5) -> None:
    """Run ``fn()`` (which writes ``path``), retrying transient OSErrors.

    On network storage a single write can occasionally fail despite ample free space, and a build
    writing tens of thousands of files must not die on one glitch. Any partial file is removed
    between attempts; the last error is raised if all attempts fail.
    """
    last = None
    for i in range(tries):
        try:
            fn()
            return
        except OSError as e:
            last = e
            try:
                p = Path(path)
                if p.exists():
                    p.unlink()
            except OSError:
                pass
            time.sleep(delay * (i + 1))
    raise last

def read_tif(path: str | Path) -> np.ndarray:
    """Read a TIFF as a numpy array. Returns whatever shape tifffile yields (2D, 3D, or packed)."""
    import tifffile

    return tifffile.imread(str(path))

def read_planes(path: str | Path) -> np.ndarray:
    """Read a (possibly multi-plane) raw tile, normalised to ``(planes, rows, cols)``.

    Handles the OpenOrganelle "samples-packed" layout where a few z-planes are stored in the
    samples-per-pixel axis and come back as ``(rows, cols, planes)``.
    """
    a = np.asarray(read_tif(path))
    if a.ndim == 2:
        return a[None]
    if a.ndim == 3:
        # tifffile usually returns (planes, rows, cols). If the *last* axis is the small one
        # (samples-packed) and the first axis is large, move it to front.
        if a.shape[-1] <= 16 and a.shape[0] > 16:
            a = np.moveaxis(a, -1, 0)
        return a
    raise ValueError(f"Unexpected raw tile ndim={a.ndim} for {path}")

def write_png_L(path: str | Path, arr: np.ndarray) -> None:
    """Write a single-channel uint8 array as a ``mode='L'`` PNG (lossless)."""
    from PIL import Image

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    _write_with_retry(path, lambda: Image.fromarray(a, mode="L").save(str(path), format="PNG"))

def read_png_L(path: str | Path) -> np.ndarray:
    """Read a single-channel PNG as a uint8 numpy array."""
    from PIL import Image

    return np.asarray(Image.open(str(path)).convert("L"), dtype=np.uint8)

def write_tif_u16(path: str | Path, arr: np.ndarray) -> None:
    """Write an instance-id map as a uint16 TIFF (per-crop instance counts are well under 65535)."""
    import tifffile

    a = np.asarray(arr)
    if a.max() > 65535:  # extremely unlikely per crop; relabel densely if it ever happens
        _, inv = np.unique(a, return_inverse=True)
        a = inv.reshape(a.shape)
    _write_with_retry(path, lambda: tifffile.imwrite(str(path), a.astype(np.uint16)))

def read_tif_u16(path: str | Path) -> np.ndarray:
    return np.asarray(read_tif(path)).astype(np.int32)

def write_json(path: str | Path, obj) -> None:
    _write_with_retry(path, lambda: Path(path).write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8"))

def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
