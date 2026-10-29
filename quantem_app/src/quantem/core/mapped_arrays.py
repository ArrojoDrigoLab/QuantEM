"""Large working planes use temporary mappings in the application data root."""

from __future__ import annotations

import math
import tempfile

import numpy as np

MAX_HEAP_ARRAY_BYTES = 64 * 1024 * 1024


def working_array(shape, dtype=np.uint8) -> np.ndarray:
    """Allocate pageable scratch; the OS deletes its file with the last view."""
    dtype = np.dtype(dtype)
    if math.prod(shape) * dtype.itemsize < MAX_HEAP_ARRAY_BYTES:
        return np.empty(shape, dtype=dtype)
    from quantem.core.config import TMP_DIR

    directory = TMP_DIR / "mapped-arrays"
    directory.mkdir(parents=True, exist_ok=True)
    # The mapping owns the kernel file reference after this handle is closed.
    # TemporaryFile unlinks on POSIX / uses delete-on-close on Windows, so
    # crashes do not strand named multi-gigabyte scratch files.
    with tempfile.TemporaryFile(dir=directory) as stream:
        return np.memmap(stream, mode="w+", dtype=dtype, shape=shape)
