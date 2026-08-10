"""Shared helpers for resolving on-disk image asset paths."""

from __future__ import annotations

from pathlib import Path

from quantem.core.config import DATA_DIR
from quantem.core.local_storage import resolve_stored_path


def get_file_absolute_path(image) -> Path:
    """
    Resolve the absolute path to an image-backed file.

    Prefers a caller-provided, already-resolved ``absolute_path`` (e.g. an
    ``AssetOpenable`` whose ``absolute_path`` property is the live-resolved
    path). Otherwise resolves the relative ``file_path`` under the single
    settings-defined storage root. Never trusts a stored absolute path that
    does not exist on disk.

    Raises:
        FileNotFoundError: If the resolved path does not exist on disk.
    """
    candidate = str(getattr(image, "absolute_path", "") or "").strip()
    if candidate:
        file_path = Path(candidate).expanduser().resolve(strict=False)
        if file_path.exists():
            return file_path

    file_rel_path = getattr(image, "file_path", "") or ""
    file_path = resolve_stored_path(file_rel_path, relative_to=DATA_DIR)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path
