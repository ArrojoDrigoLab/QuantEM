"""Single-node local storage helpers.

This app is single-node: there is no remote/master storage, no cross-node sync,
and no node-routing. This module is the small local-only surface other modules
depend on:

* ``StorageError`` — error type used as an except target.
* ``StoragePath`` — a storage-root-relative path descriptor used by the job
  artifact registry and the local artifact-write leases.
* ``validate_storage_relpath`` / ``storage_path`` / ``storage_relpath_for_path``
  — enforce/derive storage-root-relative paths.
* ``ensure_cached_storage_path`` — resolve a relative path under the single
  settings-defined storage root (a relative path always resolves locally).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from quantem.core.config import STORAGE_DIR

StoragePathType = Literal["file", "dir"]


def path_value_is_absolute_like(value: str | Path | None) -> bool:
    """OS-independent "does this look absolute?" test.

    ``Path.is_absolute()`` alone is platform-dependent, and stored paths can have
    been written on another OS.
    """
    raw = str(value or "").strip()
    if not raw:
        return False
    normalized = raw.replace("\\", "/")
    # POSIX-absolute (covers //UNC, /mnt/<drive>, /srv, /Volumes, ...).
    if normalized.startswith("/"):
        return True
    # Windows drive-letter (C:/, D:/...).
    if len(normalized) >= 3 and normalized[0].isalpha() and normalized[1:3] == ":/":
        return True
    return Path(raw).expanduser().is_absolute()


class StorageError(RuntimeError):
    """Raised when a storage path is invalid."""


@dataclass(frozen=True)
class StoragePath:
    relpath: str
    path_type: StoragePathType = "file"
    required: bool = False
    lease_required: bool = False

    @property
    def is_dir(self) -> bool:
        return self.path_type == "dir"


@dataclass(frozen=True)
class StorageConfig:
    storage_dir: Path


def get_storage_config(
    *,
    env: dict[str, str] | None = None,
    storage_dir: str | Path | None = None,
) -> StorageConfig:
    return StorageConfig(
        storage_dir=Path(storage_dir or STORAGE_DIR).expanduser().resolve(strict=False),
    )


def validate_storage_relpath(raw_value: str | Path) -> str:
    normalized = str(raw_value).strip().replace("\\", "/")
    if not normalized:
        raise StorageError("Storage-relative paths must not be blank.")
    if path_value_is_absolute_like(normalized):
        raise StorageError(
            f"Storage-relative path must not be absolute: {raw_value!r}"
        )
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise StorageError(
            f"Storage-relative path must stay under storage root: {raw_value!r}"
        )
    return "/".join(parts)


def storage_path(relpath: str | Path, *, storage_dir: str | Path | None = None) -> Path:
    root = Path(storage_dir or STORAGE_DIR).expanduser().resolve(strict=False)
    return (root / validate_storage_relpath(relpath)).resolve(strict=False)


def storage_relpath_for_path(
    path: str | Path,
    *,
    storage_dir: str | Path | None = None,
) -> str:
    root = Path(storage_dir or STORAGE_DIR).expanduser().resolve(strict=False)
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise StorageError(f"Path is outside storage root: {resolved}") from exc
    return validate_storage_relpath(relative)


def ensure_cached_storage_path(
    relpath: str | Path,
    *,
    path_type: StoragePathType = "file",
    refresh_existing: bool = False,
    config: StorageConfig | None = None,
) -> Path:
    """Resolve a storage-relative path under the local storage root.

    Single-node: there is nothing to fetch; a relative path resolves locally.
    ``path_type`` / ``refresh_existing`` are accepted for backwards compatibility
    with existing call sites and are otherwise unused.
    """
    config = config or get_storage_config()
    return (config.storage_dir / validate_storage_relpath(relpath)).resolve(strict=False)


# ---------------------------------------------------------------------------
# Stored-path helpers.
#
# These are inlined here alongside `path_value_is_absolute_like`, which they use.
# ---------------------------------------------------------------------------


def _normalize_str_path(value: str | Path) -> str:
    return str(Path(value)).replace("\\", "/")


def resolve_stored_path(
    raw_value: str | Path,
    *,
    relative_to: str | Path,
) -> Path:
    """Resolve a stored path value against a root, honouring absolute values."""
    raw = str(raw_value).strip()
    if path_value_is_absolute_like(raw):
        return Path(raw).expanduser().resolve(strict=False)
    return (Path(str(relative_to)).expanduser() / Path(raw)).resolve(strict=False)


def normalize_stored_path_value(
    raw_value: str | Path,
    *,
    relative_to: str | Path,
) -> str:
    """Reduce a stored path to a root-relative, forward-slash string when possible."""
    raw = str(raw_value).strip()
    if not path_value_is_absolute_like(raw):
        return _normalize_str_path(raw)
    root_path = Path(str(relative_to)).expanduser().resolve(strict=False)
    resolved = Path(raw).expanduser().resolve(strict=False)
    try:
        return _normalize_str_path(resolved.relative_to(root_path))
    except ValueError:
        # Outside the configured root: the caller flags the still-absolute
        # return value as unfixable.
        return _normalize_str_path(resolved)
