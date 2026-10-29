"""Publish complete files under fresh names before switching database pointers."""

import os
import time
from pathlib import Path
from uuid import uuid4


def publish_directory(source: Path, target: Path):
    """Rename a complete directory, tolerating brief Windows sharing locks."""
    for attempt in range(5):
        try:
            source.rename(target)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 4:
                raise
            time.sleep(0.05 * (2**attempt))


def write_immutable_file(directory: Path, *, prefix: str, suffix: str, write) -> Path:
    """A rollback may orphan a new file, but can never damage a referenced one.

    ``write`` must close its handles and validate the payload before returning.
    Staging and publication use the same filesystem. Readers never see the
    temporary name, and a repeated operation always gets a different final name.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{prefix}-{uuid4().hex}{suffix}"
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        write(temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)
