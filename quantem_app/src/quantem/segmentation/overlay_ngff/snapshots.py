"""Immutable display snapshots shared by raster, palette and picking requests."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

from .constants import OVERLAY_STORE_DIRNAME
from .paths import OverlayStoreError, _remove_tree, get_overlay_version_dir

logger = logging.getLogger(__name__)
SNAPSHOT_FILENAME = "snapshot.json"
LABEL_MAP_FILENAME = "labels.json"
# Keep recent readers alive across rapid edits. Older snapshots remain for an
# hour; the newest three are retained even after a quiet session.
RETAIN_SNAPSHOTS = 3
RETAIN_SECONDS = 3600


def clone_store(source: Path, destination: Path) -> None:
    """Share unchanged chunks; detach each changed chunk before writing it.

    Filesystems without hard links fall back to copies. Explicit detachment
    keeps snapshot immutability independent of Zarr's LocalStore internals.
    """
    if destination.parent.exists():
        _remove_tree(destination.parent)

    def link_or_copy(src, dst):
        try:
            return os.link(src, dst)
        except OSError:
            return shutil.copy2(src, dst)

    shutil.copytree(source, destination, copy_function=link_or_copy)


def detach_chunk(store: Path, array_key: str, level: int, chunk_x: int, chunk_y: int) -> None:
    """Make a staged chunk private before any read-modify-write operation."""
    path = store / array_key / str(level) / f"{chunk_y}.{chunk_x}"
    if not path.exists() or path.stat().st_nlink <= 1:
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.copy")
    try:
        shutil.copy2(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_snapshot(
    store: Path,
    *,
    bundle_version: int,
    applied_revision: int,
    assignments,
    max_label=None,
    geometry_revision=None,
) -> None:
    (store.parent / SNAPSHOT_FILENAME).write_text(
        json.dumps(
            {
                "bundle_version": bundle_version,
                "applied_revision": applied_revision,
                "geometry_revision": geometry_revision,
                "max_label": max_label
                if max_label is not None
                else max((label for label, _ in assignments), default=0),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (store.parent / LABEL_MAP_FILENAME).write_text(
        json.dumps(
            [(int(label), str(object_id)) for label, object_id in assignments],
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def read_snapshot(
    segmentation_id: str, source_model: str | None, revision: str, *, include_labels=True
):
    """Resolve an exact version-revision pair, never silently substitute latest."""
    try:
        version, applied = (int(part) for part in revision.split("-"))
        if version <= 0 or applied < 0:
            raise ValueError
        directory = get_overlay_version_dir(segmentation_id, version, source_model)
        payload = json.loads((directory / SNAPSHOT_FILENAME).read_text(encoding="utf-8"))
        if payload["bundle_version"] != version or payload["applied_revision"] != applied:
            raise ValueError
        # Tile requests read only the tiny identity record, never an O(N) map.
        if include_labels:
            payload["labels"] = json.loads(
                (directory / LABEL_MAP_FILENAME).read_text(encoding="utf-8")
            )
        return directory / OVERLAY_STORE_DIRNAME, payload
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise OverlayStoreError("The requested overlay snapshot is no longer available.") from exc


def prune_snapshots(segmentation_id: str, source_model: str | None, active_version: int) -> None:
    """Bound disk retention without deleting the active snapshot or fresh readers."""
    root = get_overlay_version_dir(segmentation_id, active_version, source_model).parent
    try:
        versions = sorted(
            (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
            reverse=True,
        )
        cutoff = time.time() - RETAIN_SECONDS
        for path in versions[RETAIN_SNAPSHOTS:]:
            if int(path.name) < active_version and path.stat().st_mtime < cutoff:
                _remove_tree(path)
    except (OSError, OverlayStoreError):
        # Cleanup must not turn a successfully published overlay into a failure.
        logger.warning("Could not prune old overlay snapshots in %s", root, exc_info=True)
