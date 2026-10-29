"""Immutable display snapshots shared by raster, palette and picking requests."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from .constants import OVERLAY_STORE_DIRNAME
from .paths import OverlayStoreError, _remove_tree, get_overlay_version_dir

logger = logging.getLogger(__name__)
SNAPSHOT_FILENAME = "snapshot.json"
LABEL_MAP_FILENAME = "labels.json"
# Protect the newest three and active readers; other generations share count,
# byte and age limits. Bytes are logical payload, not filesystem allocation.
RETAIN_SNAPSHOTS = 3
RETAIN_SECONDS = 3600
MAX_UNLEASED_SNAPSHOTS = 8
MAX_RETAINED_LOGICAL_BYTES = 512 * 1024 * 1024
READER_GRACE_SECONDS = 60
_PRUNE_SLOT = threading.Lock()


def _reader_lease(directory: Path) -> None:
    from .cache import reader_lease_cache

    if reader_lease_cache.get(str(directory)) is not None:
        return
    try:
        (directory / ".reader").touch()
        reader_lease_cache.put(str(directory), True)
    except OSError:
        # A read-only store remains readable; the newest generations still
        # have retention protection and expired URLs fail explicitly.
        pass


def schedule_snapshot_prune(segmentation_id, source_model, active_version):
    """Reclaim idle derived generations without putting file deletion in polls."""
    from .cache import snapshot_prune_cache

    key = (str(segmentation_id), source_model)
    if active_version <= RETAIN_SNAPSHOTS or snapshot_prune_cache.get(key) is not None:
        return
    if not _PRUNE_SLOT.acquire(blocking=False):
        return
    snapshot_prune_cache.put(key, True)

    def prune():
        try:
            prune_snapshots(str(segmentation_id), source_model, active_version)
        finally:
            _PRUNE_SLOT.release()

    try:
        threading.Thread(target=prune, name="overlay-snapshot-cleanup", daemon=True).start()
    except RuntimeError:
        _PRUNE_SLOT.release()
        logger.warning("Could not start overlay snapshot cleanup", exc_info=True)


def clone_store(
    source: Path, destination: Path, *, replaced_chunks: set[str] | None = None
) -> None:
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

    from quantem.core.machine import get_machine_profile

    workers = max(1, min(8, get_machine_profile().raster_workers))
    replaced_chunks = replaced_chunks or set()

    def link_batch(batch):
        for src, dst in batch:
            link_or_copy(src, dst)

    # File-entry work dominates sparse stores on Windows. Overlap a bounded
    # number of metadata operations, without submitting every file or copying
    # complete arrays into memory. Readers still see only the final generation.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="overlay-clone") as pool:
        pending = deque()
        batch = []
        for root, _directories, files in os.walk(source):
            folder = destination / Path(root).relative_to(source)
            folder.mkdir(parents=True, exist_ok=True)
            for name in files:
                if (Path(root).relative_to(source) / name).as_posix() in replaced_chunks:
                    # Repainting writes the complete chunk. Cloning and then
                    # detaching its previous bytes is pure write amplification,
                    # especially on filesystems where hard links fall back to copies.
                    continue
                batch.append((Path(root) / name, folder / name))
                if len(batch) == 64:
                    pending.append(pool.submit(link_batch, batch))
                    batch = []
                    if len(pending) >= workers * 2:
                        pending.popleft().result()
        if batch:
            pending.append(pool.submit(link_batch, batch))
        for future in pending:
            future.result()


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
    logical_bytes = sum(path.stat().st_size for path in store.rglob("*") if path.is_file())
    (store.parent / SNAPSHOT_FILENAME).write_text(
        json.dumps(
            {
                "bundle_version": bundle_version,
                "applied_revision": applied_revision,
                "geometry_revision": geometry_revision,
                "logical_bytes": logical_bytes + len(assignments) * 48,
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
    segmentation_id: str,
    source_model: str | None,
    revision: str,
    *,
    include_labels=True,
    lease=True,
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
        if lease:
            _reader_lease(directory)
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
        now = time.time()
        retained_bytes = 0
        for index, path in enumerate(versions):
            try:
                payload = json.loads((path / SNAPSHOT_FILENAME).read_text(encoding="utf-8"))
                size = int(payload.get("logical_bytes", 0))
            except (OSError, ValueError, TypeError, AttributeError):
                size = 0
            try:
                leased = (path / ".reader").stat().st_mtime >= now - READER_GRACE_SECONDS
            except OSError:
                leased = False
            over_budget = retained_bytes + size > MAX_RETAINED_LOGICAL_BYTES
            expired = path.stat().st_mtime < now - RETAIN_SECONDS
            if (
                index >= RETAIN_SNAPSHOTS
                and not leased
                and int(path.name) < active_version
                and (index >= MAX_UNLEASED_SNAPSHOTS or over_budget or expired)
            ):
                _remove_tree(path)
            else:
                retained_bytes += size
    except (OSError, OverlayStoreError):
        # Cleanup must not turn a successfully published overlay into a failure.
        logger.warning("Could not prune old overlay snapshots in %s", root, exc_info=True)
