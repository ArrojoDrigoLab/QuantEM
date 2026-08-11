"""Getting the weights onto this machine, once, with the user watching.

The checkpoint is **not bundled in the installer** (owner ruling R14). It is
375 MB for a feature many users will never open, and it comes down on demand
the first time someone asks for it.

Why this is not the model registry's downloader
-----------------------------------------------
``quantem.registry`` downloads *model packs*: multi-file, digest-manifested,
resolved through ``inference.specs.MODEL_SPECS``, installed into a
content-addressed blob store, and gated by ``pack_id in MODEL_SPECS`` at the
install endpoint. A SAM checkpoint is one ``.pt`` file from a third-party host
and fits none of that -- and forcing it in would mean adding a key to
``ARCHITECTURE``/``MODEL_SPECS``, which several modules iterate and index, so
the key would raise ``KeyError`` at import in four places before it did
anything useful.

What is kept from the registry is the *shape* of the experience: a byte-level
progress figure the UI can poll, digest verification before the file is trusted,
an atomic rename so a half-written file is never mistaken for a good one, and a
failure that arrives as one plain sentence naming what to do next.

The transfer runs on a daemon thread rather than the job queue because
registering a job type means editing ``jobs/constants.py`` and
``jobs/handlers/__init__.py``, which this round belongs to other people. One
file, one thread, one status endpoint is proportionate to a single-user
loopback desktop server -- and the swap is a small one if it is ever wanted.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from quantem.sam.config import CHECKPOINT, CheckpointSpec

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20
_DOWNLOAD_TIMEOUT_SECONDS = 120


class CheckpointMissing(RuntimeError):
    """The weights are not on disk, so nothing can be prompted yet."""


class CheckpointDownloadFailed(RuntimeError):
    """The transfer or its verification failed. Carries the user-facing sentence."""


def models_dir() -> Path:
    """``<data dir>/models/sam``, resolved from the running install.

    Never a literal path: this app runs on many machines and the data directory
    is wherever that machine put it.
    """
    from quantem.core.config import MODELS_DIR

    return Path(MODELS_DIR) / "sam"


def checkpoint_path(spec: CheckpointSpec = CHECKPOINT) -> Path:
    return models_dir() / spec.filename


def installed(spec: CheckpointSpec = CHECKPOINT) -> bool:
    """True when the file is present and the right size.

    Size, not digest: hashing 375 MB to answer "is it there" on every page load
    is not a service. The digest is checked once, at download, before the file
    is put in place.
    """
    path = checkpoint_path(spec)
    try:
        return path.is_file() and path.stat().st_size == spec.size_bytes
    except OSError:
        return False


@dataclass
class DownloadState:
    """What the status endpoint reports. One transfer at a time, process-wide."""

    status: str = "IDLE"  # IDLE | RUNNING | SUCCESS | FAILED
    bytes_done: int = 0
    bytes_total: int = 0
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "status": self.status,
                "bytes_done": self.bytes_done,
                "bytes_total": self.bytes_total,
                "error": self.error,
            }


_STATE = DownloadState()
_START_LOCK = threading.Lock()


def status(spec: CheckpointSpec = CHECKPOINT) -> dict[str, object]:
    """Everything the client needs to decide what to show."""
    snapshot = _STATE.snapshot()
    ready = installed(spec)
    percent: float | None = None
    total = int(snapshot["bytes_total"]) or spec.size_bytes
    if snapshot["status"] == "RUNNING" and total > 0:
        percent = round(100.0 * int(snapshot["bytes_done"]) / total, 1)
    return {
        "model": spec.display_name,
        "installed": ready,
        "download": {**snapshot, "percent": percent},
        "size_bytes": spec.size_bytes,
    }


def _verify(path: Path, spec: CheckpointSpec) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != spec.sha256:
        raise CheckpointDownloadFailed(
            f"The downloaded {spec.display_name} file does not match its "
            "published checksum, so it has not been installed. This is usually "
            "an interrupted transfer. Try the download again."
        )


def _download(spec: CheckpointSpec) -> None:
    destination = checkpoint_path(spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")

    try:
        with urllib.request.urlopen(spec.url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            declared = int(response.headers.get("Content-Length") or spec.size_bytes)
            with _STATE._lock:
                _STATE.bytes_total = declared
            done = 0
            with open(partial, "wb") as handle:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    with _STATE._lock:
                        _STATE.bytes_done = done

        _verify(partial, spec)
        # Rename last, so an interrupted run leaves a ``.partial`` that nothing
        # reads rather than a truncated checkpoint that loads and misbehaves.
        os.replace(partial, destination)
    except CheckpointDownloadFailed:
        partial.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise CheckpointDownloadFailed(
            f"QuantEM could not download {spec.display_name}. Check this "
            "computer's internet connection and try again."
        ) from exc


def _run(spec: CheckpointSpec) -> None:
    try:
        _download(spec)
    except CheckpointDownloadFailed as exc:
        logger.warning("SAM checkpoint download failed: %s", exc)
        with _STATE._lock:
            _STATE.status = "FAILED"
            _STATE.error = str(exc)
        return
    except Exception:  # pragma: no cover - the unforeseen still needs a sentence
        logger.exception("SAM checkpoint download failed unexpectedly")
        with _STATE._lock:
            _STATE.status = "FAILED"
            _STATE.error = (
                f"QuantEM could not install {spec.display_name}. Try the "
                "download again."
            )
        return

    from quantem.sam.backends import reset_backend

    # The next prompt should build against the file that just landed, not go on
    # reporting that there is no file.
    reset_backend()
    with _STATE._lock:
        _STATE.status = "SUCCESS"
        _STATE.error = ""
        _STATE.bytes_done = _STATE.bytes_total


def start_download(spec: CheckpointSpec = CHECKPOINT) -> dict[str, object]:
    """Begin the transfer if it is not already running or already done."""
    if installed(spec):
        return status(spec)
    with _START_LOCK:
        if _STATE.snapshot()["status"] == "RUNNING":
            return status(spec)
        with _STATE._lock:
            _STATE.status = "RUNNING"
            _STATE.bytes_done = 0
            _STATE.bytes_total = spec.size_bytes
            _STATE.error = ""
        thread = threading.Thread(
            target=_run,
            args=(spec,),
            name="quantem-sam-checkpoint",
            daemon=True,
        )
        thread.start()
    return status(spec)


def reset_state_for_tests() -> None:
    with _STATE._lock:
        _STATE.status = "IDLE"
        _STATE.bytes_done = 0
        _STATE.bytes_total = 0
        _STATE.error = ""
