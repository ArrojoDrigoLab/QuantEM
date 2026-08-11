"""The QuantEM Hugging Face model repository: what is published, fetch one file, verify it.

The eight released packs live in one public repository,
``ArrojoeDrigoLab/quantem``, laid out as:

* ``<family>-<organelle>.safetensors`` -- the pack's head (neck + decoder +
  whatever encoder tensors the pack owns), plus ``<family>-<organelle>.json``,
  a model card carrying the architecture, the inference contract, and the
  artifact's size and sha256;
* ``quantem-vitb-trunk.safetensors`` / ``omniem-vitl.safetensors`` -- the two
  shared encoder trunks, named by each card's ``requires`` list.

Two different digest sources, on purpose
----------------------------------------
The **head** is verified against the sha256 its own model card publishes: the
card is the artifact's contract and travels with it. The **trunks** have no
card, so their expected digest is taken from the repository's file metadata at
the pinned revision -- for an LFS file the object id *is* its sha256. Both are
read at the same pinned revision, so a later force-push cannot silently change
what an already-released app verifies against.

Revision pinning
----------------
:data:`QUANTEM_HF_REVISION` is the commit this build resolves against,
overridable with ``$QUANTEM_HF_REVISION``. Pinning is what makes a future
release additive: digests recorded here keep verifying even if ``main`` moves.

Nothing in this module imports torch or Django. Downloads go through
``huggingface_hub`` (the only fetcher in this ecosystem that resumes a dropped
1.2 GB download), always with an explicit ``cache_dir`` under the app data
directory -- never HF's default home, which on a machine that has never used HF
would silently grow a cache in the user profile.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantem.registry import cache

logger = logging.getLogger(__name__)

#: The public repository all eight packs are published in.
HF_REPO_ID = "ArrojoeDrigoLab/quantem"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_ID}"

#: The commit this build downloads and verifies against. Recorded 2026-08-08.
QUANTEM_HF_REVISION = "f76dd9743a51a095191d27fc18724152dfd30c5a"

#: Environment override for the pinned revision (a tag, branch or commit sha).
HF_REVISION_ENV_VAR = "QUANTEM_HF_REVISION"

#: Called with ``(bytes_done, bytes_total)`` while a file downloads.
BytesProgress = Callable[[int, int], None]

#: How often the download watcher samples progress, in seconds.
_POLL_SECONDS = 0.5


class HfError(RuntimeError):
    """Something about the published repository is wrong (bad card, bad digest)."""


class HfUnavailableError(HfError):
    """The repository could not be reached. The message names the offline path."""


def hf_revision() -> str:
    """The revision to resolve against: ``$QUANTEM_HF_REVISION``, else the pinned sha."""
    return os.environ.get(HF_REVISION_ENV_VAR, "").strip() or QUANTEM_HF_REVISION


def hf_cache_dir() -> Path:
    """``<QUANTEM_DATA_DIR>/cache/hf`` -- the explicit huggingface_hub cache.

    Resolved through :mod:`quantem.core.config` when importable (one process,
    one answer), with the same environment fallback as
    :func:`quantem.registry.cache.models_root` so a plain script still works.
    """
    try:
        from quantem.core.config import CACHE_DIR

        return Path(CACHE_DIR) / "hf"
    except Exception:  # pragma: no cover - only when core.config is unavailable
        raw = os.environ.get("QUANTEM_DATA_DIR", "").strip()
        if not raw:
            raise HfError(
                "QuantEM cannot tell where its data folder is, so it has "
                "nowhere to download a model to."
            ) from None
        return Path(raw) / "cache" / "hf"


def artifact_basename(pack_id: str) -> str:
    """``"quantem:mito"`` -> ``"quantem-mito"`` (the published artifact naming)."""
    family, organelle = pack_id.split(":", 1)
    return f"{family}-{organelle}"


def sidecar_filename(pack_id: str) -> str:
    return f"{artifact_basename(pack_id)}.json"


def head_filename(pack_id: str) -> str:
    return f"{artifact_basename(pack_id)}.safetensors"


@dataclass(frozen=True)
class Sidecar:
    """One pack's published model card, parsed from ``<family>-<organelle>.json``."""

    model_id: str
    family: str
    organelle: str
    architecture: dict[str, Any]
    inference: dict[str, Any]
    #: ``{"filename", "bytes", "sha256"}`` of the head safetensors.
    artifact: dict[str, Any]
    #: Artifact basenames this pack needs, trunk first when one is shared.
    requires: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def head_sha256(self) -> str:
        return str(self.artifact.get("sha256") or "")

    @property
    def head_bytes(self) -> int:
        return int(self.artifact.get("bytes") or 0)

    @property
    def head_file(self) -> str:
        return str(self.artifact.get("filename") or "")

    @property
    def trunk_basename(self) -> str | None:
        """The shared trunk this head sits on, or None for a self-contained pack."""
        own = self.head_file.removesuffix(".safetensors")
        for name in self.requires:
            if name != own:
                return name
        return None


@dataclass(frozen=True)
class RemoteFile:
    """Size and digest of one repository file at the pinned revision."""

    filename: str
    size_bytes: int
    sha256: str


def _prepare_hub_env() -> None:
    """Keep huggingface_hub quiet and inside the app data directory.

    ``setdefault`` throughout: an environment that already configured these wins.
    ``HF_XET_CACHE`` matters -- with the ``hf_xet`` backend installed, chunk
    caches would otherwise land under HF's default home even when ``cache_dir``
    is explicit.

    ``HF_XET_LOG_DEST`` matters just as much: hf-xet's telemetry log ignores
    both ``HF_XET_CACHE`` and ``HF_HOME`` and, with no explicit destination,
    writes rolling log files under ``~/.cache/huggingface/xet/logs`` on every
    download (xet-core ``xet_runtime/src/logging``: an unset ``dest`` falls
    back to the hard-derived default cache home). Routing the destination is
    chosen over ``HF_HUB_DISABLE_XET=1`` because it keeps the xet fast path --
    chunked, resumable, deduplicated downloads of the 1.2 GB trunks -- while
    disabling xet would cost exactly that and *still* leave the logger
    configured for the next process that uses xet. The value must name a
    directory (hf-xet treats a path ending in a separator, or an existing
    directory, as "directory mode" and keeps its rolling files + 250 MB
    cleanup behaviour there), so it is created first and passed with a
    trailing separator.
    """
    cache_dir = hf_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    xet_log_dir = Path(os.environ["HF_XET_CACHE"]) / "logs"
    try:
        xet_log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:  # an unwritable override is hf-xet's problem, not a crash
        logger.debug("Could not create the xet log directory %s", xet_log_dir)
    os.environ.setdefault("HF_XET_LOG_DEST", str(xet_log_dir) + os.sep)


def _offline_error(what: str, exc: Exception) -> HfUnavailableError:
    """An honest no-network error: name the repo, then the road that still works."""
    # App copy: this lands on a failed install job and is shown verbatim on the
    # Models screen, so it names a screen and a button, never a command (I-12).
    return HfUnavailableError(
        f"Could not reach the QuantEM model repository ({HF_REPO_URL}) to fetch {what}. "
        f"If this machine has no internet access, use the offline route instead: "
        f"download a QuantEM model release on a machine that does, copy it here and "
        f'unzip it, then use "Install from a local folder" on the Models screen. '
        # The class name used to lead this clause ("OSError: no route to
        # host"). It is a Python type, it means nothing to the reader, and
        # I-12 forbids it; the message it prefixed is the part that carries
        # information.
        f"What went wrong: {exc}"
    )


def _looks_offline(exc: Exception) -> bool:
    """Best-effort classification: a network failure, not a bad repository.

    huggingface_hub raises different exception types across versions and
    backends (requests, httpx, xet), so this goes by family name rather than
    importing each one.
    """
    names = {type(e).__name__ for e in _walk_causes(exc)}
    offline_names = {
        "OfflineModeIsEnabled",
        "LocalEntryNotFoundError",
        "ConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "TimeoutError",
        "NameResolutionError",
        "MaxRetryError",
        "NewConnectionError",
        "SSLError",
        "ProxyError",
        "gaierror",
    }
    return bool(names & offline_names)


def _walk_causes(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def fetch_sidecar(pack_id: str, *, revision: str | None = None) -> Sidecar:
    """Download and parse a pack's model card (a few hundred bytes).

    Raises:
        HfUnavailableError: no network.
        HfError: the card is unreadable or carries no digest -- an unverifiable
            artifact is refused before a single weight byte is fetched.
    """
    path = download_file(sidecar_filename(pack_id), revision=revision)
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HfError(f"{sidecar_filename(pack_id)} is not readable JSON: {exc}") from exc

    card = Sidecar(
        model_id=str(raw.get("model_id", "")),
        family=str(raw.get("family", "")),
        organelle=str(raw.get("organelle", "")),
        architecture=dict(raw.get("architecture") or {}),
        inference=dict(raw.get("inference") or {}),
        artifact=dict(raw.get("artifact") or {}),
        requires=[str(r) for r in (raw.get("requires") or [])],
        raw=raw,
    )
    if not card.head_file:
        raise HfError(
            f"{sidecar_filename(pack_id)} names no artifact filename; the published "
            "card is incomplete."
        )
    if not card.head_sha256:
        raise HfError(
            f"{sidecar_filename(pack_id)} carries no sha256 for {card.head_file}. "
            "Refusing to download an unverifiable artifact."
        )
    return card


def remote_file_info(filename: str, *, revision: str | None = None) -> RemoteFile:
    """Size and sha256 of one repository file, from HF file metadata.

    For an LFS file the object id **is** the file's sha256, which is what makes
    the shared trunks verifiable without a card of their own. Refuses a
    non-LFS file: those have no content digest to pin an install to.
    """
    _prepare_hub_env()
    from huggingface_hub import HfApi

    rev = revision or hf_revision()
    try:
        paths = HfApi().get_paths_info(HF_REPO_ID, [filename], revision=rev)
    except Exception as exc:
        if _looks_offline(exc):
            raise _offline_error(f"the metadata of {filename}", exc) from exc
        raise HfError(
            f"Could not read the metadata of {filename} in {HF_REPO_ID}@{rev}: {exc}"
        ) from exc

    entry = next((p for p in paths if getattr(p, "path", "") == filename), None)
    if entry is None:
        raise HfError(f"{HF_REPO_ID}@{rev} has no file named {filename}.")
    lfs = getattr(entry, "lfs", None)
    sha256 = ""
    if lfs is not None:
        sha256 = str(getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else "") or "")
    if not sha256:
        raise HfError(
            f"{filename} in {HF_REPO_ID}@{rev} is not stored as an LFS object, so the "
            "repository publishes no sha256 for it. Refusing an unverifiable download."
        )
    size = int(getattr(entry, "size", 0) or 0)
    return RemoteFile(filename=filename, size_bytes=size, sha256=sha256)


def download_file(
    filename: str,
    *,
    revision: str | None = None,
    expected_bytes: int | None = None,
    on_bytes: BytesProgress | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Path:
    """Fetch one repository file into the app's HF cache; return its local path.

    The download itself is ``hf_hub_download`` -- resumable, atomically staged
    by the hub library inside :func:`hf_cache_dir`. Progress is reported by
    watching the hub cache's in-flight files, because the library exposes no
    byte callback: coarse, but honest, and it costs the download nothing.

    ``cancel_check`` is called between progress samples and may raise to abort
    waiting. The underlying transfer cannot be interrupted mid-file; it is left
    to finish (or die with the process) on a daemon thread, and whatever it
    completes lands only in the content-addressed hub cache -- never as an
    installed pack.
    """
    _prepare_hub_env()
    from huggingface_hub import hf_hub_download

    rev = revision or hf_revision()
    cache_dir = hf_cache_dir()

    result: dict[str, Any] = {}
    done = threading.Event()

    def _work() -> None:
        try:
            result["path"] = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                revision=rev,
                cache_dir=str(cache_dir),
            )
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller's thread
            result["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(
        target=_work, name=f"hf-download-{filename}", daemon=True
    )
    worker.start()

    last_reported = -1
    while not done.wait(timeout=_POLL_SECONDS):
        if cancel_check is not None:
            cancel_check()
        if on_bytes is not None and expected_bytes:
            got = _in_flight_bytes(cache_dir)
            if got != last_reported:
                last_reported = got
                on_bytes(min(got, expected_bytes), expected_bytes)

    error = result.get("error")
    if error is not None:
        if not isinstance(error, Exception):  # KeyboardInterrupt and kin
            raise error
        if _looks_offline(error):
            raise _offline_error(filename, error) from error
        raise HfError(
            f"Downloading {filename} from the QuantEM model repository failed: {error}"
        ) from error

    path = Path(str(result["path"]))
    if on_bytes is not None and expected_bytes:
        on_bytes(expected_bytes, expected_bytes)
    return path


def _in_flight_bytes(cache_dir: Path) -> int:
    """Bytes of partially-downloaded files currently in the hub cache.

    ``hf_hub_download`` stages every transfer as ``*.incomplete`` beside the
    final blob. Summing them is what makes progress reportable without a
    callback API. With two concurrent downloads this over-reports each one --
    a limitation, not a lie, and the terminal 100% still comes from the real
    completion.
    """
    total = 0
    try:
        for p in cache_dir.rglob("*.incomplete"):
            try:
                total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def verify_sha256(path: Path, expected: str, *, what: str, source: str) -> str:
    """Re-hash ``path`` and compare. A mismatch names both digests and raises."""
    actual = cache.sha256_file(path)
    if actual.lower() != expected.lower():
        raise HfError(
            f"{what} failed verification: expected sha256 {expected} ({source}), "
            f"got {actual}. Nothing was installed. Delete the cached file and retry; "
            f"if it keeps happening, the published artifact and its digest disagree "
            f"and the repository maintainers need to know."
        )
    return actual


def safetensors_metadata(path: Path) -> dict[str, str]:
    """The free-form metadata block of a safetensors file (never the tensors)."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            return dict(f.metadata() or {})
    except Exception:  # metadata is provenance, not correctness
        logger.debug("No readable safetensors metadata in %s", path, exc_info=True)
        return {}


#: Bytes a fresh HF install of a pack must fetch, without touching the network.
def planned_download(pack_id: str) -> dict[str, Any]:
    """Static plan from :mod:`quantem.registry.manifest` sizes. Never network."""
    from quantem.inference.specs import MODEL_SPECS
    from quantem.registry.catalogue import download_bytes

    spec = MODEL_SPECS[pack_id]
    return {
        "repo_id": HF_REPO_ID,
        "revision": hf_revision(),
        "url": HF_REPO_URL,
        "download_bytes": download_bytes(spec),
    }
