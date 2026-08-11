"""The single authority on whether an asset's pyramid may be read or served.

Three rounds of guards were each falsified by an entry point nobody had listed.
The reason is in the shape of the old code, not in the guards: readiness was a
*derived opinion*, recomputed from the filesystem at fifteen call sites from
twelve different predicates; publication *mutated the path readers were using*;
missing data was *valid data* to zarr; "this attempt failed" was written down
nowhere durable; and nothing fenced a job to the attempt it belonged to.

This module replaces all five.

**One definition.** An asset's pyramid may be read or served exactly when its
state row names a *published generation* and the reader resolves the store
through that name. Nothing outside this module may compute a second definition:
:func:`resolve_pyramid` is the only way in, it returns a
:class:`PublishedPyramid` or an :class:`Unavailable` carrying a *reason*, and
readers never build a path or call ``zarr.open_array`` themselves.

**Immutable generations.** A build writes ``<asset>.zarr/gen-<hex>/`` and is
published by one database ``UPDATE``. Nothing under a live reader is ever
renamed, moved or overwritten, so the two-rename window that handed readers
silent zeros does not exist -- there are no renames. Withdrawal is a single
column write.

**A fence, not a guard.** Publication is a compare-and-swap on the attempt
token *and* on the generation the build started from::

    UPDATE ... SET published_generation = :new
     WHERE asset = :asset
       AND attempt_token = :token_the_build_started_with
       AND published_generation = :generation_it_started_from
       AND outcome IN ('PENDING', 'SUCCEEDED')

``rowcount == 0`` means the build is stale: another attempt began, or a
terminal outcome was recorded, or another build published first. The builder
discards its generation and reports ``superseded``. **No guard has to fire at
the right instant**, including in the between-attempt ``ENCODING`` stretch
where a stage-based guard is blind, because the token is bumped on every
attempt boundary rather than only on terminal ones.

**Where the state lives, and where it is going.** The design this implements
specifies a new 1:1 table, ``AssetImportState``. ``assets/models.py`` and
``assets/migrations/`` belong to another workflow for the duration of this
change, so the same fields live in the ``metadata["pyramid"]`` object of the
asset's ``NGFF`` :class:`~quantem.assets.models.Rendition` row -- which no
other code writes, which is 1:1 with the asset, and which gives a real
single-statement compare-and-swap through a JSON key lookup in the ``WHERE``
clause. Every read and write of it is in :func:`_load_state` and
:func:`_write_state` below, so promoting it to its own table later is a change
to two functions and a migration, not to any caller.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import zarr
from django.db import transaction
from zarr.storage import LocalStore

from quantem.core.config import NGFF_TMP_DIR

logger = logging.getLogger(__name__)

#: How long a superseded generation is kept after its successor is published,
#: so a reader that resolved the old pointer can finish. The only clock in the
#: sweep contract, and it protects readers rather than guessing at liveness.
NGFF_DRAIN_SECONDS = float(os.environ.get("QUANTEM_NGFF_DRAIN_SECONDS", "120"))

#: An orphan with no readable ``owner.json`` is debris from a process that died
#: between ``mkdir`` and the first write. Young ones are left alone only long
#: enough for the owning process to write the file.
_UNOWNED_GRACE_SECONDS = 5.0

GENERATION_PREFIX = "gen-"

OUTCOME_PENDING = "PENDING"
OUTCOME_SUCCEEDED = "SUCCEEDED"
OUTCOME_FAILED = "FAILED"
OUTCOME_CANCELLED = "CANCELLED"

#: Outcomes a publish may land on. A terminal outcome fences every build that
#: was already running.
_PUBLISHABLE_OUTCOMES = (OUTCOME_PENDING, OUTCOME_SUCCEEDED)

#: Preprocessing stages that mean "the import is over and produced nothing".
#: Kept as a *second* terminal signal beside ``outcome`` because the job layer
#: writes ``preprocess_stage`` from its own failure reconciler, which is
#: outside this module.
TERMINAL_STAGES = {"FAILED": OUTCOME_FAILED, "CANCELLED": OUTCOME_CANCELLED}

#: Nothing is published. A sentinel string rather than ``None``: Django's JSON
#: key lookups treat ``None`` as SQL NULL *and* as JSON null depending on the
#: backend, and the compare-and-swap must compare a value, never a null.
NOT_PUBLISHED = ""


class PyramidChunkMissing(FileNotFoundError):
    """A chunk the store promised is not on disk.

    zarr substitutes ``fill_value`` for an absent chunk and raises nothing, so
    a store that vanishes mid-read hands the caller a correctly shaped plane of
    zeros. Measured: 15 all-zero windows in 3 874 reads across 40 publishes,
    and partially-zeroed windows where some chunks resolved and some did not.
    This exception is what those reads raise instead.
    """


class Reason(StrEnum):
    NEVER_BUILT = "never_built"
    BUILDING = "building"
    TERMINAL_FAILURE = "terminal_failure"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    GEOMETRY_MISMATCH = "geometry_mismatch"
    STALE_DECODER = "stale_decoder"
    NO_ASSET = "no_asset"


class Intent(StrEnum):
    SERVE = "serve"
    READ = "read"
    BUILD = "build"


@dataclass(frozen=True)
class Unavailable:
    """Why there is no pyramid to read. A reason, never a bool."""

    reason: Reason
    detail: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - guarded against by design
        raise TypeError(
            "Unavailable is not a boolean. Readiness is a reason, not a flag: "
            "match on .reason so the caller states what it does about each case."
        )


class QuantemChunkStore(LocalStore):
    """A ``LocalStore`` that refuses to answer "absent" for a chunk key.

    Metadata keys pass straight through -- an early version of this raised on
    ``zarr.json`` and broke ``open_array`` outright. MEASURED cost against the
    plain ``LocalStore`` over 36 x 512^2 ROI reads: -0.3 %, i.e. free.
    """

    @staticmethod
    def _is_chunk_key(key: str) -> bool:
        leaf = str(key).rsplit("/", 1)[-1]
        return bool(leaf) and all(part.isdigit() for part in leaf.split(".") if part != "")

    async def get(self, key, prototype, byte_range=None):  # type: ignore[override]
        value = await super().get(key, prototype, byte_range)
        if value is None and self._is_chunk_key(key):
            raise PyramidChunkMissing(
                f"NGFF chunk vanished while it was being read: {self.root}/{key}"
            )
        return value


@lru_cache(maxsize=64)
def _open_generation_level(level_root: str):
    """A strict-store zarr array for one level of one generation.

    Cached on the path alone, with no mtime in the key: a published generation
    directory is immutable, so the array it describes cannot change under the
    cache. If the generation is swept while a reader still holds this array,
    the next chunk read raises :class:`PyramidChunkMissing` rather than
    returning zeros.
    """

    return zarr.open_array(store=QuantemChunkStore(root=level_root), mode="r")


@dataclass(frozen=True)
class PublishedPyramid:
    """A generation that is published right now, and how to read it."""

    asset_id: str
    generation_id: str
    root: Path
    manifest: dict

    def open_level(self, level: int):
        return _open_generation_level(str(self.root / str(int(level))))

    @property
    def level_count(self) -> int:
        return len(self.manifest.get("levels") or [])

    def level_shape(self, level: int) -> tuple[int, ...]:
        levels = self.manifest.get("levels") or []
        return tuple(int(value) for value in levels[int(level)]["shape"])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        shape = self.level_shape(0)
        return int(shape[-2]), int(shape[-1])

    def file_path(self, relative: str) -> Path:
        """An arbitrary path inside this generation, for the HTTP route."""

        return self.root / str(relative).lstrip("/")


@dataclass(frozen=True)
class BuildTicket:
    """Permission to build one generation, and the fence it will publish under.

    A build with no ticket cannot happen: :func:`quantem.assets.ngff.build_pyramid`
    takes one of these, not a path. That is what leaves the ``.png``-suffix
    test that round 3 rebuilt a staged upload through with nowhere to live.
    """

    asset_id: str
    attempt_token: str
    generation_id: str
    root: Path
    from_generation: str
    decoder_version: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Where the state lives
# ---------------------------------------------------------------------------

_EMPTY_STATE = {
    "attempt_token": "",
    "outcome": OUTCOME_PENDING,
    "failure_detail": "",
    "published_generation": NOT_PUBLISHED,
    "published_manifest": {},
    "published_at": "",
}


def _asset_of(target):
    """The ``Asset`` behind an ``Asset`` or an ``AssetOpenable``."""

    asset = getattr(target, "asset", None)
    return asset if asset is not None else target


def _state_rendition(asset):
    """The NGFF rendition row carrying this asset's pyramid state, or ``None``.

    Reads the prefetched renditions when the caller supplied them, so the
    library list still costs one query for sixty assets.
    """

    from .models import Rendition

    cache = getattr(asset, "_prefetched_objects_cache", {})
    prefetched = cache.get("renditions")
    if prefetched is not None:
        for rendition in prefetched:
            if rendition.type == Rendition.TYPE_NGFF:
                return rendition
        return None
    return Rendition.objects.filter(asset=asset, type=Rendition.TYPE_NGFF).first()


def _load_state(asset) -> dict | None:
    rendition = _state_rendition(asset)
    if rendition is None:
        return None
    metadata = rendition.metadata if isinstance(rendition.metadata, dict) else {}
    stored = metadata.get("pyramid")
    if not isinstance(stored, dict):
        return None
    state = dict(_EMPTY_STATE)
    state.update({key: stored.get(key, value) for key, value in _EMPTY_STATE.items()})
    return state


def _ensure_state(asset) -> dict:
    """Create the state row if this asset has never had one, and return the state."""

    from .models import Rendition

    state = _load_state(asset)
    if state is not None:
        return state
    state = dict(_EMPTY_STATE)
    state["attempt_token"] = str(uuid.uuid4())
    state["outcome"] = TERMINAL_STAGES.get(asset.preprocess_stage) or (
        OUTCOME_SUCCEEDED if asset.preprocess_stage == "DONE" else OUTCOME_PENDING
    )
    Rendition.objects.update_or_create(
        asset=asset,
        type=Rendition.TYPE_NGFF,
        defaults={
            "storage_root": "NGFF_TMP_DIR",
            "stored_path": "",
            "path_exists": False,
            "is_directory": False,
            "stored_channels": 1,
            "stored_bit_depth": 8,
            "metadata": {"display_name": asset.display_name, "pyramid": state},
        },
    )
    _invalidate_prefetch(asset)
    return state


def _invalidate_prefetch(asset) -> None:
    cache = getattr(asset, "_prefetched_objects_cache", None)
    if isinstance(cache, dict):
        cache.pop("renditions", None)


def _write_state(asset, state: dict, *, published_root: Path | None) -> None:
    """Unconditional write of the state row (not the compare-and-swap)."""

    from django.utils import timezone

    from .models import Rendition

    relative = ""
    if published_root is not None:
        relative = published_root.relative_to(NGFF_TMP_DIR).as_posix()
    Rendition.objects.update_or_create(
        asset=asset,
        type=Rendition.TYPE_NGFF,
        defaults={
            "storage_root": "NGFF_TMP_DIR",
            "stored_path": relative,
            "path_exists": bool(published_root is not None and published_root.is_dir()),
            "is_directory": published_root is not None,
            "stored_width": asset.logical_width,
            "stored_height": asset.logical_height,
            "stored_channels": 1,
            "stored_bit_depth": 8,
            "metadata": {
                "display_name": asset.display_name,
                "pyramid": state,
                "written_at": timezone.now().isoformat(),
            },
        },
    )
    _invalidate_prefetch(asset)


# ---------------------------------------------------------------------------
# Generation directories on disk
# ---------------------------------------------------------------------------


def asset_generation_dir(asset_id) -> Path:
    return NGFF_TMP_DIR / f"{asset_id}.zarr"


def _new_generation_id() -> str:
    return f"{GENERATION_PREFIX}{uuid.uuid4().hex[:12]}"


@lru_cache(maxsize=1)
def boot_id() -> str:
    """An identifier that changes when the machine reboots.

    The sweep's second rule -- "a generation from a different boot is debris,
    with no age threshold at all" -- is what makes "a kill at any instant
    leaves nothing permanent" true, and it is the rule that does not depend on
    a later build of the same image ever happening.
    """

    try:
        import psutil  # noqa: PLC0415 - optional accelerator, imported once

        return f"boot-{int(psutil.boot_time())}"
    except Exception:  # noqa: BLE001 - psutil is optional
        return f"boot-{int((time.time() - time.monotonic()) // 60)}"


def _pid_is_this_app(pid: int) -> bool:
    """True when ``pid`` is alive *and* is a QuantEM process.

    Never a name-based sweep in the other direction: this only ever decides
    whether to leave a directory alone.
    """

    try:
        import psutil  # noqa: PLC0415

        process = psutil.Process(int(pid))
        name = (process.name() or "").lower()
        if "python" in name or "quantem" in name:
            return True
        return "quantem" in " ".join(process.cmdline()).lower()
    except Exception:  # noqa: BLE001
        pass
    if pid == os.getpid():
        return True
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    except Exception:  # noqa: BLE001
        return False
    return True


#: Open handles on the per-generation lock files this process owns, so they stay
#: held for the life of the build and are released by the OS if it is killed.
_OWNER_LOCKS: dict[str, object] = {}

OWNER_LOCK_NAME = "owner.lock"


def _take_owner_lock(root: Path) -> None:
    """Hold a file open inside this generation for as long as we are building it.

    **A pid is not an identity on Windows.** Measured while writing the kill
    harness: after ten kills of a builder, two interrupted generations survived
    the sweep, because Windows had already recycled the dead child's pid onto a
    live python process and the "is that pid alive and is it this app?" test
    said yes. That is not a hypothetical -- it happened on the first run, and it
    would have made the sweep silently useless in exactly the case it exists
    for.

    A held file handle *is* an identity. Windows refuses to delete a file
    another process has open, and releases the handle when that process dies,
    however it dies. So the sweeper's liveness test is "can I delete this
    generation's lock?" -- and if it can, the owner is gone, whatever the pid
    table says. ``pid`` stays in ``owner.json`` for diagnosis and for the
    fallback below.
    """

    if str(root) in _OWNER_LOCKS:
        return
    try:
        handle = (root / OWNER_LOCK_NAME).open("wb")  # noqa: SIM115 - held on purpose
        handle.write(str(os.getpid()).encode("ascii"))
        handle.flush()
    except OSError:
        return
    _OWNER_LOCKS[str(root)] = handle


def release_owner_lock(root: Path) -> None:
    """Let go of a generation we have finished with (sealed or discarded)."""

    handle = _OWNER_LOCKS.pop(str(root), None)
    if handle is not None:
        with contextlib.suppress(OSError):
            handle.close()
    with contextlib.suppress(OSError):
        (root / OWNER_LOCK_NAME).unlink(missing_ok=True)


def _owner_is_gone(root: Path, owner: dict) -> bool:
    """True when nothing is still building this generation.

    The lock file is authoritative when it is there. Only a generation written
    before this mechanism existed falls back to the pid, which is why the
    fallback can stay simple.
    """

    lock = root / OWNER_LOCK_NAME
    if lock.exists():
        try:
            lock.unlink()
        except OSError:
            return False
        return True
    pid = owner.get("pid")
    return not isinstance(pid, int) or not _pid_is_this_app(pid)


def _write_owner(root: Path, ticket_fields: dict) -> None:
    payload = {
        "pid": os.getpid(),
        "boot_id": boot_id(),
        "started_at": time.time(),
        "sealed": False,
        "sealed_at": None,
        **ticket_fields,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "owner.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _take_owner_lock(root)


def write_owner_for_ticket(ticket: BuildTicket) -> None:
    """(Re)write a generation's ownership tag from its ticket."""

    _write_owner(
        ticket.root,
        {
            "generation_id": ticket.generation_id,
            "asset_id": ticket.asset_id,
            "attempt_token": ticket.attempt_token,
        },
    )


def seal_generation(root: Path, manifest: dict) -> None:
    """Mark a generation finished: manifest first, then the seal."""

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    owner = _read_owner(root) or {}
    owner["sealed"] = True
    owner["sealed_at"] = time.time()
    (root / "owner.json").write_text(json.dumps(owner, indent=2), encoding="utf-8")
    # Sealed generations are judged by the drain window, not by liveness, and a
    # handle held for the rest of the process's life would be a leak.
    release_owner_lock(root)


def _read_owner(root: Path) -> dict | None:
    try:
        return json.loads((root / "owner.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The authority
# ---------------------------------------------------------------------------


def _terminal_reason(asset, state: dict) -> Reason | None:
    stage_outcome = TERMINAL_STAGES.get(asset.preprocess_stage or "")
    outcome = state.get("outcome")
    if outcome == OUTCOME_CANCELLED or stage_outcome == OUTCOME_CANCELLED:
        return Reason.CANCELLED
    if outcome == OUTCOME_FAILED or stage_outcome == OUTCOME_FAILED:
        return Reason.TERMINAL_FAILURE
    return None


def failure_detail(asset) -> str:
    """The *terminal* message for a failed import.

    Written only by :func:`record_attempt_failure`, which is called from the
    import's own unwinding, so a retry note from the job layer -- which writes
    ``preprocess_error`` and nothing else -- can never replace the real cause
    with a storage-lease conflict.
    """

    state = _load_state(_asset_of(asset))
    detail = (state or {}).get("failure_detail") or ""
    return detail or (asset.preprocess_error or "")


def resolve_pyramid(target, *, intent: Intent = Intent.READ) -> PublishedPyramid | Unavailable:
    """The one answer to "may this asset's pyramid be read or served?"."""

    asset = _asset_of(target)
    if asset is None:
        return Unavailable(Reason.NO_ASSET, "no asset row")

    state = _load_state(asset) or dict(_EMPTY_STATE)
    generation = state.get("published_generation") or NOT_PUBLISHED

    if generation == NOT_PUBLISHED:
        # Nothing is published, so this is the cold path and one more query is
        # affordable -- and necessary. ``preprocess_stage`` is written by the
        # job layer's reconciler through a queryset ``update()``, so an in-memory
        # ``Asset`` a caller has been holding says ``ENCODING`` long after the
        # import concluded. Reading it fresh here is what stops a stale object
        # turning a terminal failure into "never built, go and build it".
        current = _refreshed_stage(asset)
        terminal = _terminal_reason(current, state)
        if terminal is not None:
            return Unavailable(
                terminal, state.get("failure_detail") or asset.preprocess_error or ""
            )
        if _build_is_in_flight(current, state):
            return Unavailable(Reason.BUILDING, "the pyramid for this image is being built")
        return Unavailable(Reason.NEVER_BUILT, "no pyramid has been built for this image yet")

    terminal = _terminal_reason(asset, state)
    if terminal is not None:
        # A terminal import is never openable, whatever is on disk. The
        # published pointer is cleared by ``record_attempt_failure`` in the same
        # transaction that bumps the token, so a published generation and a
        # terminal outcome cannot both be true -- this is the invariant stated,
        # not a second guard on top of it.
        return Unavailable(terminal, state.get("failure_detail") or asset.preprocess_error or "")

    root = asset_generation_dir(asset.id) / generation
    if not root.is_dir():
        logger.warning(
            "Asset %s names published generation %s but %s is not on disk; "
            "treating it as unbuilt so it is rebuilt rather than served.",
            asset.id,
            generation,
            root,
        )
        return Unavailable(Reason.NEVER_BUILT, "the published pyramid is no longer on disk")

    manifest = state.get("published_manifest") or {}
    from .canonical_decode import DECODER_VERSION

    if manifest.get("decoder_version") and manifest["decoder_version"] != DECODER_VERSION:
        return Unavailable(
            Reason.STALE_DECODER,
            f"built by decoder {manifest['decoder_version']}, this build is {DECODER_VERSION}",
        )

    if intent is not Intent.SERVE:
        expected = (
            int(getattr(target, "height", 0) or asset.logical_height or 0),
            int(getattr(target, "width", 0) or asset.logical_width or 0),
        )
        levels = manifest.get("levels") or []
        if levels and expected != (0, 0):
            shape = tuple(int(value) for value in levels[0]["shape"])
            if (shape[-2], shape[-1]) != expected:
                return Unavailable(
                    Reason.GEOMETRY_MISMATCH,
                    f"level 0 is {shape[-1]}x{shape[-2]} but the image is "
                    f"{expected[1]}x{expected[0]}",
                )

    return PublishedPyramid(
        asset_id=str(asset.id),
        generation_id=generation,
        root=root,
        manifest=manifest,
    )


class _StageOnly:
    """Just enough of an ``Asset`` for :func:`_terminal_reason`."""

    __slots__ = ("preprocess_stage", "preprocess_error")

    def __init__(self, stage: str, error: str) -> None:
        self.preprocess_stage = stage
        self.preprocess_error = error


def _refreshed_stage(asset):
    """``asset``'s preprocessing stage as the database has it right now."""

    from .models import Asset

    row = (
        Asset.objects.filter(pk=asset.pk)
        .values_list("preprocess_stage", "preprocess_error")
        .first()
    )
    if row is None:
        return asset
    return _StageOnly(row[0] or "", row[1] or "")


#: The import pipeline is still working through this asset.
_IN_FLIGHT_STAGES = frozenset({"NONE", "ENCODING", "FEATURES", ""})


def _build_is_in_flight(current, state: dict) -> bool:
    """True when this asset's import has not concluded one way or the other.

    Used only to decide whether an unpublished asset gets a 202-and-enqueue or
    a 202-and-wait. Deliberately *not* a query over the job table: this is the
    cold path of a route the viewer polls, and the stage is already in hand
    from the same read that answered the terminal question. If the job dies
    without concluding, the reconciler moves the stage and the next GET
    enqueues -- so "in flight" cannot wedge.
    """

    return (
        state.get("outcome") == OUTCOME_PENDING
        and (current.preprocess_stage or "") in _IN_FLIGHT_STAGES
    )


# ---------------------------------------------------------------------------
# Attempt boundaries -- the fence
# ---------------------------------------------------------------------------


def begin_attempt(asset) -> str:
    """Start an import attempt: new token, ``PENDING``, published pointer kept.

    The published generation is deliberately *not* cleared here. A rebuild of
    an asset that is already viewable must keep serving the generation it has
    until the new one is published; that is what makes a rebuild invisible.
    """

    with transaction.atomic():
        state = dict(_ensure_state(asset))
        state["attempt_token"] = str(uuid.uuid4())
        state["outcome"] = OUTCOME_PENDING
        _write_state(asset, state, published_root=_published_root(asset, state))
    return state["attempt_token"]


def record_attempt_failure(asset, detail: str) -> str:
    """This attempt failed. Bump the fence and stop being openable.

    Called from the import's own unwinding, which is the only place that knows
    the real cause. Three things happen in one transaction:

    * the token is bumped, which fences out any build already running for the
      previous token -- **including one enqueued a second before the failure**,
      and including the between-attempt ``ENCODING`` stretch where a
      stage-based guard sees nothing;
    * the published pointer is cleared, which is the whole of withdrawal: one
      column write, no filesystem work, and it cannot fail;
    * ``failure_detail`` records the cause, on a field the job layer's retry
      note does not write.
    """

    with transaction.atomic():
        state = dict(_ensure_state(asset))
        state["attempt_token"] = str(uuid.uuid4())
        state["published_generation"] = NOT_PUBLISHED
        state["published_manifest"] = {}
        state["published_at"] = ""
        if detail:
            state["failure_detail"] = detail
        _write_state(asset, state, published_root=None)
    logger.info(
        "Asset %s: attempt failed, so nothing is published and the fence has moved; "
        "the asset is not openable until a later attempt publishes.",
        asset.id,
    )
    return state["attempt_token"]


def record_import_success(asset) -> str:
    with transaction.atomic():
        state = dict(_ensure_state(asset))
        state["attempt_token"] = str(uuid.uuid4())
        state["outcome"] = OUTCOME_SUCCEEDED
        state["failure_detail"] = ""
        _write_state(asset, state, published_root=_published_root(asset, state))
    return state["attempt_token"]


def record_terminal_failure(asset, detail: str) -> str:
    with transaction.atomic():
        state = dict(_ensure_state(asset))
        state["attempt_token"] = str(uuid.uuid4())
        state["outcome"] = OUTCOME_FAILED
        state["published_generation"] = NOT_PUBLISHED
        state["published_manifest"] = {}
        state["published_at"] = ""
        if detail:
            state["failure_detail"] = detail
        _write_state(asset, state, published_root=None)
    return state["attempt_token"]


def _published_root(asset, state: dict) -> Path | None:
    generation = state.get("published_generation") or NOT_PUBLISHED
    if generation == NOT_PUBLISHED:
        return None
    return asset_generation_dir(asset.id) / generation


# ---------------------------------------------------------------------------
# Tickets and the compare-and-swap
# ---------------------------------------------------------------------------


def request_build(asset, *, decoder_version: str = "") -> BuildTicket | Unavailable:
    """Permission to build one generation for ``asset``, or a reason not to."""

    state = _ensure_state(asset)
    terminal = _terminal_reason(_refreshed_stage(asset), state)
    if terminal is not None:
        return Unavailable(terminal, state.get("failure_detail") or asset.preprocess_error or "")

    # Cheap and keeps the common case tidy between maintenance passes; the
    # first build in a process also does the whole-tree pass that collects what
    # a kill left behind (rule 2, "from a previous boot", with no age threshold).
    try:
        _sweep_once_per_process()
        sweep_asset(asset.id, published_generation=state.get("published_generation") or "")
    except Exception:  # noqa: BLE001 - a sweep must never cost a build
        logger.warning("Sweeping %s's generations before a build failed.", asset.id, exc_info=True)

    generation_id = _new_generation_id()
    root = asset_generation_dir(asset.id) / generation_id
    ticket = BuildTicket(
        asset_id=str(asset.id),
        attempt_token=str(state.get("attempt_token") or ""),
        generation_id=generation_id,
        root=root,
        from_generation=str(state.get("published_generation") or NOT_PUBLISHED),
        decoder_version=decoder_version,
    )
    _write_owner(
        root,
        {
            "generation_id": generation_id,
            "asset_id": str(asset.id),
            "attempt_token": ticket.attempt_token,
        },
    )
    return ticket


def request_lazy_build(asset):
    """Enqueue *at most one* NGFF job for this asset's current attempt.

    Twenty simultaneous tile GETs used to produce twenty passes through a
    ``filter(...).first()`` and, measured, three jobs: they fought for the
    storage lease, their ``StorageLeaseConflict`` reconciled last, and the user
    was told about a lease instead of about the disk error that had actually
    broken the import.

    The design closes this with a unique ``Job.idempotency_key``. That column
    is in ``jobs/models.py``, which belongs to another workflow for the
    duration of this change, so the collapse is done with the primitive this
    application already has: **SQLite serialises writers**, and this whole
    check-and-insert runs inside one transaction that takes the write lock on
    its first statement. Concurrent callers therefore queue behind each other
    rather than interleaving between the ``SELECT`` and the ``INSERT``. The key
    itself -- asset plus attempt token -- is carried on the job payload, so a
    *new* attempt still gets its own job while a stale one can never be
    revived. Handoff: make it a unique column when ``jobs/models.py`` is free.

    Returns the job, or ``None`` when the asset has no attempt state yet.
    """

    from quantem.jobs.constants import JOB_TYPE_ENSURE_IMAGE_NGFF, QUEUE_P2_UPLOAD
    from quantem.jobs.models import Job

    with transaction.atomic():
        state = _ensure_state(asset)
        token = str(state.get("attempt_token") or "")
        # Takes the write lock, which is what serialises the concurrent callers.
        _write_state(asset, dict(state), published_root=_published_root(asset, state))
        existing = (
            Job.objects.filter(
                type=JOB_TYPE_ENSURE_IMAGE_NGFF,
                status__in=("PENDING", "RUNNING", "RETRY"),
                payload_json__asset_id=str(asset.id),
                payload_json__attempt_token=token,
            )
            .order_by("created_at")
            .first()
        )
        if existing is not None:
            return existing
        return Job.enqueue(
            job_type=JOB_TYPE_ENSURE_IMAGE_NGFF,
            payload={"asset_id": str(asset.id), "attempt_token": token},
            priority="high",
            resource_class="cpu",
            queue_name=QUEUE_P2_UPLOAD,
            tags=[f"asset:{asset.id}"],
        )


def publish(ticket: BuildTicket, manifest: dict) -> bool:
    """Make ``ticket``'s generation the published one, or discover it is stale.

    One ``UPDATE``. ``rowcount == 0`` means some other attempt began, a
    terminal outcome was recorded, or another build of the same attempt
    published first -- in every case this generation must not become live and
    nothing is touched.
    """

    from django.utils import timezone

    from .models import Rendition

    row = Rendition.objects.filter(asset_id=ticket.asset_id, type=Rendition.TYPE_NGFF).first()
    if row is None:
        return False
    metadata = dict(row.metadata if isinstance(row.metadata, dict) else {})
    state = dict(_EMPTY_STATE)
    state.update(metadata.get("pyramid") or {})
    state["published_generation"] = ticket.generation_id
    state["published_manifest"] = manifest
    state["published_at"] = timezone.now().isoformat()
    metadata["pyramid"] = state

    updated = Rendition.objects.filter(
        asset_id=ticket.asset_id,
        type=Rendition.TYPE_NGFF,
        metadata__pyramid__attempt_token=ticket.attempt_token,
        metadata__pyramid__published_generation=ticket.from_generation,
        metadata__pyramid__outcome__in=list(_PUBLISHABLE_OUTCOMES),
    ).update(
        metadata=metadata,
        stored_path=ticket.root.relative_to(NGFF_TMP_DIR).as_posix(),
        path_exists=True,
        is_directory=True,
    )
    if updated:
        logger.info(
            "Asset %s: published pyramid generation %s.", ticket.asset_id, ticket.generation_id
        )
        return True
    logger.info(
        "Asset %s: generation %s was superseded before it could be published "
        "(the attempt it belongs to is no longer current); discarding it.",
        ticket.asset_id,
        ticket.generation_id,
    )
    return False


def discard_generation(ticket: BuildTicket) -> None:
    """Delete a generation that will never be published. Never raises."""

    release_owner_lock(ticket.root)
    try:
        shutil.rmtree(ticket.root)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.info(
            "Could not delete the superseded generation %s yet (%s); the sweeper will.",
            ticket.root,
            exc,
        )


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    checked: int = 0
    removed: int = 0
    bytes_freed: int = 0
    still_held: int = 0
    kept: int = 0

    def summary(self) -> str:
        return (
            f"checked {self.checked}, removed {self.removed} "
            f"({self.bytes_freed} bytes), still held {self.still_held}, kept {self.kept}"
        )


def _tree_bytes(path: Path) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += (Path(base) / name).stat().st_size
    return total


def _remove_tree(path: Path, result: SweepResult) -> None:
    """Delete honestly. **Never** ``ignore_errors=True``.

    MEASURED: ``rmtree(ignore_errors=True)`` with a handle open inside leaves
    the root directory in place and reports nothing, which is how a 44 MB build
    root survived a restart *and* the rebuild that was supposed to collect it.
    A refusal here means a reader still holds a chunk; it is counted, left, and
    tried again next pass.
    """

    size = _tree_bytes(path)
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        result.still_held += 1
        logger.info("NGFF sweep: %s is still held (%s); leaving it for the next pass.", path, exc)
        return
    result.removed += 1
    result.bytes_freed += size


def _should_delete(child: Path, published: str, now: float) -> tuple[bool, str]:
    if child.name == published:
        return False, "published"
    if not child.name.startswith(GENERATION_PREFIX):
        # A legacy in-place store, a round-3 ``.building``/``.superseded``
        # directory, or anything else that is not a generation. The NGFF tree
        # is a rebuildable cache; none of these can be published, so none of
        # them may stay.
        return True, "not a generation directory"
    owner = _read_owner(child)
    if owner is None:
        try:
            age = now - child.stat().st_mtime
        except OSError:
            return True, "unreadable"
        if age > _UNOWNED_GRACE_SECONDS:
            return True, "no owner.json"
        return False, "too young to judge"
    if str(owner.get("boot_id") or "") != boot_id():
        return True, "from a previous boot"
    if not owner.get("sealed") and _owner_is_gone(child, owner):
        return True, "owning process is gone"
    if owner.get("sealed"):
        sealed_at = owner.get("sealed_at") or 0.0
        if now - float(sealed_at) > NGFF_DRAIN_SECONDS:
            return True, "sealed, superseded and drained"
        return False, "draining"
    return False, "being built by a live process"


def sweep_asset(asset_id, *, published_generation: str = "") -> SweepResult:
    """Collect every generation of one asset that is not the published one."""

    result = SweepResult()
    root = asset_generation_dir(asset_id)
    now = time.time()
    try:
        children = sorted(root.iterdir())
    except OSError:
        return result
    for child in children:
        result.checked += 1
        if not child.is_dir():
            try:
                child.unlink()
                result.removed += 1
            except OSError:
                result.still_held += 1
            continue
        delete, _why = _should_delete(child, published_generation, now)
        if delete:
            _remove_tree(child, result)
        else:
            result.kept += 1
    return result


#: The whole-tree pass runs once per process. The design puts it in the
#: scheduler's maintenance tick with an explicit pass at startup; ``jobs/**``
#: belongs to another workflow for the duration of this change, so the pass is
#: hung off the first build this process performs instead -- which is the same
#: moment for a server that does any work, and which is exercised directly by
#: ``test_ngff_kill_debris``. Handoff: one ``_sweep_ngff_if_due()`` beside
#: ``_sweep_uploads_if_due()`` plus a startup call, and this flag can go.
_process_sweep_done = False


def _sweep_once_per_process() -> None:
    global _process_sweep_done
    if _process_sweep_done:
        return
    _process_sweep_done = True
    sweep_ngff_generations()


def sweep_ngff_generations() -> SweepResult:
    """Every asset's generations, plus the whole tree of assets that are gone.

    Runs from the scheduler's maintenance tick and at startup. The startup pass
    is the one that matters: rule "a generation from a previous boot is debris"
    collects everything a kill left, with no age threshold, and without
    depending on a later build of the same image ever happening.
    """

    from .models import Asset

    result = SweepResult()
    try:
        entries = sorted(NGFF_TMP_DIR.iterdir())
    except OSError:
        return result

    published: dict[str, str] = {}
    live: set[str] = set()
    for asset in Asset.objects.all().only("id", "lifecycle_status").iterator():
        if asset.lifecycle_status != Asset.LIFECYCLE_DELETED:
            live.add(str(asset.id))
    for asset in Asset.objects.filter(lifecycle_status=Asset.LIFECYCLE_ACTIVE).prefetch_related(
        "renditions"
    ):
        state = _load_state(asset)
        if state:
            published[str(asset.id)] = state.get("published_generation") or ""

    for entry in entries:
        if not entry.is_dir() or not entry.name.endswith(".zarr"):
            continue
        asset_id = entry.name[: -len(".zarr")]
        if asset_id not in live:
            # Deleted or never-existed asset: the whole tree goes. This also
            # closes the pre-existing leak where ``tombstone_asset`` left the
            # published store behind.
            result.checked += 1
            _remove_tree(entry, result)
            continue
        one = sweep_asset(asset_id, published_generation=published.get(asset_id, ""))
        result.checked += one.checked
        result.removed += one.removed
        result.bytes_freed += one.bytes_freed
        result.still_held += one.still_held
        result.kept += one.kept
    if result.removed or result.still_held:
        logger.info("NGFF generation sweep: %s.", result.summary())
    return result
