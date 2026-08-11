"""Stream an upload straight into the staging directory.

Django's default handling writes a large upload to ``tempfile.gettempdir()``
and hands the view a file object; :func:`quantem.assets.utils
.save_uploaded_file_to_path` then copies it into ``UPLOADS_DIR``. Two full
copies of the image reach disk before the request can be answered -- three
counting the buffer waitress spills the request body into -- and only the last
one is kept.

The copy cannot simply become a rename. MEASURED on Windows 11 / Python 3.13:
``tempfile.NamedTemporaryFile``, which is what ``TemporaryUploadedFile`` uses,
opens with ``O_TEMPORARY``. ``os.replace`` on that path either raises
``PermissionError`` or -- worse -- succeeds and is then undone when the handle
closes, because ``FILE_FLAG_DELETE_ON_CLOSE`` deletes by the file's *current*
name. Renaming Django's temporary file is a silent data-loss bug on this
platform.

So the fix is upstream of the copy: an upload handler that puts the bytes in
their final directory to begin with, in an ordinary file we own, which the
view then claims with a same-directory rename. The saving is real work, not
bookkeeping -- this volume sustains ~50 MB/s of genuinely new writes
(MEASURED), so a 1 GB import loses a ~20 s write and a ~20 s read from the
window where the user is staring at "Uploading...".

It also stops the shipped app writing the image into the signed-in user's own
temporary folder under AppData, on the system drive: the Tauri shell sets
``QUANTEM_DATA_DIR`` and nothing else, so Django's default temporary directory
is on the system drive even when the data directory is not.

**Nothing in this directory is garbage-collected by the filesystem**, which is
the other half of what this module owns. Two leaks were MEASURED after a single
wave-0 verification session, both of them the user's disk:

* a rejected upload kept its full body -- 3 000 000 B and 500 008 B, answered
  400, still there six minutes and one restart later. The claim happens
  *before* the file is read, so ``extract_image_metadata`` rejecting a ``.tif``
  that is not a TIFF leaves the bytes behind under the asset id of an asset
  that was never created;
* killing the server mid-import left ``incoming-….tif``, **2 074 034 677 B**.

Storage lives with the installation (invariant I-11), so mis-dropping a 2 GB
file twice costs 4 GB of the volume QuantEM was installed onto, with nothing on
any screen to say so. :meth:`StagedUploadedFile.close` closes the first;
:func:`sweep_abandoned_uploads`, run by the job scheduler, closes the second.

**Why there is exactly one handler.** The first version of this module kept
Django's ``MemoryFileUploadHandler`` in front of the staging one, on the
reasoning that a body under ``FILE_UPLOAD_MAX_MEMORY_SIZE`` (2.5 MiB =
2 621 440 B) has no disk write to save. That reasoning was wrong twice over.
It saves nothing -- the small upload is still written to disk once, by
``save_uploaded_file_to_path``, just at the far end -- and it created a second
code path with no cleanup on it at all, because every rejected upload is
released by :meth:`StagedUploadedFile.close`, which an in-memory upload never
reaches. MEASURED on this tree, from an empty staging directory, one rejected
upload per row:

===========  =========================  ==========================
body         under the memory handler   with only the staging one
===========  =========================  ==========================
100 000 B    100 000 B kept forever     0 B
500 000 B    500 000 B kept forever     0 B
2 000 000 B  2 000 000 B kept forever   0 B
2 600 000 B  2 600 000 B kept forever   0 B
5 000 000 B  0 B                        0 B
===========  =========================  ==========================

Fifteen mis-drops across those five sizes left **15 600 000 B** with no asset
row, surviving a restart, because they were younger than the sweep's one-hour
threshold. So the handler list is one entry long and must stay that way: a size
threshold in front of a cleanup path means the cleanup is only tested at the
sizes the tests happen to use. ``assets/tests/test_upload_leak_sizes.py``
parametrises both sides of the old threshold and asserts the list itself.

**What this module still cannot do.** Django closes an uploaded file when the
*response* is closed, which under waitress happens after the response bytes
have been handed to the socket -- MEASURED at 5.0 ms median, 197.7 ms worst of
10 runs, and the client won the race in 10 of 10. So a client that reads the
error and immediately lists the directory can still see the body it just had
refused, for a few milliseconds. Removing that last window needs the code that
*knows the import failed* to say so while it is still inside the request;
:func:`discard_upload_if_unreferenced` is the hook for it, and the caller that
should use it is ``quantem.assets.asset_mutations.create_uploaded_asset``,
around everything it does after the claim.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from django.core.files.uploadedfile import UploadedFile
from django.core.files.uploadhandler import (
    FileUploadHandler,
    TemporaryFileUploadHandler,
)

from quantem.core.config import UPLOADS_DIR

logger = logging.getLogger(__name__)

#: Prefix for a not-yet-claimed staged upload. Distinguishes an in-flight or
#: abandoned body from ``<asset_id>.<ext>``, the staged file a created asset
#: owns, so a sweep can tell them apart.
STAGING_PREFIX = "incoming-"

#: How old an upload must be before :func:`sweep_abandoned_uploads` will touch
#: it. **How this number was chosen.** The window a file is legitimately
#: unowned is the request that is streaming it plus the metadata read that
#: follows the claim. MEASURED: this volume sustains ~50 MB/s of genuinely new
#: writes, and the wave-0 session's 837 MB upload took 8.7 s over loopback, so a
#: 2 GB import is ~40 s of body and a fanciful 20 GB one is ~7 minutes. An hour
#: is one to two orders of magnitude past the worst realistic case while still
#: bounding the leak to a single sweep interval, and it is not the only guard:
#: a file another handle is still writing cannot be deleted on Windows at all,
#: and this process's own in-flight paths are excluded by name.
DEFAULT_ABANDONED_UPLOAD_MAX_AGE_SECONDS = 3600

#: The floor an override cannot go under. A threshold shorter than an upload
#: takes would have the sweeper delete bytes out from under the request writing
#: them -- on POSIX, where unlinking an open file succeeds and the claim then
#: fails with a file-not-found the user would see as a corrupt import.
MIN_ABANDONED_UPLOAD_MAX_AGE_SECONDS = 300

#: Override for the above, in seconds.
UPLOAD_SWEEP_MAX_AGE_ENV_VAR = "QUANTEM_UPLOAD_SWEEP_MAX_AGE_SECONDS"

#: Staged paths this process has open right now. Age is not sufficient on its
#: own: Windows can hold a file's last-write time in the open handle's control
#: block, so a slow body being written for minutes can still look untouched.
_live_staged_paths: set[Path] = set()
_live_lock = threading.Lock()


def abandoned_upload_max_age_seconds() -> int:
    """Sweep threshold in seconds, honouring the override and its floor."""
    raw = os.environ.get(
        UPLOAD_SWEEP_MAX_AGE_ENV_VAR, str(DEFAULT_ABANDONED_UPLOAD_MAX_AGE_SECONDS)
    )
    try:
        requested = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_ABANDONED_UPLOAD_MAX_AGE_SECONDS
    return max(MIN_ABANDONED_UPLOAD_MAX_AGE_SECONDS, requested)


def _remember_live(path: Path) -> None:
    with _live_lock:
        _live_staged_paths.add(path)


def _forget_live(path: Path) -> None:
    with _live_lock:
        _live_staged_paths.discard(path)


def _live_paths() -> set[Path]:
    with _live_lock:
        return set(_live_staged_paths)


class StagedUploadedFile(UploadedFile):
    """An upload already sitting in ``UPLOADS_DIR``, claimable by rename."""

    def __init__(
        self,
        name,
        content_type,
        size,
        charset,
        content_type_extra=None,
        *,
        request_scoped: bool = False,
    ):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(name or "").suffix.lower()
        staged_path = UPLOADS_DIR / f"{STAGING_PREFIX}{uuid.uuid4().hex}{suffix}"
        super().__init__(
            # Deliberately not a context manager: the handle has to outlive
            # this constructor so the parser can stream into it. Ownership
            # passes to this object, which closes it in `claim` or `close` --
            # the same contract Django's own TemporaryUploadedFile has.
            open(staged_path, "w+b"),  # noqa: SIM115
            name,
            content_type,
            size,
            charset,
            content_type_extra,
        )
        self._staged_path = staged_path
        self._claimed = False
        self._claimed_target: Path | None = None
        self._closed = False
        #: Set by :class:`StagedFileUploadHandler`, i.e. true exactly when this
        #: object was made to serve an HTTP request. Only then is a claim
        #: *provisional* -- see :meth:`close`. Constructed directly (a test, a
        #: future caller with its own lifecycle) the object keeps the older,
        #: simpler contract: what it claims is the caller's from then on.
        self._request_scoped = bool(request_scoped)
        _remember_live(staged_path)

    def temporary_file_path(self) -> str:
        return str(self._staged_path)

    def claim(self, target_path: Path) -> None:
        """Take ownership of the bytes by moving them to ``target_path``.

        The handle is closed first: an open handle blocks a rename on Windows.
        After this the file is no longer *this object's* to delete on the old
        path -- but for a request-scoped upload the claim is provisional until
        the request records the asset; see :meth:`close`.
        """

        if self._claimed:
            raise RuntimeError("This staged upload has already been claimed")
        if not self.file.closed:
            self.file.flush()
            self.file.close()
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self._staged_path, target_path)
        _forget_live(self._staged_path)
        self._claimed = True
        self._claimed_target = target_path

    def close(self) -> None:
        """Release the bytes unless something now points at them.

        Django closes every file in ``request.FILES`` when the response is
        closed, so for an upload this is the end of the request.

        Two cases. **Never claimed** -- rejected before the copy, aborted,
        parsed and then ignored: the bytes are nobody's and go.

        **Claimed, but nothing in the database refers to them.** This is the
        leak F3 measured. ``create_uploaded_asset`` claims the file *before*
        reading it, so every failure after that point -- a ``.tif`` that is not
        a TIFF, an unreadable PNG, a reader raising anything at all, a 500 --
        answers the user with an error and keeps the whole body forever. There
        is no hook for "the view rejected this", and hunting the rejection
        paths one at a time is how the second one gets missed, so the question
        asked here is the one that actually matters: *is any Rendition pointing
        at this file?* If not, nobody can ever open it again.

        A database that cannot answer counts as "referenced": keeping bytes
        costs disk, deleting an accepted upload costs the user their image.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if not self.file.closed:
                self.file.close()
        finally:
            _forget_live(self._staged_path)
            if not self._claimed:
                # An upload that was rejected, aborted, or never claimed. Its
                # bytes are nobody's; leaving them would grow UPLOADS_DIR by
                # the size of every failed import.
                self._discard(self._staged_path, "abandoned staged upload")
            elif self._request_scoped and self._claimed_target is not None:
                self._discard_unclaimed_target(self._claimed_target)

    def _discard_unclaimed_target(self, target_path: Path) -> None:
        # ``require_upload_name=False``: this object staged the bytes itself, so
        # their provenance is known whatever the caller renamed them to. The
        # public helper is stricter because its caller may not know.
        discard_upload_if_unreferenced(target_path, require_upload_name=False)

    @staticmethod
    def _discard(path: Path, what: str) -> None:
        _discard_upload_bytes(path, what)


def _discard_upload_bytes(path: Path, what: str) -> bool:
    """Unlink ``path``, logging what went and how much of it. Never raises."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best-effort cleanup
        logger.warning("Could not remove %s %s", what, path, exc_info=True)
        return False
    logger.info("Removed %s %s (%d bytes).", what, path.name, size)
    return True


def discard_upload_if_unreferenced(
    path: Path, *, require_upload_name: bool = True
) -> bool:
    """Release upload bytes that nothing points at. ``True`` if they went.

    The question asked is the one that decides whether the file can ever be
    opened again: *is any rendition, or any asset id, pointing at it?* Hunting
    the individual rejection paths is how the next one gets missed -- the
    original leak was ``extract_image_metadata`` refusing a ``.tif`` that is
    not a TIFF, and the one after it was a size threshold, and neither was on
    anybody's list.

    Four refusals, all of them deliberate, because deleting an accepted upload
    costs the user their image while keeping a rejected one only costs disk:

    * anything outside ``UPLOADS_DIR`` -- another directory has its own owner;
    * anything this module did not name (``require_upload_name``, on by
      default): a file a user dropped in by hand is not ours to delete;
    * anything a handle in this process is still writing;
    * anything still referenced, *including* when the database cannot be asked.

    This is the hook the request layer needs. Django closes an uploaded file
    when the response closes, which is after the response bytes reach the
    client, so :meth:`StagedUploadedFile.close` -- correct, and the backstop
    for aborts and for code paths nobody remembered -- cannot make the
    directory right *before* the user is told the import failed. Calling this
    from the import path's own error handling can, and should:

    .. code-block:: python

        save_uploaded_file_to_path(uploaded_file, staged_path)
        try:
            metadata = extract_image_metadata(staged_path)
            ...
        except BaseException:
            discard_upload_if_unreferenced(staged_path)
            raise
    """
    path = Path(path)
    if path.parent.resolve(strict=False) != UPLOADS_DIR.resolve(strict=False):
        return False
    if require_upload_name and not _looks_like_an_upload_artifact(path):
        return False
    live = _live_paths()
    if path in live or path.resolve(strict=False) in live:
        return False
    if upload_is_referenced(path) is not False:
        return False
    return _discard_upload_bytes(
        path, "staged upload whose import did not complete"
    )


class StagedFileUploadHandler(TemporaryFileUploadHandler):
    """``TemporaryFileUploadHandler`` writing into ``UPLOADS_DIR`` instead."""

    def new_file(self, *args, **kwargs):
        FileUploadHandler.new_file(self, *args, **kwargs)
        self.file = StagedUploadedFile(
            self.file_name,
            self.content_type,
            0,
            self.charset,
            self.content_type_extra,
            request_scoped=True,
        )


def staged_upload_handlers(request) -> list[FileUploadHandler]:
    """The import endpoint's handlers: one, deliberately.

    **Do not add ``MemoryFileUploadHandler`` back.** It was here, first in the
    list, and every upload under ``FILE_UPLOAD_MAX_MEMORY_SIZE`` therefore
    became an ``InMemoryUploadedFile`` -- a file with no
    :meth:`StagedUploadedFile.close`, so every rejected import at those sizes
    kept its whole body under the id of an asset that was never created. The
    module docstring has the measurements; the short version is that four
    mis-drops of a half-megabyte file cost 2 000 016 B that survived a restart,
    and that the leak was invisible to a test suite which only exercised sizes
    above the threshold.

    It bought nothing, either. A small upload buffered in memory is still
    written to disk exactly once, by ``save_uploaded_file_to_path``; routing it
    through here writes it once too, and claims it with a rename that moves no
    bytes. What the second path added was a second lifecycle, at a boundary
    (2.5 MiB) that has nothing to do with anything a user can see.

    ``request`` is unused beyond what the handler needs; it is threaded through
    because that is Django's handler signature.
    """

    return [StagedFileUploadHandler(request)]


def _looks_like_an_upload_artifact(path: Path) -> bool:
    """True for a name this application put in ``UPLOADS_DIR``.

    Two shapes exist: ``incoming-<hex>.<ext>`` while the body is arriving, and
    ``<asset id>.<ext>`` once it is claimed. Anything else -- a file a user
    dropped in by hand, something a future feature parks here -- is left alone,
    so the sweeper can only ever delete bytes this module's own code wrote.
    """
    name = path.name
    if name.startswith(STAGING_PREFIX):
        return True
    try:
        uuid.UUID(path.stem)
    except ValueError:
        return False
    return True


def upload_is_referenced(path: Path) -> bool | None:
    """Whether anything in the database points at ``path``.

    ``None`` when the database could not be asked -- the caller must treat that
    as "referenced", never as "orphaned".

    Two questions, because either answer alone would be wrong. A ``Rendition``
    row is what makes a file openable, so its absence is what makes the bytes
    unreachable. But an ``Asset`` and its rendition are written in one
    transaction under the name ``<asset id>.<ext>``, so a matching asset id is a
    second, independent reason to keep a file: if that row exists, the import
    got far enough to own these bytes whatever the rendition says.
    """
    try:
        from quantem.assets.models import Asset, Rendition
        from quantem.core.config import DATA_DIR
        from quantem.core.local_storage import normalize_stored_path_value

        candidates = {
            normalize_stored_path_value(path, relative_to=DATA_DIR),
            str(path),
            path.as_posix(),
        }
        if Rendition.objects.filter(stored_path__in=sorted(candidates)).exists():
            return True
        try:
            asset_id = uuid.UUID(path.stem)
        except ValueError:
            return False
        return Asset.objects.filter(id=asset_id).exists()
    except Exception:
        logger.warning(
            "Could not check whether %s is still referenced; keeping it.",
            path,
            exc_info=True,
        )
        return None


def _owned_uploads() -> tuple[set[Path], set[str]] | None:
    """``(paths some rendition points at, ids of assets that exist)``.

    One query each, both read once per sweep rather than once per file. See
    :func:`upload_is_referenced` for why both halves are needed.
    """
    try:
        from quantem.assets.models import Asset, Rendition
        from quantem.core.config import DATA_DIR
        from quantem.core.local_storage import resolve_stored_path

        uploads = UPLOADS_DIR.resolve(strict=False)
        referenced: set[Path] = set()
        stored_paths = (
            Rendition.objects.exclude(stored_path="")
            .values_list("stored_path", flat=True)
            .iterator()
        )
        for raw in stored_paths:
            try:
                resolved = resolve_stored_path(raw, relative_to=DATA_DIR)
            except Exception:
                continue
            if resolved.parent == uploads:
                referenced.add(resolved)
        asset_ids = {
            str(asset_id)
            for asset_id in Asset.objects.values_list("id", flat=True).iterator()
        }
        return referenced, asset_ids
    except Exception:
        logger.warning(
            "Could not read the asset tables; keeping every claimed upload.",
            exc_info=True,
        )
        return None


@dataclass(frozen=True)
class UploadSweepResult:
    """What one pass of :func:`sweep_abandoned_uploads` did."""

    removed: tuple[Path, ...] = ()
    freed_bytes: int = 0
    kept: int = 0
    failed: tuple[Path, ...] = ()

    def summary(self) -> str:
        return (
            f"removed {len(self.removed)} abandoned upload file(s), "
            f"{self.freed_bytes} bytes; kept {self.kept}"
        )


def sweep_abandoned_uploads(
    *,
    max_age_seconds: int | None = None,
    now: float | None = None,
) -> UploadSweepResult:
    """Delete upload bytes nothing owns and nothing is still writing.

    Called at scheduler start-up and every
    ``JobScheduler.UPLOAD_SWEEP_INTERVAL_SECONDS`` after that. What it removes
    is a file that is **all** of:

    * one this module named (:func:`_looks_like_an_upload_artifact`);
    * older than :func:`abandoned_upload_max_age_seconds`, so an upload still
      arriving is never a candidate;
    * not open in this process (``incoming-`` files being streamed right now);
    * and, unless it carries the ``incoming-`` prefix -- which no database row
      ever refers to -- not referenced by any asset or rendition.

    On Windows there is a fourth guard for free: a handle writing the file
    denies delete sharing, so ``unlink`` raises and the file is kept. That
    covers the case age cannot, which is another process mid-import.

    ``max_age_seconds`` and ``now`` exist for the tests.
    """
    if not UPLOADS_DIR.exists():
        return UploadSweepResult()

    cutoff = (time.time() if now is None else now) - (
        abandoned_upload_max_age_seconds() if max_age_seconds is None else max_age_seconds
    )
    live = _live_paths()
    owned: tuple[set[Path], set[str]] | None = None
    owned_loaded = False

    removed: list[Path] = []
    failed: list[Path] = []
    freed = 0
    kept = 0

    for path in sorted(UPLOADS_DIR.iterdir()):
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        if not _looks_like_an_upload_artifact(path):
            kept += 1
            continue
        if path in live or path.resolve(strict=False) in live:
            kept += 1
            continue
        if stat.st_mtime > cutoff:
            kept += 1
            continue
        if not path.name.startswith(STAGING_PREFIX):
            if not owned_loaded:
                owned = _owned_uploads()
                owned_loaded = True
            if owned is None:
                kept += 1
                continue
            referenced_paths, asset_ids = owned
            if path.resolve(strict=False) in referenced_paths or path.stem in asset_ids:
                kept += 1
                continue
        try:
            size = stat.st_size
            path.unlink()
        except OSError:
            # Almost always a handle still writing it: on Windows that denies
            # delete sharing, which is exactly the answer we want.
            logger.info(
                "Left %s in place; it is still in use or could not be removed.",
                path.name,
            )
            failed.append(path)
            kept += 1
            continue
        removed.append(path)
        freed += size

    result = UploadSweepResult(
        removed=tuple(removed),
        freed_bytes=freed,
        kept=kept,
        failed=tuple(failed),
    )
    if removed:
        logger.info("Upload sweep: %s.", result.summary())
    return result
