"""First-launch model installs requested by the desktop installer (Ruling C).

The NSIS installer offers checkboxes for the eight model packs, but it must
not download anything itself: the app's install machinery is the tested path
-- digest verification before anything lands, byte-level progress, cancel, the
antivirus-rename retry. Raw NSIS has none of that. So the installer only
*writes down the request*, as a small file in the chosen data directory::

    <data dir>/pending-model-installs.json
    {"packs": ["omniem:mito", "omniem:er", ...]}

and on server startup -- ``quantem serve``, which is also what the frozen
desktop build runs -- :func:`process_pending_model_installs` turns it into one
ordinary ``install_model_pack`` job per not-yet-installed pack, with exactly
the payload the Models screen's install endpoint enqueues
(:class:`quantem.registry.views.ModelInstallView`). The downloads then show up
in Tasks & Queues with progress and a cancel button, and a failure surfaces on
the Models screen verbatim with a Retry, like any other install.

The file is a **request, not state**:

* it is deleted even when only some of it could be queued -- a pack id the
  installer wrote that this build does not know, a pack already installed, an
  install already in flight are each logged and skipped, never re-attempted on
  every launch;
* a malformed file is logged and deleted;
* nothing in here may ever crash the server. A user whose install request is
  lost can install from the Models screen in two clicks; a user whose server
  will not start can do nothing at all.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Written by the NSIS installer into the install's data directory. The desktop
#: packaging (quantem_app/desktop) spells this name in its installer hooks;
#: change it there too or first launches will silently stop finding requests.
PENDING_INSTALLS_FILENAME = "pending-model-installs.json"

#: Job statuses under which a second install job for the same pack would race
#: the first rather than help it.
_ACTIVE_JOB_STATUSES = ("PENDING", "RUNNING", "RETRY")


def pending_installs_path() -> Path:
    """``<data dir>/pending-model-installs.json`` for this process's data dir."""
    from quantem.core.config import STORAGE_DIR

    return Path(STORAGE_DIR) / PENDING_INSTALLS_FILENAME


def process_pending_model_installs() -> list[str]:
    """Queue the installer-requested packs; return the pack ids queued.

    Called from ``quantem.cli.cmd_serve`` after the migrations, so the job
    table exists by the time anything is enqueued. Never raises.
    """
    try:
        return _process()
    except Exception:
        logger.warning(
            "Processing the pending model install request failed; continuing "
            "to serve. Models can be installed from the Models screen.",
            exc_info=True,
        )
        return []


def _process() -> list[str]:
    path = pending_installs_path()
    if not path.is_file():
        return []

    queued: list[str] = []
    try:
        pack_ids = _read_request(path)
        for pack_id in pack_ids:
            if _queue_pack(pack_id):
                queued.append(pack_id)
    finally:
        # One-shot, deleted even on partial queueing: re-running a request
        # that half-failed on every launch would re-log the same skips forever
        # and re-download nothing -- the surviving jobs are already in the
        # queue, which is the durable record.
        try:
            path.unlink()
        except OSError:
            logger.warning("Could not delete %s; it may be re-read.", path)

    if queued:
        logger.info(
            "Queued %d model install(s) at the installer's request: %s",
            len(queued),
            ", ".join(queued),
        )
    return queued


def _read_request(path: Path) -> list[str]:
    """The requested pack ids, or ``[]`` (with a log line) for a bad file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        packs = data["packs"]
        if not isinstance(packs, list):
            raise TypeError(f'"packs" is {type(packs).__name__}, not a list')
        pack_ids = [str(p).strip() for p in packs]
        return [p for p in pack_ids if p]
    except Exception as exc:
        logger.warning(
            "Ignoring malformed %s (%s: %s). The installer writes "
            '{"packs": ["omniem:mito", ...]}; models can always be installed '
            "from the Models screen.",
            path,
            exc.__class__.__name__,
            exc,
        )
        return []


def _queue_pack(pack_id: str) -> bool:
    """Enqueue one pack's download unless there is nothing to do.

    The job is byte-for-byte the one ``POST /api/models/<pack_id>/install/``
    enqueues (plus a ``requested_by`` breadcrumb), so the handler, the
    progress polling, the Models screen's failed-install surfacing and Retry
    all see a job they already know.
    """
    from quantem.inference.specs import MODEL_SPECS
    from quantem.jobs.constants import JOB_TYPE_INSTALL_MODEL_PACK, QUEUE_P2_UPLOAD
    from quantem.jobs.models import Job
    from quantem.registry import cache, hf

    if pack_id not in MODEL_SPECS:
        logger.warning(
            "The installer requested unknown model pack %r; skipping it. "
            "Known packs: %s",
            pack_id,
            ", ".join(sorted(MODEL_SPECS)),
        )
        return False
    if cache.installed(pack_id):
        logger.info("Model pack %s is already installed; nothing to queue.", pack_id)
        return False
    if (
        Job.objects.filter(
            type=JOB_TYPE_INSTALL_MODEL_PACK,
            status__in=_ACTIVE_JOB_STATUSES,
            payload_json__pack_id=pack_id,
        ).exists()
    ):
        logger.info(
            "An install of model pack %s is already queued or running; not "
            "queueing another.",
            pack_id,
        )
        return False

    job = Job.enqueue(
        job_type=JOB_TYPE_INSTALL_MODEL_PACK,
        payload={
            "pack_id": pack_id,
            "source": "huggingface",
            "force": False,
            "repo_id": hf.HF_REPO_ID,
            "revision": hf.hf_revision(),
            "requested_by": "installer",
        },
        priority="default",
        resource_class="cpu",
        queue_name=QUEUE_P2_UPLOAD,
        max_attempts=1,
        tags=[f"model:{pack_id}"],
    )
    logger.info(
        "Queued install of model pack %s (job %s) at the installer's request.",
        pack_id,
        job.id,
    )
    return True


__all__ = [
    "PENDING_INSTALLS_FILENAME",
    "pending_installs_path",
    "process_pending_model_installs",
]
