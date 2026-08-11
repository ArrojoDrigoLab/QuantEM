"""Bringing a domain object down with the job that was carrying it.

Every long-running thing in QuantEM is two rows: a ``Job`` the queue owns, and a
domain object the screen shows -- an ``ImageSegmentation``, an ``AnalysisRun``,
an ``Adapter``. The handler moves the domain object through its own states, so
when the handler *never returns* -- the worker process died -- the job is failed
by the queue and the domain object is left exactly where the handler last put
it, which is usually ``PENDING``.

A user hit this twice. The analysis worker died with
``worker subprocess exited with code 3221225794`` immediately after a torch
fine-tuning job, and the Analysis screen then showed, at once: a history row
saying **PENDING**, a panel saying **FAILED**, and *"This run is pending.
Results appear when it finishes."* Two rows, two truths, one screen. Only a
restart cleared it.

So failure reconciliation lives here, keyed by job type, and every path that
fails a job goes through :func:`reconcile_domain_objects_for_failed_job` --
the in-worker exception handler, the runner's dead-process detection, the
scheduler's orphan reaper and its undispatched-job release. Adding a new job
type with a domain object behind it means adding one entry to
:data:`_RECONCILERS`; the alternative is another screen that disagrees with
itself.

The domain models are reached through ``apps.get_model`` rather than imported.
The queue must not depend on the feature packages at module scope -- they depend
on it -- and this module is imported by the runner, which is imported before
Django's app registry is fully populated in a spawned worker.

:func:`worker_exit_message` is the other half. ``3221225794`` is not a message
for a biologist; it is Windows ``STATUS_DLL_INIT_FAILED``, and what the person
needs to know is that restarting clears it.
"""

from __future__ import annotations

import contextlib
import logging

from django.apps import apps
from django.utils import timezone

from quantem.jobs.constants import (
    ACTIVE_SEGMENTATION_JOB_TYPES,
    JOB_TYPE_ENSURE_IMAGE_NGFF,
    JOB_TYPE_RUN_ANALYSIS,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
)

logger = logging.getLogger(__name__)

#: Domain statuses that a dead worker leaves behind and that this module
#: corrects. A record already ``SUCCESS`` or ``FAILED`` reached a conclusion of
#: its own and is never overwritten -- a handler that failed cleanly wrote a
#: better message than "the worker exited".
_UNFINISHED_STATUSES: frozenset[str] = frozenset({"PENDING", "RUNNING"})

#: Windows NTSTATUS values a worker process can die with, and what each one
#: actually means for the person waiting on the run. Reported *alongside* the
#: raw code, never instead of it: the number is what a bug report needs and the
#: sentence is what the user needs.
WINDOWS_EXIT_CODE_NOTES: dict[int, str] = {
    0xC0000005: (
        "The worker crashed while reading memory it does not own (an access "
        "violation). This is a fault in a native library, not in your image."
    ),
    0xC0000142: (
        "The worker could not start because one of the libraries it loads "
        "failed to initialise. This usually follows a heavy GPU job that left "
        "the graphics driver in a bad state. Restart QuantEM and run it again."
    ),
    0xC000013A: ("The worker was interrupted (Ctrl-C, or the console it belongs to was closed)."),
    0xC0000409: (
        "The worker was stopped by Windows after a native library overran a "
        "buffer. This is a fault in that library, not in your image."
    ),
    0xC0000017: (
        "The worker ran out of memory before it could start. Close other "
        "applications, or run over an ROI rather than the whole image."
    ),
}

#: POSIX signals a worker can be killed by; ``multiprocessing`` reports them as
#: a negative exit code.
_SIGNAL_NOTES: dict[int, str] = {
    9: (
        "The worker was killed by the operating system, which on this platform "
        "almost always means it ran out of memory. Try an ROI rather than the "
        "whole image."
    ),
    11: "The worker crashed with a segmentation fault inside a native library.",
    15: "The worker was asked to stop and did.",
}


#: What a job says when its exception had nothing sayable in it. A bare
#: ``KeyError('asset_id')`` is a fact about a dict, not a sentence, and the
#: traceback keeps every bit of it for whoever is debugging.
UNEXPLAINED_FAILURE_MESSAGE = (
    "This one stopped before it finished and did not say why. Nothing already "
    "saved was lost; run it again, and if it keeps happening the details are "
    "in the log file beside your data."
)


def failure_message(exc: BaseException) -> str:
    """What a person is told when a job fails.

    The queue used to write ``f"failed: {exc.__class__.__name__}: {exc}"``, and
    the Tasks drawer renders ``Job.message`` verbatim, so a user watching a run
    was handed::

        failed: ModelWeightsNotInstalled: Model pack 'quantem:er' is not
        installed. Install it on the Models screen.
        failed: ValueError: Error decoding PNG to 8-bit grayscale: image file
        is truncated

    The *sentences* are the app's own copy. The ``failed: <ClassName>:`` in
    front of them is the name of a Python class, which invariant I-12 forbids
    in anything a user reads and which tells them nothing: the row is already
    badged FAILED, and no reader has ever been helped by knowing which class
    the exception was. The class name is not lost -- it is in
    ``error_traceback``, and in the log, where a maintainer looks and a user
    does not.

    An exception whose text is not a sentence (a bare key, a lone number, an
    empty message) is replaced rather than shown: quoting ``'asset_id'`` at
    somebody is not more honest than saying the run stopped without explaining
    itself, it is only more confusing.
    """
    text = str(exc).strip()
    if not text or " " not in text:
        return UNEXPLAINED_FAILURE_MESSAGE
    return text


def worker_exit_message(exit_code: int | None) -> str:
    """What to tell a user whose worker process disappeared.

    ``None`` means the process object could not report a code at all. Otherwise
    the raw code is always included -- it is the only thing that identifies the
    crash in a bug report -- and a plain-language explanation is added for the
    codes we recognise.
    """
    if exit_code is None:
        return "The worker stopped before it finished, without reporting why."
    if exit_code == 0:
        return (
            "The worker exited without recording a result. Nothing already "
            "saved was lost; run it again."
        )

    if exit_code < 0:
        note = _SIGNAL_NOTES.get(-exit_code)
        detail = f"was killed by signal {-exit_code}"
    else:
        note = WINDOWS_EXIT_CODE_NOTES.get(exit_code)
        detail = f"exited with code {exit_code} / 0x{exit_code:08X}"

    if note:
        return f"{note} (The worker process {detail}.)"
    return (
        f"The worker process {detail} before the job finished. Nothing already "
        "saved was lost; run it again."
    )


#: Stages a segmentation does not get moved out of by a background failure.
#: ``COMPLETED`` is a segmentation's SUCCESS -- somebody clicked Mark Image Done
#: -- and it carries the completion lock. Overwriting it with FAILED silently
#: revoked a guarantee the user set by hand: every mutation that had been
#: returning 409 started succeeding again, the "locked" notice vanished, and
#: nothing recorded that Done had ever been set. ``FAILED`` is already a
#: conclusion. The other three reconcilers filter on their own unfinished sets
#: for the same reason; this one did not.
#:
#: ``FAILED`` is only *conditionally* protected, though -- see
#: ``supersede_stale_failure`` on :func:`_reconcile_segmentation`. A FAILED
#: stage is this job's own handler conclusion exactly when the handler marked
#: its exception (:func:`mark_domain_status_recorded`); a FAILED stage at the
#: death of a job whose handler wrote nothing belongs to an OLDER attempt, and
#: keeping it showed the previous run's error over the newest failure.
_CONCLUDED_SEGMENTATION_STAGES: frozenset[str] = frozenset({"COMPLETED", "FAILED"})

#: Attribute a handler sets on an exception after it has already written its
#: own (better) failure message onto the domain object. The failure paths read
#: it to decide whether an existing FAILED state is this attempt's conclusion
#: (skip: the handler's message wins) or a stale one from an older attempt
#: (supersede: the newest failure must show).
_DOMAIN_STATUS_RECORDED_ATTR = "_quantem_domain_status_recorded"


def mark_domain_status_recorded(exc: BaseException) -> BaseException:
    """Stamp ``exc``: its domain object's FAILED state was written by this attempt."""
    with contextlib.suppress(Exception):  # an exception refusing attributes
        setattr(exc, _DOMAIN_STATUS_RECORDED_ATTR, True)
    return exc


def domain_status_recorded(exc: BaseException) -> bool:
    """True when a handler already wrote this failure onto its domain object."""
    return bool(getattr(exc, _DOMAIN_STATUS_RECORDED_ATTR, False))


def _payload_segmentation_ids(payload: dict) -> list[str]:
    """Every segmentation a job was carrying.

    One for a single-organelle run; several for the one-run-per-image job, whose
    payload lists its organelles in ``legs``. A multi-organelle job that dies
    without its handler writing anything -- the worker was killed, the machine
    slept -- would otherwise leave every one of its organelles stuck at
    "Running" for ever, because the reconciler could only find the singular key.
    """
    payload = payload or {}
    ids = []
    single = str(payload.get("segmentation_id") or "").strip()
    if single:
        ids.append(single)
    for leg in payload.get("legs") or []:
        if not isinstance(leg, dict):
            continue
        leg_id = str(leg.get("segmentation_id") or "").strip()
        if leg_id and leg_id not in ids:
            ids.append(leg_id)
    return ids


def _reconcile_segmentation(
    payload: dict, error_message: str, *, supersede_stale_failure: bool = False
) -> None:
    segmentation_ids = _payload_segmentation_ids(payload)
    if not segmentation_ids:
        return
    model = apps.get_model("segmentation", "ImageSegmentation")
    # COMPLETED is always protected (the completion lock). FAILED is protected
    # only from a job whose handler wrote it -- when this job died with
    # nothing written (supersede_stale_failure=True), a FAILED stage can only
    # be an OLDER attempt's conclusion, and its error must not outlive the
    # newer attempt that just crashed: the header would keep explaining the
    # previous failure while saying nothing about this one.
    protected = (
        frozenset({"COMPLETED"}) if supersede_stale_failure else _CONCLUDED_SEGMENTATION_STAGES
    )
    if len(segmentation_ids) > 1:
        # A multi-organelle run: one job, several organelles, each with its own
        # outcome. An organelle that reached ``CANDIDATES_READY`` produced real
        # objects that are already on screen, and the job's error is about a
        # *different* organelle -- writing it over the finished one would say
        # the mitochondria failed because the nucleus model was missing. The
        # single-organelle case is untouched: there, the job's failure and the
        # segmentation's are the same event.
        protected = protected | {"THRESHOLD_READY", "CANDIDATES_READY"}
    updated = (
        model.objects.filter(id__in=segmentation_ids)
        .exclude(status_stage__in=protected)
        .update(status_stage="FAILED", status_error=error_message)
    )
    if updated:
        logger.info(
            "Marked %d segmentation(s) failed with their job: %s.",
            updated,
            ", ".join(segmentation_ids),
        )


def _reconcile_analysis_run(
    payload: dict, error_message: str, *, supersede_stale_failure: bool = False
) -> None:
    del supersede_stale_failure  # AnalysisRun keeps its own concluded set.
    run_id = str((payload or {}).get("analysis_run_id") or "").strip()
    if not run_id:
        return
    model = apps.get_model("analysis", "AnalysisRun")
    updated = model.objects.filter(
        id=run_id,
        status__in=_UNFINISHED_STATUSES,
    ).update(
        status="FAILED",
        error=error_message,
        finished_at=timezone.now(),
    )
    if updated:
        logger.info("Marked analysis run %s failed with its job.", run_id)


def _reconcile_adapter(
    payload: dict, error_message: str, *, supersede_stale_failure: bool = False
) -> None:
    del supersede_stale_failure  # Adapter keeps its own concluded set.
    adapter_id = str((payload or {}).get("adapter_id") or "").strip()
    if not adapter_id:
        return
    model = apps.get_model("finetune", "Adapter")
    updated = model.objects.filter(
        id=adapter_id,
        status__in=_UNFINISHED_STATUSES,
    ).update(status="FAILED", error=error_message)
    if updated:
        logger.info("Marked adapter %s failed with its job.", adapter_id)


#: Preprocessing stages an asset can still be moved out of. ``DONE``,
#: ``CANCELLED`` and ``SKIPPED`` are conclusions somebody reached on purpose.
_UNFINISHED_PREPROCESS_STAGES: frozenset[str] = frozenset({"NONE", "ENCODING", "FEATURES"})


def _reconcile_asset_preprocessing(
    payload: dict, error_message: str, *, supersede_stale_failure: bool = False
) -> None:
    del supersede_stale_failure  # Asset preprocessing keeps its own concluded set.
    asset_id = str((payload or {}).get("asset_id") or "").strip()
    if not asset_id:
        return
    model = apps.get_model("assets", "Asset")
    updated = model.objects.filter(
        id=asset_id,
        preprocess_stage__in=_UNFINISHED_PREPROCESS_STAGES,
    ).update(preprocess_stage="FAILED", preprocess_error=error_message)
    if updated:
        logger.info("Marked asset %s preprocessing failed with its job.", asset_id)


#: job type -> the domain object it carries. Every job type whose handler owns a
#: row a screen polls belongs here.
_RECONCILERS: dict[str, object] = {
    JOB_TYPE_RUN_ANALYSIS: _reconcile_analysis_run,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER: _reconcile_adapter,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE: _reconcile_asset_preprocessing,
    JOB_TYPE_ENSURE_IMAGE_NGFF: _reconcile_asset_preprocessing,
    **dict.fromkeys(ACTIVE_SEGMENTATION_JOB_TYPES, _reconcile_segmentation),
}


def reconcile_domain_objects_for_failed_job(
    job_type: str,
    payload: dict,
    error_message: str,
    *,
    supersede_stale_failure: bool = False,
) -> None:
    """Fail the domain object this job was carrying, if it is still unfinished.

    ``supersede_stale_failure`` says whether an *existing* FAILED state may be
    overwritten. The failure paths pass True unless this job's own handler
    already wrote its conclusion (:func:`domain_status_recorded` on the raised
    exception): a job that died with nothing written cannot own the FAILED
    state it finds, so that state is an older attempt's and its error message
    must not outlive this newer failure. The cancel and queue-removal paths
    leave it False -- they conclude unfinished work and preserve concluded
    work, as before.

    Never raises: this runs on the path that is already handling a failure, and
    a reconciliation that throws would replace a legible job error with a
    traceback from the error handler.
    """
    reconcile = _RECONCILERS.get(job_type)
    if reconcile is None:
        return
    try:
        reconcile(
            payload or {},
            error_message,
            supersede_stale_failure=supersede_stale_failure,
        )
    except Exception:
        logger.warning(
            "Could not reconcile the domain object for failed %s job.",
            job_type,
            exc_info=True,
        )


# --- Retrying attempts (paper-cut 1) ----------------------------------------
#
# A job that goes RETRY is not concluded, so the failed-job reconcilers above
# must not run -- but doing *nothing* left the domain object showing whatever
# error some previous run wrote, while newer, different failures accrued only
# in the queue. The labeling header read "the model pack is not installed" for
# the whole retry cycle of a job that was actually dying of something else.
#
# The honest surface: the error field carries the most recent attempt's
# failure, clearly marked as "attempt N of M failed; retrying", and *only* the
# error field -- the stage/status is the queue's business, and a successful
# retry clears the note (the segmentation status callback and the analysis
# success write both reset their error field to "").


def retrying_attempt_detail(attempts: int, max_attempts: int, error_message: str) -> str:
    """The one sentence a retrying job leaves on its domain object."""
    return f"Attempt {attempts} of {max_attempts} failed; retrying automatically. {error_message}"


def _note_segmentation_retry(payload: dict, error_message: str) -> None:
    segmentation_id = str((payload or {}).get("segmentation_id") or "").strip()
    if not segmentation_id:
        return
    model = apps.get_model("segmentation", "ImageSegmentation")
    # Unlike the failed-job reconciler, FAILED is *not* excluded: a stale
    # FAILED message from an earlier run is exactly what this note supersedes.
    # COMPLETED still is -- it carries the completion lock the user set by hand.
    model.objects.filter(id=segmentation_id).exclude(status_stage="COMPLETED").update(
        status_error=error_message
    )


def _note_analysis_run_retry(payload: dict, error_message: str) -> None:
    run_id = str((payload or {}).get("analysis_run_id") or "").strip()
    if not run_id:
        return
    model = apps.get_model("analysis", "AnalysisRun")
    model.objects.filter(id=run_id).exclude(status="SUCCESS").update(error=error_message)


def _note_adapter_retry(payload: dict, error_message: str) -> None:
    adapter_id = str((payload or {}).get("adapter_id") or "").strip()
    if not adapter_id:
        return
    model = apps.get_model("finetune", "Adapter")
    model.objects.filter(id=adapter_id).exclude(status="SUCCESS").update(error=error_message)


#: Preprocessing conclusions somebody reached on purpose; a retry note never
#: lands on them. ``FAILED`` is updatable for the same reason as above.
_CONCLUDED_PREPROCESS_STAGES: frozenset[str] = frozenset({"DONE", "CANCELLED", "SKIPPED"})


def _note_asset_preprocess_retry(payload: dict, error_message: str) -> None:
    asset_id = str((payload or {}).get("asset_id") or "").strip()
    if not asset_id:
        return
    model = apps.get_model("assets", "Asset")
    model.objects.filter(id=asset_id).exclude(
        preprocess_stage__in=_CONCLUDED_PREPROCESS_STAGES
    ).update(preprocess_error=error_message)


#: job type -> the error-field-only note for a retrying attempt. Mirrors
#: :data:`_RECONCILERS`; a job type carrying a domain object belongs in both.
_RETRY_RECONCILERS: dict[str, object] = {
    JOB_TYPE_RUN_ANALYSIS: _note_analysis_run_retry,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER: _note_adapter_retry,
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE: _note_asset_preprocess_retry,
    JOB_TYPE_ENSURE_IMAGE_NGFF: _note_asset_preprocess_retry,
    **dict.fromkeys(ACTIVE_SEGMENTATION_JOB_TYPES, _note_segmentation_retry),
}


def reconcile_domain_objects_for_retrying_job(
    job_type: str,
    payload: dict,
    error_message: str,
) -> None:
    """Surface the newest attempt's failure on the domain object it carries.

    Called from every path that moves a job to RETRY after an attempt actually
    ran: the worker's exception arm and the orphan reaper. Writes only the
    error field -- the domain status is left alone, and a successful retry
    clears the note. Never raises, for the same reason as
    :func:`reconcile_domain_objects_for_failed_job`.
    """
    reconcile = _RETRY_RECONCILERS.get(job_type)
    if reconcile is None:
        return
    try:
        reconcile(payload or {}, error_message)
    except Exception:
        logger.warning(
            "Could not record the retrying attempt on the domain object of %s job.",
            job_type,
            exc_info=True,
        )


#: What a cancelled job leaves on the domain object it was carrying.
#:
#: Neither ``AnalysisRun`` nor ``Adapter`` has a CANCELLED state, and adding one
#: is a migration plus every screen that renders a status. FAILED with a
#: sentence that says what happened is honest and unsticks the record; a real
#: CANCELLED state would be an improvement, not a correction.
CANCELLED_DETAIL = (
    "Cancelled before it finished, so it produced no result. Nothing was saved; "
    "start it again when you are ready."
)


def reconcile_domain_objects_for_cancelled_job(
    job_type: str,
    payload: dict,
    detail: str = CANCELLED_DETAIL,
) -> None:
    """Conclude the domain object a cancelled job was carrying.

    Cancel was the one terminal path that did not reconcile. The consequences
    were not symmetric with a crash, they were worse: a cancelled analysis left
    its ``AnalysisRun`` at PENDING forever beside a queue entry reading
    CANCELLED, and a cancelled head training left its ``Adapter`` at RUNNING --
    which is the row the Adapt wizard reads to decide what is in flight, so the
    wizard became permanently unusable for that segmentation, with no button to
    start again and no way to clear it.

    Cancel is also the button the app invites you to press on work it says will
    take "tens of minutes", so this is the *likely* path, not the rare one.
    """
    reconcile_domain_objects_for_failed_job(job_type, payload, detail)


#: What a job removed from the queue before it ever ran leaves behind.
#:
#: Deliberately different from :data:`CANCELLED_DETAIL`: a cancelled job got
#: some way in and might have written partial state, while this one was never
#: handed to a worker at all. Saying so is the difference between "your run
#: stopped" and "your run never started".
REMOVED_FROM_QUEUE_DETAIL = (
    "Removed from the queue before it started, so it never ran and produced no "
    "result. Nothing was saved; start it again when you are ready."
)


def reconcile_domain_objects_for_removed_job(
    job_type: str,
    payload: dict,
    detail: str = REMOVED_FROM_QUEUE_DETAIL,
) -> None:
    """Conclude the domain object behind a queued job the user removed.

    ``DELETE /api/jobs/<id>/`` is the *only* way out of a queued job --
    ``JobCancelView`` refuses anything that is not RUNNING with a 409 -- and it
    hard-deletes the row. That made it the one terminal path with no way back:
    :meth:`JobScheduler._recover_orphaned_jobs` iterates ``status="RUNNING"``,
    so a row that no longer exists is unreachable by every safety net in this
    module. Measured before this was wired in, one click on "Remove" (or
    "Cancel all") left:

    ==============================  ====================================
    queued job removed              domain object left at
    ==============================  ====================================
    ``run_analysis``                ``AnalysisRun.status = PENDING``
    ``run_segmentation_full_task``  ``ImageSegmentation`` stage PENDING
    ``train_organelle_adapter``     ``Adapter.status = PENDING``
    ==============================  ====================================

    all with an empty ``error`` and no queue row anywhere to explain them --
    the Analysis screen saying "This run is pending. Results appear when it
    finishes" about a run nothing will ever pick up, and the Adapt wizard
    reading a PENDING adapter as work in flight.
    """
    reconcile_domain_objects_for_failed_job(job_type, payload, detail)


__all__ = [
    "CANCELLED_DETAIL",
    "REMOVED_FROM_QUEUE_DETAIL",
    "UNEXPLAINED_FAILURE_MESSAGE",
    "WINDOWS_EXIT_CODE_NOTES",
    "domain_status_recorded",
    "failure_message",
    "mark_domain_status_recorded",
    "reconcile_domain_objects_for_cancelled_job",
    "reconcile_domain_objects_for_failed_job",
    "reconcile_domain_objects_for_removed_job",
    "reconcile_domain_objects_for_retrying_job",
    "retrying_attempt_detail",
    "worker_exit_message",
]
