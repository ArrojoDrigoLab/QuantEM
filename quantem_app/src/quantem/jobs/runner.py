import atexit
import contextlib
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
import weakref
from collections.abc import Iterator
from datetime import timedelta

from django.apps import apps
from django.utils import timezone

from quantem.jobs.artifact_registry import lease_paths_for_job
from quantem.jobs.constants import (
    JOB_TYPE_UPLOAD_IMAGE_PIPELINE,
    NO_RETRY_JOB_TYPES,
)
from quantem.jobs.failure_reconcile import (
    domain_status_recorded,
    failure_message,
    reconcile_domain_objects_for_cancelled_job,
    reconcile_domain_objects_for_failed_job,
    reconcile_domain_objects_for_retrying_job,
    retrying_attempt_detail,
    worker_exit_message,
)
from quantem.jobs.pool import (
    WORKER_PROCESS_ENV_VAR,
    django_pool_initializer,
    install_parent_death_watchdog,
)

logger = logging.getLogger(__name__)

RUNNING_HEARTBEAT_SECONDS = 30.0

#: Torch device a worker process was pinned to, published to that process's
#: environment so the inference layer can honour it. Unset means "no
#: accelerator was assigned" — run on CPU.
WORKER_DEVICE_ENV_VAR = "QUANTEM_WORKER_DEVICE"

# ``WORKER_PROCESS_ENV_VAR`` -- set inside a spawned job worker, so that only
# the server process runs a scheduler -- is imported from
# :mod:`quantem.jobs.pool` rather than declared again here. ``pool`` holds the
# copy a pool child can import before ``django.setup()``; this module used to
# declare a second string with the same value and a comment asking the next
# reader to keep them in step. The dependency runs one way only: ``pool`` must
# never import this module, which reaches Django models at import time.

#: How long :meth:`JobRunner.shutdown` waits for a terminated worker to go.
SHUTDOWN_JOIN_SECONDS = 5.0

#: Single persistent-worker pool for accelerator-class jobs.
GPU_POOL_KEY = "gpu"


def _parse_worker_count(raw_value: str | None, default: int) -> int:
    try:
        value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _detect_accelerator_devices() -> list[str]:
    """Torch device names workers may be pinned to, in round-robin order.

    Empty on a CPU-only machine: the worker device is then left unset and every
    job runs on CPU. Apple Silicon reports a single ``mps`` device — there is
    nothing to distribute across, but naming it lets the inference layer use it
    rather than silently falling back to CPU.
    """
    if os.environ.get("QUANTEM_CUDA_AVAILABLE", "").strip() in {"0", "false", "False"}:
        return []

    raw_count = os.environ.get("QUANTEM_GPU_COUNT")
    if raw_count is not None:
        try:
            count = max(0, int(str(raw_count).strip()))
        except (TypeError, ValueError):
            pass
        else:
            return [f"cuda:{index}" for index in range(count)]

    try:
        import torch
    except Exception:
        return []

    try:
        if torch.cuda.is_available():
            return [
                f"cuda:{index}"
                for index in range(max(0, int(torch.cuda.device_count())))
            ]
    except Exception:
        pass

    try:
        if torch.backends.mps.is_available():
            return ["mps"]
    except Exception:
        pass

    return []


def _configure_worker_device(device_name: str | None) -> None:
    normalized = str(device_name or "").strip()
    if not normalized:
        os.environ.pop(WORKER_DEVICE_ENV_VAR, None)
        return

    os.environ[WORKER_DEVICE_ENV_VAR] = normalized
    if not normalized.startswith("cuda"):
        return

    try:
        import torch
    except Exception:
        return

    try:
        torch.cuda.set_device(normalized)
    except Exception:
        logger.warning("Failed to bind worker process to CUDA device %s.", normalized)


def _get_job_model():
    return apps.get_model("jobs", "Job")


def _job_should_retry(job) -> bool:
    if job.type in NO_RETRY_JOB_TYPES:
        return False
    return job.attempts < job.max_attempts


def _setup_django() -> None:
    """Prepare a spawned job worker: exactly what a pool child gets.

    This is :func:`quantem.jobs.pool.django_pool_initializer`, called rather
    than re-implemented. The two had drifted into near-copies of each other --
    claim ``QUANTEM_JOB_WORKER`` before ``django.setup()`` opens the first
    connection (otherwise the child inherits ``QUANTEM_AUTOSTART_JOBS=1`` and
    runs a second scheduler racing this one for the same rows), point
    ``DJANGO_SETTINGS_MODULE`` at the settings module, then set Django up once
    -- and only one of them also installs the parent-death watchdog. Delegating
    means a job worker and a pool child cannot diverge again.
    """
    django_pool_initializer()


@contextlib.contextmanager
def _worker_process_marker() -> Iterator[None]:
    """Put ``QUANTEM_JOB_WORKER`` back the way it was when the job returns.

    :func:`_setup_django` claims the whole process as a worker, which is right
    for a spawned worker but permanent. Called in-process -- which the tests
    do, since this entry point *is* the real failure path they need to exercise
    -- it poisons the interpreter for everything that runs afterwards: file
    logging is suppressed in workers, so an unrelated later test can watch a
    server promise a log file that can never appear. The persistent worker
    claims the marker itself before its loop, so restoring the *previous* value
    leaves that claim standing.
    """
    previous = os.environ.get(WORKER_PROCESS_ENV_VAR)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(WORKER_PROCESS_ENV_VAR, None)
        else:
            os.environ[WORKER_PROCESS_ENV_VAR] = previous


def run_job_in_subprocess(job_id: str, device_name: str | None = None) -> None:
    # Before _configure_worker_device, which imports torch: that import is
    # seconds long and holds hundreds of megabytes, and a parent force-killed
    # during it would otherwise leave exactly the orphan this guards. A no-op in
    # the server process and in the in-process (inline) mode the tests use.
    install_parent_death_watchdog()
    with _worker_process_marker():
        _run_job_in_subprocess(job_id, device_name=device_name)


def _run_job_in_subprocess(job_id: str, device_name: str | None = None) -> None:
    _configure_worker_device(device_name)
    _setup_django()

    from quantem.jobs.registry import get_handler
    from quantem.jobs.reporter import CancelToken, JobCancelledError, JobReporter
    from quantem.jobs.storage_leases import (
        acquire_storage_artifact_leases,
        release_storage_artifact_leases,
    )

    Job = _get_job_model()
    job = Job.objects.get(id=job_id)
    reporter = JobReporter(job_id)
    cancel = CancelToken(job_id)

    try:
        cancel.check_cancelled()
        job.refresh_from_db()
        acquire_storage_artifact_leases(job, lease_paths_for_job(job))

        Job.objects.filter(id=job_id).update(heartbeat_at=timezone.now())
        job.refresh_from_db()
        handler = get_handler(job.type)
        result = handler(job.payload_json, reporter, cancel)
        job.refresh_from_db()
        release_storage_artifact_leases(job)
        Job.objects.filter(id=job_id).update(
            status="SUCCESS",
            progress=100.0,
            result_json=result or {},
            finished_at=timezone.now(),
            message="completed",
            heartbeat_at=timezone.now(),
        )
    except JobCancelledError:
        release_storage_artifact_leases(job)
        Job.objects.filter(id=job_id).update(
            status="CANCELLED",
            finished_at=timezone.now(),
            message="cancelled",
        )
        # Cancel is terminal too. Without this the job reads CANCELLED while the
        # thing it was carrying reads PENDING (analysis) or RUNNING (an adapter,
        # which is the row the Adapt wizard reads to decide what is in flight)
        # for the rest of the session.
        reconcile_domain_objects_for_cancelled_job(job.type, job.payload_json)
    except Exception as exc:
        error_trace = traceback.format_exc()
        Job = _get_job_model()
        job = Job.objects.get(id=job_id)
        release_storage_artifact_leases(job)
        next_status = "FAILED"
        next_run_at = timezone.now()
        if _job_should_retry(job):
            backoff_seconds = min(3600, 2 ** min(job.attempts, 8))
            next_status = "RETRY"
            next_run_at = timezone.now() + timedelta(seconds=backoff_seconds)
        # The exception's own sentence, never its class name: this string is
        # rendered verbatim in the Tasks drawer. See
        # :func:`quantem.jobs.failure_reconcile.failure_message`.
        message = failure_message(exc)
        conclusion = {
            "status": next_status,
            "error_traceback": error_trace,
            "finished_at": timezone.now(),
            "next_run_at": next_run_at,
            "message": message,
            "heartbeat_at": timezone.now(),
        }
        if next_status == "RETRY" and job.progress_units_total is not None:
            # The next attempt walks the tiles again from the first one. Leaving
            # the previous attempt's count on the row would show the wave more
            # done than it is and then take it back when the retry starts.
            conclusion["progress_units_done"] = 0
        Job.objects.filter(id=job_id).update(**conclusion)
        if next_status == "FAILED":
            reconcile_domain_objects_for_failed_job(
                job.type,
                job.payload_json,
                message,
                # A handler that wrote its own conclusion (marked on the
                # exception) owns the FAILED state and keeps its better
                # message. A job that died *before* writing anything cannot:
                # a FAILED stage it finds belongs to an OLDER attempt, whose
                # error must not outlive this newer failure.
                supersede_stale_failure=not domain_status_recorded(exc),
            )
        else:
            # RETRY is not a conclusion, but silence here left the domain
            # object showing a *previous* run's error for the whole retry
            # cycle. Surface this attempt's failure, marked as retrying;
            # a successful retry clears it.
            reconcile_domain_objects_for_retrying_job(
                job.type,
                job.payload_json,
                retrying_attempt_detail(job.attempts, job.max_attempts, message),
            )
        from quantem.jobs.storage_leases import StorageLeaseConflict

        if next_status == "RETRY" and isinstance(exc, StorageLeaseConflict):
            # Expected crash fallout: an unclean shutdown leaves a lease the
            # bounded retry clears in seconds. A full traceback here read as
            # an unexplained failure in the log of a session that healed
            # itself; the traceback is reserved for retries running out.
            logger.info(
                "Job %s: storage artifact still leased (likely an unclean "
                "shutdown); retrying automatically (attempt %d of %d).",
                job_id,
                job.attempts,
                job.max_attempts,
            )
        else:
            logger.error("Job %s failed: %s", job_id, error_trace)


def run_job_in_persistent_worker(
    job_queue: object,
    result_queue: object,
    device_name: str | None = None,
) -> None:
    # The process this most matters for: it holds a warm CUDA context and its
    # model weights for the whole session, and between jobs it blocks in
    # `job_queue.get()` -- a wait nothing else can interrupt. `daemon = True`
    # below does not help, because a force-quit is TerminateProcess and no
    # atexit hook runs.
    install_parent_death_watchdog()
    _configure_worker_device(device_name)
    _setup_django()
    Job = _get_job_model()
    while True:
        job_id = job_queue.get()
        if job_id is None:
            return
        run_job_in_subprocess(str(job_id), device_name=device_name)
        result_queue.put(str(job_id))
        # A failed CUDA job can leave the context in a state the next job
        # inherits, so retire the worker and let the pool spawn a clean one.
        # CPU and MPS workers carry no such state and stay alive.
        if str(device_name or "").startswith("cuda"):
            final_status = Job.objects.filter(id=job_id).values_list("status", flat=True).first()
            if final_status != "SUCCESS":
                return


class PersistentJobWorker:
    def __init__(self, ctx: object, *, pool_key: str, device_name: str | None = None) -> None:
        self.pool_key = pool_key
        self.device_name = str(device_name or "").strip() or None
        self.active_job_id: str | None = None
        self.job_queue = ctx.Queue(maxsize=1)
        self.result_queue = ctx.Queue(maxsize=1)
        self.process = ctx.Process(
            target=run_job_in_persistent_worker,
            args=(self.job_queue, self.result_queue, self.device_name),
        )
        self.process.daemon = True
        self.process.start()

    @property
    def exitcode(self) -> int | None:
        return self.process.exitcode

    def assign(self, job_id: str) -> None:
        if self.active_job_id is not None:
            raise RuntimeError(
                f"Persistent worker {self.pool_key} is already running job {self.active_job_id}."
            )
        self.active_job_id = str(job_id)
        self.job_queue.put(self.active_job_id)

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.process.join(timeout=timeout)

    def terminate(self) -> None:
        self.process.terminate()

    def try_consume_completion(self) -> bool:
        if self.active_job_id is None:
            return False
        try:
            completed_job_id = self.result_queue.get_nowait()
        except queue.Empty:
            return False

        expected_job_id = self.active_job_id
        self.active_job_id = None
        if str(completed_job_id) != str(expected_job_id):
            logger.warning(
                "Persistent worker %s reported completion for unexpected job %s (expected %s).",
                self.pool_key,
                completed_job_id,
                expected_job_id,
            )
        return True

    def close(self) -> None:
        for channel in (self.job_queue, self.result_queue):
            close = getattr(channel, "close", None)
            if callable(close):
                close()
            join_thread = getattr(channel, "join_thread", None)
            if callable(join_thread):
                join_thread()


def _is_persistent_job_worker(process: object) -> bool:
    return hasattr(process, "try_consume_completion") and hasattr(process, "pool_key")


#: Every live :class:`JobRunner`, so one ``atexit`` hook can stop all of them.
#: Weak, so a runner a test threw away is not kept alive to the end of the
#: session and does not have its (already collected) workers poked at exit.
_LIVE_RUNNERS: weakref.WeakSet = weakref.WeakSet()
_ATEXIT_REGISTERED = False


def _shutdown_live_runners() -> None:
    for runner in list(_LIVE_RUNNERS):
        try:
            runner.shutdown()
        except Exception:  # pragma: no cover - shutdown is best effort
            logger.debug("A job runner did not shut down cleanly.", exc_info=True)


def _register_runner_for_shutdown(runner: "JobRunner") -> None:
    """Arrange for ``runner``'s workers to be stopped when this process exits.

    The clean-exit half of the orphan problem. ``multiprocessing`` registers its
    own ``atexit`` hook when it is imported, and that hook *joins* non-daemon
    children -- so a plain Ctrl-C or a Quit from the shell blocked until the
    running segmentation finished, which for a full-image run is minutes of an
    app that looks hung. Users answer that by force-quitting, which is how the
    905 MB orphan got made in the first place. ``atexit`` runs handlers in
    reverse registration order and this one is registered later, so it
    terminates the workers before multiprocessing tries to wait for them.

    The killed job is not lost information: its row stays RUNNING and the next
    launch's startup reaper (``JobScheduler._recover_orphaned_jobs``) retries or
    fails it with the reason.
    """
    global _ATEXIT_REGISTERED

    _LIVE_RUNNERS.add(runner)
    if not _ATEXIT_REGISTERED:
        atexit.register(_shutdown_live_runners)
        _ATEXIT_REGISTERED = True


class RunningJob:
    def __init__(self, process: object, resource_class: str, job_type: str = ""):
        self.process = process
        self.resource_class = resource_class
        self.job_type = job_type
        self.started_at = time.monotonic()
        self.last_heartbeat = 0.0


class JobRunner:
    def __init__(self) -> None:
        mp.set_executable(sys.executable)
        self.inline = os.environ.get("JOB_RUNNER_INLINE", "").strip() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.inline_sync = os.environ.get("JOB_RUNNER_SYNC", "").strip() in {
            "1",
            "true",
            "yes",
            "on",
        }
        cpu_default = max(1, (os.cpu_count() or 2) - 1)
        self.cpu_slots = _parse_worker_count(os.environ.get("JOB_CPU_WORKERS"), cpu_default)
        self.upload_pipeline_slots = _parse_worker_count(
            os.environ.get("JOB_UPLOAD_PIPELINE_WORKERS"),
            5,
        )
        self.gpu_devices = _detect_accelerator_devices()
        # One worker per accelerator, and the default has to come from how many
        # there are. It was a flat 1, so `_next_gpu_device_name` round-robined
        # over cards that `_get_or_create_idle_gpu_worker` would never ask for:
        # a two-card workstation enumerated both and used one.
        #
        # One *per* card, not more: MEASURED, two processes sharing a card gain
        # 1.10x of throughput and four gain 1.20x, while per-run latency gets
        # 3.1x worse and VRAM use quadruples. The streaming multiprocessors are
        # already saturated by a single stream; the only thing concurrency
        # overlaps is host-side loading.
        self.gpu_slots = _parse_worker_count(
            os.environ.get("JOB_GPU_WORKERS"), max(1, len(self.gpu_devices))
        )
        self.ctx = mp.get_context("spawn")
        self.running: dict[str, RunningJob] = {}
        self.gpu_workers: dict[str, list[PersistentJobWorker]] = {}
        _register_runner_for_shutdown(self)

    def shutdown(self) -> None:
        """Terminate every worker process this runner started.

        Only processes, and only ones from ``self`` -- never a name-based sweep.
        Inline mode runs jobs on threads, which cannot be terminated and are
        daemons anyway, so they are skipped.
        """
        workers: list[object] = [job.process for job in self.running.values()]
        workers.extend(
            worker for pool in self.gpu_workers.values() for worker in pool
        )
        terminated = []
        for worker in workers:
            terminate = getattr(worker, "terminate", None)
            is_alive = getattr(worker, "is_alive", None)
            if not callable(terminate) or not callable(is_alive):
                continue  # a thread, in inline mode
            try:
                if not is_alive():
                    continue
                terminate()
                terminated.append(worker)
            except Exception:  # pragma: no cover - the process may be mid-exit
                logger.debug("Could not terminate a job worker.", exc_info=True)
        for worker in terminated:
            join = getattr(worker, "join", None)
            if callable(join):
                with contextlib.suppress(Exception):
                    join(timeout=SHUTDOWN_JOIN_SECONDS)
        if terminated:
            logger.info(
                "Stopped %d job worker process(es) on shutdown.", len(terminated)
            )

    def _next_gpu_device_name(self) -> str | None:
        if not self.gpu_devices:
            return None
        assigned_workers = sum(len(workers) for workers in self.gpu_workers.values())
        return self.gpu_devices[assigned_workers % len(self.gpu_devices)]

    def _available_slots(self, resource_class: str, job_type: str = "") -> int:
        if job_type == JOB_TYPE_UPLOAD_IMAGE_PIPELINE:
            active_cpu = sum(1 for job in self.running.values() if job.resource_class == "cpu")
            active_upload = sum(
                1
                for job in self.running.values()
                if job.job_type == JOB_TYPE_UPLOAD_IMAGE_PIPELINE
            )
            cpu_remaining = max(0, self.cpu_slots - active_cpu)
            upload_remaining = max(0, self.upload_pipeline_slots - active_upload)
            return min(cpu_remaining, upload_remaining)
        if resource_class == "gpu":
            active = sum(
                1 for job in self.running.values() if job.resource_class == "gpu"
            )
            return max(0, self.gpu_slots - active)
        active = sum(1 for job in self.running.values() if job.resource_class == "cpu")
        return max(0, self.cpu_slots - active)

    def can_dispatch(self, resource_class: str, job_type: str = "") -> bool:
        return self._available_slots(resource_class, job_type) > 0

    def _discard_gpu_worker(self, worker: object) -> None:
        pool_key = getattr(worker, "pool_key", None)
        if not isinstance(pool_key, str):
            return
        workers = self.gpu_workers.get(pool_key, [])
        if worker in workers:
            workers.remove(worker)
        close = getattr(worker, "close", None)
        if callable(close):
            close()

    def _get_or_create_idle_gpu_worker(self) -> PersistentJobWorker:
        workers = self.gpu_workers.setdefault(GPU_POOL_KEY, [])

        for worker in list(workers):
            if worker.is_alive():
                continue
            worker.join(timeout=0.1)
            self._discard_gpu_worker(worker)

        for worker in workers:
            if worker.active_job_id is None:
                return worker

        if len(workers) >= self.gpu_slots:
            raise RuntimeError(
                f"No idle GPU workers are available for pool {GPU_POOL_KEY}."
            )

        worker = PersistentJobWorker(
            self.ctx,
            pool_key=GPU_POOL_KEY,
            device_name=self._next_gpu_device_name(),
        )
        workers.append(worker)
        return worker

    def start_job(self, job_id: str, resource_class: str, job_type: str = "") -> None:
        if self.inline:
            if self.inline_sync:
                run_job_in_subprocess(job_id)
                return
            thread = threading.Thread(
                target=run_job_in_subprocess, args=(job_id,), daemon=True
            )
            thread.start()
            self.running[job_id] = RunningJob(thread, resource_class, job_type)
            return
        if resource_class == "gpu":
            worker = self._get_or_create_idle_gpu_worker()
            worker.assign(job_id)
            self.running[job_id] = RunningJob(worker, resource_class, job_type)
            return
        proc = self.ctx.Process(target=run_job_in_subprocess, args=(job_id,))
        proc.start()
        self.running[job_id] = RunningJob(proc, resource_class, job_type)

    def poll(self) -> None:
        Job = _get_job_model()
        finished = []
        now_monotonic = time.monotonic()
        for job_id, running in self.running.items():
            proc = running.process
            job = Job.objects.filter(id=job_id).first()
            should_heartbeat = (
                running.last_heartbeat <= 0.0
                or (now_monotonic - running.last_heartbeat) >= RUNNING_HEARTBEAT_SECONDS
            )
            if should_heartbeat:
                running.last_heartbeat = now_monotonic
                Job.objects.filter(id=job_id, status="RUNNING").update(
                    heartbeat_at=timezone.now()
                )
            if (
                job
                and job.cancel_requested
                and hasattr(proc, "is_alive")
                and proc.is_alive()
                and hasattr(proc, "terminate")
            ):
                proc.terminate()
                Job.objects.filter(id=job_id).update(
                    status="CANCELLED",
                    finished_at=timezone.now(),
                    message="cancelled",
                )
                # Terminated, so the worker's own release never runs.
                from quantem.jobs.storage_leases import release_storage_artifact_leases

                release_storage_artifact_leases(job)
                # This is the ordinary cancel -- the user pressed the button on a
                # job this runner owns -- and it is the one path that has to
                # reconcile *here*. Terminating the process means the worker's
                # own `except JobCancelledError` arm never runs, and the
                # scheduler's orphan-cancel branch only sees jobs with no live
                # worker. Without this the job reads CANCELLED while its
                # AnalysisRun or Adapter reads RUNNING for the rest of the
                # session: exactly the two-rows-two-truths screen the reconciler
                # exists to prevent, reached by the commonest route to it.
                reconcile_domain_objects_for_cancelled_job(job.type, job.payload_json)
            if _is_persistent_job_worker(proc) and proc.try_consume_completion():
                finished.append(job_id)
                continue
            if not proc.is_alive():
                exit_code = getattr(proc, "exitcode", None)
                if job and job.status == "RUNNING":
                    # Translated here rather than at the screen: the raw
                    # NTSTATUS reached the Analysis panel as "worker subprocess
                    # exited with code 3221225794", which is not a message for
                    # a biologist.
                    stopped_message = worker_exit_message(exit_code)
                    Job.objects.filter(id=job_id, status="RUNNING").update(
                        status="FAILED",
                        finished_at=timezone.now(),
                        message=stopped_message,
                        error_traceback=stopped_message,
                    )
                    # The worker died without releasing its storage leases;
                    # left ACTIVE they brick the segmentation for the lease TTL
                    # (6 h) -- see the scheduler reaper for the full account.
                    from quantem.jobs.storage_leases import (
                        release_storage_artifact_leases,
                    )

                    release_storage_artifact_leases(job)
                    reconcile_domain_objects_for_failed_job(
                        job.type,
                        job.payload_json,
                        stopped_message,
                        # The worker vanished without writing anything, so an
                        # existing FAILED stage is an older attempt's.
                        supersede_stale_failure=True,
                    )
                proc.join(timeout=0.1)
                if _is_persistent_job_worker(proc):
                    self._discard_gpu_worker(proc)
                finished.append(job_id)
        for job_id in finished:
            self.running.pop(job_id, None)
