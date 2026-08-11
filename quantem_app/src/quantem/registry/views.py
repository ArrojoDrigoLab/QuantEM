"""Model registry endpoints, per the API contract's Models section.

Two routes: what models exist and whether they can run here
(:class:`ModelListView`), and make one usable (:class:`ModelInstallView`). No
authentication anywhere — QuantEM is single-user and loopback-only.

The list body is assembled in :mod:`quantem.registry.catalogue`, which is plain
Python with no Django and no torch, so the shape can be tested without a
request. The views here are the thin HTTP layer over it.

Install sources
---------------
The contract lists three sources in order: an already-installed copy, a local
path, then the remote registry -- and all three exist. With no ``source_path``
the pack is **downloaded from the QuantEM Hugging Face repository** as a real
background job (:data:`quantem.jobs.constants.JOB_TYPE_INSTALL_MODEL_PACK`)
that reports byte-level progress and verifies every artifact's digest before
anything is installed; the response is a ``202`` whose ``job_id`` polls through
the ordinary jobs API.

"A local path" means, first and foremost, **a release bundle the user
downloaded and unzipped** -- see :mod:`quantem.registry.release`. Anywhere
inside the unzipped release counts: the release directory, its ``packs/``
folder, or one pack folder in it. A directory of raw training outputs is still
accepted, because a maintainer testing an unreleased head needs it to be.
Local installs are a file copy and run inline in the request; the download is
the one source slow enough to need the queue.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

from django.utils import timezone
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.inference.specs import MODEL_SPECS
from quantem.jobs.constants import JOB_TYPE_INSTALL_MODEL_PACK, QUEUE_P2_UPLOAD
from quantem.registry import cache, catalogue, release

logger = logging.getLogger(__name__)

#: Job ``type`` recorded for an install so the contract's
#: ``GET /api/jobs/<job_id>/`` polling flow works unchanged.
#:
#: One type serves both sources, differently: an HF download is enqueued
#: PENDING and executed by the registered handler
#: (:func:`quantem.jobs.handlers.handle_install_model_pack`) with real
#: progress, while a local-path copy runs inline in this request and its row is
#: written already terminal, so a client polling either gets the same shape.
INSTALL_JOB_TYPE = JOB_TYPE_INSTALL_MODEL_PACK


def _error(message: str, code: int = status.HTTP_400_BAD_REQUEST) -> Response:
    return Response({"error": message}, status=code)


class ModelListView(APIView):
    """``GET /api/models/``

    The eight released packs, the user's adapters, and the device inference
    would run on. Each pack carries ``runnable`` and, when false, a ``reason``:
    installing a pack only verifies files, and whether those files can be built
    into a module is a separate fact that used to be discovered several seconds
    into a run. See :func:`quantem.registry.catalogue.probe_runnable`.
    """

    def get(self, request: Request) -> Response:
        return Response(catalogue.catalogue(), status=status.HTTP_200_OK)


def _bundle_root(source_path: Path) -> Path | None:
    """The release bundle ``source_path`` is, or sits inside. None if neither.

    A user who unzips a release and then browses to "the model" lands in one of
    three places, and all three are the same release: the directory holding
    ``MANIFEST.json``, its ``packs/`` folder, and one pack directory inside that
    -- which is precisely what the on-screen instruction describes, so it is the
    one that has to work. Deciding by walking up to the manifest rather than by
    asking the caller keeps that a fact about the directory instead of a fact
    the user has to know.
    """
    for candidate in (source_path, source_path.parent, source_path.parent.parent):
        if (candidate / release.MANIFEST_NAME).is_file():
            return candidate
    return None


def _resolve_local_source(pack_id: str, source_path: Path) -> dict[str, Any]:
    """Turn a user-supplied directory of loose model files into install kwargs.

    Only reached when ``source_path`` is not, and is not inside, a release
    bundle. Accepts either shape a loose copy is plausibly in:

    * the model directory itself -- ``head.pt`` and ``resolved_config.yaml`` at
      the top, with ``encoder_ts.pt`` and/or ``checkpoint_index.json`` and
      ``encoder.pth`` beside them;
    * a directory of model directories, in which case ``<organelle>_<family>/``
      under it is used (the training-output directory naming).

    Raises:
        FileNotFoundError: naming the shapes that would have worked.
    """
    from quantem.registry.install import head_dirname

    head_dir = source_path
    if not (head_dir / cache.HEAD_NAME).is_file():
        candidate = source_path / head_dirname(pack_id)
        if (candidate / cache.HEAD_NAME).is_file():
            head_dir = candidate
    if not (head_dir / cache.HEAD_NAME).is_file():
        raise FileNotFoundError(
            f"{source_path} holds no {release.MANIFEST_NAME}, so it is not an unzipped "
            f"QuantEM model release and is not inside one, and no {cache.HEAD_NAME}, so "
            f"it is not a model directory either. Point source_path at one of:\n"
            f"  - the folder you unzipped a QuantEM model release into (it holds "
            f"{release.MANIFEST_NAME} and {release.PACKS_DIRNAME}/);\n"
            f"  - that release's {release.PACKS_DIRNAME}/ folder;\n"
            f"  - one pack folder inside it, "
            f"{release.PACKS_DIRNAME}/{cache.pack_dirname(pack_id)}/;\n"
            f"  - or a folder holding {cache.HEAD_NAME} and {cache.CONFIG_NAME} "
            f"(training outputs, named {head_dirname(pack_id)}/ under a heads folder)."
        )

    index = head_dir / cache.INDEX_NAME
    encoder = head_dir / cache.ENCODER_NAME
    return {
        "head_dir": head_dir,
        "encoder_index": index if index.is_file() else None,
        "encoder_file": encoder if encoder.is_file() else None,
        "search_dirs": [head_dir, source_path],
    }


class ModelInstallView(APIView):
    """``POST /api/models/<pack_id>/install/``

    Body (all optional)::

        {"source_path": "<dir on this machine>", "force": false}

    ``source_path`` is normally an unzipped QuantEM model release, or any
    directory inside one down to a single pack's; a directory of raw training
    outputs also works. Which one it is, is decided by looking for the release's
    ``MANIFEST.json`` at and above the given path, not by asking the caller.

    **No** ``source_path`` **means download.** The pack is fetched from the
    QuantEM Hugging Face repository by a real background job: ``202`` with the
    ``job_id`` to poll, the pinned repo/revision, and ``download_bytes`` so the
    client can show a real bar before the first progress row lands. Digest
    verification happens before anything is installed; a failure surfaces as
    the job's error, verbatim.

    For a local source the copy runs inline and the ``202`` arrives when the
    pack is already usable; the job row is written in its terminal state, so a
    client that polls ``GET /api/jobs/<job_id>/`` gets ``SUCCESS`` on its
    first poll either way once the work is done.
    """

    def post(self, request: Request, pack_id: str) -> Response:
        pack_id = str(pack_id or "").strip()
        if pack_id not in MODEL_SPECS:
            return _error(
                f"Unknown model pack {pack_id!r}. Known packs: {', '.join(sorted(MODEL_SPECS))}.",
                status.HTTP_404_NOT_FOUND,
            )

        data = request.data if isinstance(request.data, dict) else {}
        force = bool(data.get("force"))
        raw_source = str(data.get("source_path") or "").strip()

        # Source 1: an already-installed copy. Nothing to do, and saying so is
        # the whole answer -- re-hashing 660 MB to tell the user what they
        # already have is not a service.
        if cache.installed(pack_id) and not force:
            return Response(
                {
                    "job_id": None,
                    "pack_id": pack_id,
                    "status": "SUCCESS",
                    "detail": "Already installed.",
                    "pack": catalogue.pack_entry(pack_id),
                },
                status=status.HTTP_200_OK,
            )

        # An install already in flight for this pack: refuse to race it. This
        # is the same guard the first-launch queueing has
        # (quantem.registry.pending_installs._queue_pack); without it here, a
        # user whose installer-requested download was at 60% clicked the
        # Models screen's Download button and started a second, concurrent
        # 1.2 GB download of the same pack.
        active = catalogue.active_install_job(pack_id)
        if active is not None:
            state = "running" if active.status == "RUNNING" else "queued"
            return Response(
                {
                    # The job id used to be in this sentence. It is a UUID:
                    # nothing on any screen shows it, and there is nothing a
                    # reader can do with it (I-12). It stays in ``job_id``
                    # below, where the client uses it to find the row.
                    "error": (
                        f"{pack_id} is already {state} for download. Not "
                        "starting another; watch it in Tasks & Queues, or "
                        "cancel it there first if you meant to start over."
                    ),
                    "job_id": str(active.id),
                    "pack_id": pack_id,
                    "status": active.status,
                    "active_install": catalogue.active_install_entry(pack_id),
                },
                status=status.HTTP_409_CONFLICT,
            )

        # Source 3: no local source given -- download from the QuantEM Hugging
        # Face repository. A real background job with byte-level progress; the
        # download size is known before a byte moves, so it is reported here.
        if not raw_source:
            from quantem.jobs.models import Job
            from quantem.registry import hf

            spec = MODEL_SPECS[pack_id]
            job = Job.enqueue(
                job_type=JOB_TYPE_INSTALL_MODEL_PACK,
                payload={
                    "pack_id": pack_id,
                    "source": "huggingface",
                    "force": force,
                    "repo_id": hf.HF_REPO_ID,
                    "revision": hf.hf_revision(),
                },
                priority="default",
                resource_class="cpu",
                queue_name=QUEUE_P2_UPLOAD,
                max_attempts=1,
                tags=[f"model:{pack_id}"],
            )
            return Response(
                {
                    "job_id": str(job.id),
                    "pack_id": pack_id,
                    "status": "PENDING",
                    "source": "huggingface",
                    "repo_id": hf.HF_REPO_ID,
                    "revision": hf.hf_revision(),
                    "download_bytes": catalogue.download_bytes(spec),
                    # Read by a person, not by a client: the client already has
                    # ``job_id`` and the jobs API. An endpoint to poll is not an
                    # instruction anyone using the app can act on (I-12).
                    "detail": (
                        f"Downloading {pack_id} from {hf.HF_REPO_URL}. "
                        "Progress appears on this pack's card and in Tasks & Queues."
                    ),
                },
                status=status.HTTP_202_ACCEPTED,
            )

        source_path = Path(raw_source).expanduser()
        # ``is_dir()`` does not merely answer False for a path Windows cannot
        # reach: a disconnected mapped drive raises, and an uncaught OSError
        # here is a 500 with a Django traceback page in place of a sentence.
        try:
            reachable = source_path.is_dir()
        except OSError:
            reachable = False
        if not reachable:
            return _error(
                f"There is no folder at {source_path} that QuantEM can read. "
                "Check the location and try again."
            )

        install_kwargs: dict[str, Any] = {}
        bundle_root = _bundle_root(source_path)
        from_bundle = bundle_root is not None
        # What is actually installed from, which is not always what was typed:
        # a pack directory resolves to the release above it.
        install_root = bundle_root if bundle_root is not None else source_path
        if not from_bundle:
            try:
                install_kwargs = _resolve_local_source(pack_id, source_path)
            except FileNotFoundError as exc:
                return _error(str(exc))

        job = self._start_job(
            pack_id, install_root, force, "release-bundle" if from_bundle else "local-path"
        )
        try:
            from quantem.registry.install import (
                install_pack_from_bundle,
                install_pack_from_path,
            )

            if from_bundle:
                installed = install_pack_from_bundle(pack_id, install_root, force=force)
            else:
                head_dir = install_kwargs.pop("head_dir")
                installed = install_pack_from_path(pack_id, head_dir, force=force, **install_kwargs)
        except Exception as exc:
            logger.exception("Install of %s from %s failed", pack_id, install_root)
            self._finish_job(job, ok=False, message=str(exc))
            return _error(
                f"Installing {pack_id} from {install_root} failed: {exc}",
                status.HTTP_400_BAD_REQUEST,
            )

        entry = catalogue.pack_entry(pack_id)
        result = {
            "pack_id": pack_id,
            "source": "release-bundle" if from_bundle else "local-path",
            "source_path": str(install_root),
            "bytes_written": installed.bytes_written,
            "reused_blobs": installed.reused_blobs,
            "runnable": entry["runnable"],
            "reason": entry["reason"],
        }
        self._finish_job(job, ok=True, message="installed", result=result)
        return Response(
            {"job_id": str(job.id) if job else None, **result, "pack": entry},
            status=status.HTTP_202_ACCEPTED,
        )

    # --- Job bookkeeping ---------------------------------------------------

    @staticmethod
    def _start_job(pack_id: str, source_path: Path, force: bool, source: str) -> Any:
        """A RUNNING job row for this install, or None if the queue is unusable.

        Never fatal: the install itself does not need the job table, and a
        client that cannot be handed a job id still gets its pack.
        """
        try:
            from quantem.jobs.models import Job

            return Job.objects.create(
                type=INSTALL_JOB_TYPE,
                status="RUNNING",
                started_at=timezone.now(),
                heartbeat_at=timezone.now(),
                max_attempts=1,
                payload_json={
                    "pack_id": pack_id,
                    "source": source,
                    "source_path": str(source_path),
                    "force": force,
                },
                tags=[f"model:{pack_id}"],
            )
        except Exception:
            logger.warning("Could not record an install job row", exc_info=True)
            return None

    @staticmethod
    def _finish_job(job: Any, *, ok: bool, message: str, result: dict | None = None) -> None:
        if job is None:
            return
        try:
            job.status = "SUCCESS" if ok else "FAILED"
            job.progress = 100.0 if ok else job.progress
            job.message = message[:2000]
            job.finished_at = timezone.now()
            job.attempts = 1
            job.result_json = result
            if not ok:
                job.error_traceback = traceback.format_exc()
            job.save()
        except Exception:
            logger.warning(
                "Could not finalise install job %s", getattr(job, "id", "?"), exc_info=True
            )
