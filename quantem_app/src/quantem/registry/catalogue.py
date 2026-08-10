"""What models exist, what they cost, and whether this machine can run them.

This is the read model behind ``GET /api/models/``. It answers three questions
that live in three different places and have to be joined before the UI can
render a model picker honestly:

* **What is released** -- :data:`quantem.inference.specs.MODEL_SPECS` and
  :data:`quantem.registry.manifest.MEASURED_SIZES`, i.e. the eight
  (family x organelle) packs and the size of the bytes each one needs.
* **What is installed** -- :func:`quantem.registry.cache.installed`.
* **What would actually run** -- :func:`probe_runnable`, below.

The third question is the one that did not exist before. Installing a pack only
records verified files; whether those files can be turned into a *module* is a
separate fact, decided by :func:`quantem.inference.engine.build_module` at the
moment a user clicks Run. Four of the eight packs (the QuantEM family) sit on a
DINOv3 ViT-B whose architecture code QuantEM deliberately does not redistribute,
so on a machine with no exported ``encoder_ts.pt`` and no ``dinov3`` package
those packs install perfectly and then raise
:class:`~quantem.inference.engine.ModelArchitectureUnavailable` seconds into a
run. :func:`probe_runnable` surfaces that up front so the picker can grey the
pack out and say why.

**The probe is cheap on purpose.** It stats one file, reads one small JSON, and
asks :func:`importlib.util.find_spec` whether a package exists. It never loads a
weight file: a list request must not pay for a 1.2 GB encoder, and eight of them
even less.

Nothing here imports torch, and nothing here imports Django models at module
scope -- :func:`adapted_entries` reaches into :mod:`quantem.finetune` lazily so
the registry does not depend on guided fine-tuning being installed.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantem.inference.specs import FAMILY_LABELS, MODEL_SPECS, ModelSpec
from quantem.registry import cache
from quantem.registry.manifest import MEASURED_SIZES
from quantem.segmentation.type_definitions import (
    ER,
    LIPID_DROPLETS,
    MITOCHONDRIA,
    NUCLEUS,
)

logger = logging.getLogger(__name__)

#: Long organelle names for pack titles, from the canonical type definitions so
#: the model picker and the segmentation list cannot disagree about what an
#: organelle is called.
ORGANELLE_TITLES: dict[str, str] = {
    "mito": MITOCHONDRIA.long_name,
    "er": ER.long_name,
    "nucleus": NUCLEUS.long_name,
    "ld": LIPID_DROPLETS.long_name,
}

#: ``MEASURED_SIZES`` key for each family's shared encoder blob.
_ENCODER_SIZE_KEY: dict[str, str] = {
    "quantem": "quantem_vitb_encoder",
    "omniem": "omniem_vitl_encoder",
}

#: Licence shown before any byte is fetched. Both point at NOTICE rather than
#: naming an SPDX id: the repository is MIT and the weights are not covered by
#: it, and the OmniEM base encoder's upstream licence is still unconfirmed.
#: The released weights are CC BY 4.0 for both families (owner ruling
#: 2026-08-09), stated authoritatively on the Hugging Face repository the packs
#: are fetched from. One value here, no per-family indirection, and no
#: restatement anywhere else in this repository -- the licence lives with the
#: artifact it covers.
_LICENCE: dict[str, str] = {
    "quantem": "CC BY 4.0",
    "omniem": "CC BY 4.0",
}

_NOTES: dict[str, str] = {
    "quantem": (
        "Encoder and head trained by the Arrojo e Drigo Lab on the QuantEM EM "
        "corpus. The architecture code for this ViT-B is Meta's DINOv3, which "
        "QuantEM does not redistribute; packs ship with an exported encoder so "
        "it is not needed."
    ),
    "omniem": (
        "The head is the Arrojo e Drigo Lab's; the ViT-L base encoder is "
        "upstream EM-DINO (bioRxiv 10.1101/2025.04.13.648639) and carries its "
        "own licence. Built through timm."
    ),
}

#: Encoder frameworks and the package each one needs to be built eagerly. Keyed
#: by the ``encoder.framework`` value in a pack's ``checkpoint_index.json``.
_EAGER_REQUIREMENT: dict[str, str] = {
    "timm_vit": "timm",
    "dinov3": "dinov3",
}

#: Env var that points at a DINOv3 checkout; mirrors
#: :data:`quantem.inference.encoders.DINOV3_PATH_ENV_VAR`. Named here rather
#: than imported because ``encoders`` imports torch at module scope and this
#: module must stay import-cheap.
DINOV3_PATH_ENV_VAR = "QUANTEM_DINOV3_PATH"


@dataclass(frozen=True)
class Runnable:
    """Whether a pack can be turned into a running model on this machine.

    ``reason`` is user-facing and is only set when ``ok`` is False. ``tier`` is
    how the encoder would be built (``"exported"`` / ``"timm"`` / ``"dinov3"``),
    kept because an exported artifact is a different, digest-covered object from
    a graph rebuilt at run time and the difference is provenance.
    """

    ok: bool
    reason: str | None = None
    tier: str | None = None


def _module_available(name: str) -> bool:
    """True when ``name`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # ValueError: a parent package exists but has no __spec__.
        return False


def _dinov3_available() -> bool:
    """DINOv3 is importable, or a checkout is pointed at by the env hint."""
    if _module_available("dinov3"):
        return True
    hint = os.environ.get(DINOV3_PATH_ENV_VAR, "").strip()
    return bool(hint) and (Path(hint) / "dinov3").is_dir()


def _encoder_framework(index_path: Path) -> str | None:
    """``encoder.framework`` from an installed ``checkpoint_index.json``.

    Read as raw JSON rather than through
    :class:`quantem.inference.encoders.EncoderManifest` because that module
    imports torch at module scope and this one must not.
    """
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        return str(raw["encoder"]["framework"])
    except (OSError, ValueError, KeyError, TypeError):
        logger.debug("Unreadable encoder index at %s", index_path, exc_info=True)
        return None


def probe_runnable(pack_id: str, *, installed: bool | None = None) -> Runnable:
    """Cheap usability probe: would ``engine.load_model(pack_id)`` succeed?

    The order mirrors :func:`quantem.inference.encoders.build_encoder`:

    1. torch at all -- without it nothing in this package runs;
    2. the pack's files are installed;
    3. an exported ``encoder_ts.pt`` beside the head (tier a, the shipping path)
       -- self-describing, needs no architecture package;
    4. otherwise the eager tier named by the pack's own
       ``checkpoint_index.json``: ``timm`` for OmniEM, ``dinov3`` for QuantEM.

    Never loads a weight file. The answer can be wrong in one direction only --
    a pack reported runnable can still fail on a corrupt file or a shape
    mismatch, which is why the run path keeps its own errors.

    Args:
        pack_id: e.g. ``"quantem:mito"``.
        installed: pass a cached ``cache.installed(pack_id)`` to avoid a second
            filesystem check when building the whole catalogue.
    """
    if pack_id not in MODEL_SPECS:
        return Runnable(False, f"Unknown model pack {pack_id!r}.")

    if not _module_available("torch"):
        return Runnable(
            False,
            "PyTorch is not installed in this environment, so no model can run "
            "here. Threshold calibration still works; inference does not.",
        )

    if installed is None:
        installed = cache.installed(pack_id)
    if not installed:
        return Runnable(False, f"Not installed yet. {cache.INSTALL_HINT}")

    root = cache.pack_dir(pack_id)
    if (root / cache.EXPORTED_ENCODER_NAME).exists():
        return Runnable(True, None, "exported")

    index_path = root / cache.INDEX_NAME
    if not index_path.exists():
        return Runnable(
            False,
            f"No exported encoder and no {cache.INDEX_NAME} beside the weights, "
            "so nothing describes the architecture to rebuild. Reinstall the pack.",
        )

    framework = _encoder_framework(index_path)
    requirement = _EAGER_REQUIREMENT.get(framework or "")
    if requirement is None:
        return Runnable(
            False,
            f"Encoder framework {framework!r} is not one QuantEM can build "
            "('timm_vit' or 'dinov3').",
        )

    if requirement == "dinov3":
        if _dinov3_available():
            return Runnable(True, None, "dinov3")
        return Runnable(
            False,
            "This copy of the pack has no exported encoder, and rebuilding its "
            "ViT-B would need Meta's `dinov3` package, which QuantEM does not "
            "redistribute. Reinstall the pack -- an install from Hugging Face or "
            f"from a release bundle carries or builds {cache.EXPORTED_ENCODER_NAME}, "
            f"which needs nothing else. {cache.INSTALL_HINT}",
        )

    if _module_available(requirement):
        return Runnable(True, None, "timm")
    return Runnable(
        False,
        f"This pack's encoder is built through `{requirement}`, which is not "
        "installed here.",
    )


def download_bytes(spec: ModelSpec) -> int:
    """Bytes a fresh install of this pack must fetch.

    Head plus the family's shared encoder, except for a pack adapted with
    ``adapt: full`` (``quantem:er``), whose head file *is* a whole fine-tuned
    ViT-B and which therefore needs no separate encoder. Sizes are measured on
    disk; they are the only part of the release inventory that is currently
    real -- every upstream ``checkpoint_index.json`` carries ``sha256: null``.

    Note this is a *download* figure, not disk usage: three QuantEM packs share
    one encoder blob, so installing all four costs one copy of it, not three.
    """
    total = MEASURED_SIZES.get(f"{spec.organelle}_{spec.family}_head", 0)
    if not spec.embeds_encoder:
        total += MEASURED_SIZES.get(_ENCODER_SIZE_KEY[spec.family], 0)
    return total


def pack_title(spec: ModelSpec) -> str:
    """e.g. ``"QuantEM — Mitochondria"``."""
    return f"{FAMILY_LABELS[spec.family]} — {ORGANELLE_TITLES[spec.organelle]}"


def pack_licence(family: str) -> str:
    """Licence line for a family, shown before any byte is fetched.

    Public because a release bundle records it per pack: an artifact that
    travels to Hugging Face and Zenodo without its licence beside it is exactly
    the thing NOTICE exists to prevent, and the value must come from here rather
    than be retyped into the builder.
    """
    return _LICENCE[family]


def pack_notes(family: str) -> str:
    """Provenance note for a family. Public for the same reason as :func:`pack_licence`."""
    return _NOTES[family]


#: Job statuses under which an install is "in flight" for a pack: queued and
#: waiting, actually downloading, or between retry attempts. Mirrors
#: :data:`quantem.registry.pending_installs._ACTIVE_JOB_STATUSES` -- the same
#: set the first-launch queueing guard uses, so "the Models screen shows it as
#: in flight" and "a second install would be refused" can never disagree.
_ACTIVE_INSTALL_JOB_STATUSES: tuple[str, ...] = ("PENDING", "RUNNING", "RETRY")


def active_install_job(pack_id: str) -> Any | None:
    """The live install job row for a pack, or None.

    Imported lazily and failure-tolerant for the same reason as
    :func:`adapted_entries`: this module must stay importable, and a list
    request must still answer, without Django models (or with an unmigrated
    job table). A RUNNING job wins over queued ones; among queued ones the
    oldest -- the one the scheduler will run first -- is reported.
    """
    try:
        from quantem.jobs.constants import JOB_TYPE_INSTALL_MODEL_PACK
        from quantem.jobs.models import Job
    except Exception:  # pragma: no cover - jobs app always ships
        logger.debug("quantem.jobs is unavailable; no active installs", exc_info=True)
        return None

    try:
        jobs = list(
            Job.objects.filter(
                type=JOB_TYPE_INSTALL_MODEL_PACK,
                status__in=_ACTIVE_INSTALL_JOB_STATUSES,
                payload_json__pack_id=pack_id,
            ).order_by("created_at")
        )
    except Exception:  # table missing on an install that never migrated
        logger.debug("Job table unavailable", exc_info=True)
        return None

    for job in jobs:
        if job.status == "RUNNING":
            return job
    return jobs[0] if jobs else None


def active_install_entry(pack_id: str) -> dict[str, Any] | None:
    """The ``active_install`` block of one pack entry, or None.

    Shape (pinned in ``API_CONTRACT.md`` §Models)::

        {"job_id": "<uuid>", "status": "QUEUED" | "RUNNING",
         "progress_current_bytes": int | null,
         "progress_total_bytes": int | null}

    ``QUEUED`` covers every not-yet-running live status (PENDING and RETRY):
    the distinction the Models screen needs is "bytes are moving" vs "it will
    start on its own", not the queue's internal retry bookkeeping. The byte
    counts come from the job row, written by the download handler; null means
    the first progress sample has not landed yet.
    """
    job = active_install_job(pack_id)
    if job is None:
        return None
    return {
        "job_id": str(job.id),
        "status": "RUNNING" if job.status == "RUNNING" else "QUEUED",
        "progress_current_bytes": job.progress_current_bytes,
        "progress_total_bytes": job.progress_total_bytes,
    }


def pack_entry(pack_id: str) -> dict[str, Any]:
    """One ``packs[]`` element of ``GET /api/models/``."""
    spec = MODEL_SPECS[pack_id]
    installed = cache.installed(pack_id)
    runnable = probe_runnable(pack_id, installed=installed)
    return {
        "id": spec.pack_id,
        "family": spec.family,
        "organelle": spec.organelle,
        "title": pack_title(spec),
        "installed": installed,
        "download_bytes": download_bytes(spec),
        "canonical_nm": spec.canonical_nm,
        "tile_size": spec.tile_size,
        "default_threshold": spec.threshold,
        "decoder": spec.decoder,
        "neck": spec.neck,
        "adapt": spec.adapt,
        "licence": _LICENCE[spec.family],
        "notes": _NOTES[spec.family],
        # Beyond the contract's minimum, and the reason this module exists: a
        # pack can be installed and still unable to run. See probe_runnable.
        "runnable": runnable.ok,
        "reason": runnable.reason,
        "encoder_tier": runnable.tier,
        # The install already in flight for this pack, if any. Without it the
        # Models screen showed "not installed" with a live Download button
        # while the installer-requested download of the same 1.2 GB pack was
        # at 60% -- and the button queued a real duplicate.
        "active_install": active_install_entry(pack_id),
    }


def packs() -> list[dict[str, Any]]:
    """Every released pack, sorted by id."""
    return [pack_entry(pack_id) for pack_id in sorted(MODEL_SPECS)]


def adapted_entries() -> list[dict[str, Any]]:
    """The user's own adapters, in the contract's ``adapted[]`` shape.

    Imported lazily: :mod:`quantem.finetune` is a Django app with a model and a
    torch-adjacent training path, and the registry must stay usable (and
    importable) without it. An install where guided fine-tuning is not enabled
    gets an empty list, not an error.

    Only successful adapters appear. Every entry carries ``split_mode`` beside
    ``heldout_dice``, because the two numbers mean different things depending on
    it and the contract's honesty rules forbid showing one without the other.
    """
    try:
        from quantem.finetune.models import STATUS_SUCCESS, Adapter
    except Exception:  # app not installed, or no Django app registry
        logger.debug("quantem.finetune is unavailable; no adapted models", exc_info=True)
        return []

    try:
        rows = list(Adapter.objects.filter(status=STATUS_SUCCESS))
    except Exception:  # table missing on an install that never migrated
        logger.debug("Adapter table unavailable", exc_info=True)
        return []

    return [
        {
            "id": f"adapted:{adapter.id}",
            "base": adapter.base_model,
            "name": adapter.name or f"{adapter.base_model} adapter",
            "created_at": adapter.created_at,
            "calibrated_threshold": adapter.calibrated_threshold,
            "heldout_dice": adapter.heldout_dice,
            "split_mode": adapter.split_mode,
            "mode": adapter.mode,
            "segmentation_id": (
                str(adapter.segmentation_id) if adapter.segmentation_id else None
            ),
            "applied_at": adapter.applied_at,
        }
        for adapter in rows
    ]


def device_block() -> dict[str, Any]:
    """The ``device`` block: what inference would run on right now.

    Imports :mod:`quantem.inference.device` lazily and tolerates its failure --
    that module probes torch, and a list request must still answer on a machine
    where torch is broken or absent.
    """
    try:
        from quantem.inference.device import (
            cuda_available,
            describe_device,
            mps_available,
            select_device,
        )

        kind = select_device("auto")
        return {
            "kind": kind,
            "name": describe_device(kind),
            "cuda": cuda_available(),
            "mps": mps_available(),
        }
    except Exception:
        logger.debug("Device probe failed", exc_info=True)
        return {"kind": "cpu", "name": "CPU", "cuda": False, "mps": False}


def registry_block() -> dict[str, Any]:
    """The ``registry`` block: where a not-installed pack would download from.

    Static facts (no network): the pinned repository and revision every install
    verifies against. The UI shows them beside the install button so "download"
    is never a mystery source, and ``download_bytes`` on each pack already says
    what it costs.
    """
    from quantem.registry import hf

    return {
        "repo_id": hf.HF_REPO_ID,
        "revision": hf.hf_revision(),
        "url": hf.HF_REPO_URL,
    }


def catalogue() -> dict[str, Any]:
    """The whole ``GET /api/models/`` body."""
    return {
        "packs": packs(),
        "adapted": adapted_entries(),
        "device": device_block(),
        "registry": registry_block(),
    }
