"""Job handlers that install or adapt a model.

Named for what they do, not for ``django.db.models``: this is
``quantem.jobs.handlers.models``, a submodule of the handler package, and it
holds no ORM model of its own.
"""

import logging

from quantem.jobs.constants import (
    JOB_TYPE_INSTALL_MODEL_PACK,
    JOB_TYPE_TRAIN_ORGANELLE_ADAPTER,
)
from quantem.jobs.handlers.common import _as_bool
from quantem.jobs.registry import job_handler
from quantem.jobs.reporter import CancelToken, JobReporter

logger = logging.getLogger(__name__)


def _model_display_name(pack_id: str) -> str:
    """What to call a model pack in a sentence a person reads.

    ``quantem:mito`` is a key in a table. The Models screen already shows the title
    (``QuantEM — Mitochondria``) and so does the download row in the Tasks
    drawer, so the job's own message uses the same words. An id this build does
    not know is not an error -- it is an older or newer release -- and the id
    is then the most honest thing available, since it is what the user typed or
    clicked.
    """
    try:
        # Lazy: the catalogue pulls in the model specs, and the other job types
        # must not pay that import.
        from quantem.registry.catalogue import MODEL_SPECS, pack_title  # noqa: PLC0415

        spec = MODEL_SPECS.get(pack_id)
        if spec is not None:
            title = pack_title(spec)
            if title:
                return title
    except Exception:
        logger.debug("no catalogue title for %s", pack_id, exc_info=True)
    return pack_id


@job_handler(JOB_TYPE_TRAIN_ORGANELLE_ADAPTER)
def handle_train_organelle_adapter(
    payload: dict, reporter: JobReporter, cancel: CancelToken
) -> dict:
    """Adapt a released model's head to the user's own annotated crops.

    The payload is passed through untouched. The older, single-segmentation
    route requires ``segmentation_id``; the scoped Fine-Tune dialog instead
    carries ``segmentation_type_id`` and ``asset_ids``. Both shapes share this
    worker, so reject only a payload that identifies neither kind of run.
    """
    cancel.check_cancelled()
    segmentation_id = str(payload.get("segmentation_id") or "").strip()
    scoped = bool(payload.get("asset_ids")) and bool(payload.get("segmentation_type_id"))
    if not segmentation_id and not scoped:
        raise ValueError("payload.segmentation_id is required")
    base_model = str(payload.get("base_model") or "").strip()
    if not base_model:
        raise ValueError("payload.base_model is required")

    # Imported lazily: adapter training pulls in torch, and every other job type
    # would pay that import cost at module load.
    from quantem.finetune.adapter_job import train_organelle_adapter_job

    reporter.update(progress=1.0, message="adapting model to your data")
    return train_organelle_adapter_job(
        payload=payload,
        reporter=reporter,
        cancel=cancel,
    )


@job_handler(JOB_TYPE_INSTALL_MODEL_PACK)
def handle_install_model_pack(payload: dict, reporter: JobReporter, cancel: CancelToken) -> dict:
    """Download a model pack from the QuantEM Hugging Face repository and install it.

    The whole pipeline -- fetch, digest verification, conversion to the pack
    format, TorchScript export, atomic promote -- lives in
    :mod:`quantem.registry.hf_install`; this handler is the progress and
    cancellation seam. Download bytes are known up front, so the bar reports a
    real fraction: 2-80% is the download, the rest is verify/convert/export.

    Cancellation is honoured between progress samples. The one thing that
    cannot be interrupted mid-flight is the byte transfer itself; an abandoned
    transfer finishes into huggingface_hub's content-addressed cache (where a
    retry reuses it) and never becomes an installed pack.
    """
    cancel.check_cancelled()
    pack_id = str(payload.get("pack_id") or "").strip()
    if not pack_id:
        raise ValueError("payload.pack_id is required")
    force = _as_bool(payload.get("force"))

    # Imported lazily: the registry's HF path pulls in huggingface_hub, and at
    # export time torch; every other job type must not pay those imports.
    from quantem.registry import catalogue
    from quantem.registry.hf_install import install_pack_from_hf

    # What to call this model in the queue. A pack id is a machine key; the
    # Models screen shows people a title, and ``Job.message`` is rendered
    # verbatim in the Tasks drawer, so the two have to agree about the name of
    # the thing being downloaded.
    model_name = _model_display_name(pack_id)

    reporter.update(progress=1.0, message=f"Asking for {model_name}")

    def on_bytes(done: int, total: int) -> None:
        if total > 0:
            reporter.update(
                progress=2.0 + 78.0 * (done / total),
                message=(f"Downloading {model_name} — {done / 1e6:.0f} of {total / 1e6:.0f} MB"),
                # Raw counts too, so the Models screen's active_install block
                # can show real bytes without parsing the message back apart.
                current_bytes=done,
                total_bytes=total,
            )

    def on_status(message: str) -> None:
        reporter.update(message=message)

    installed = install_pack_from_hf(
        pack_id,
        force=force,
        on_status=on_status,
        on_bytes=on_bytes,
        cancel_check=cancel.check_cancelled,
    )

    entry = catalogue.pack_entry(pack_id)
    summary = f"{model_name} is installed" + (
        "" if entry["runnable"] else ", but it cannot run on this machine"
    )
    reporter.update(progress=100.0, message=summary)
    return {
        "pack_id": pack_id,
        "source": "huggingface",
        "revision": installed.revision,
        "downloaded_bytes": installed.downloaded_bytes,
        "bytes_written": installed.bytes_written,
        "reused_blobs": installed.reused_blobs,
        "exported": installed.exported,
        **({"export_error": installed.export_error} if installed.export_error else {}),
        "runnable": entry["runnable"],
        "reason": entry["reason"],
        "encoder_tier": entry["encoder_tier"],
    }
