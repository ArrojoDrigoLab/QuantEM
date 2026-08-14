"""
Background tasks for image preprocessing.

These are plain Python helpers invoked by the DB job handlers.
"""

import logging
import os
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path

from django.db import transaction

from quantem.core.config import IMAGES_DIR
from quantem.core.local_storage import normalize_stored_path_value
from quantem.segmentation.roi_selection import select_roi_for_image

from .asset_openable import get_asset_openable
from .border_trim import should_trim_initial_import, trim_black_or_white_border
from .canonical_decode import decode_canonical_plane
from .models import Asset, ImageROI, Rendition
from .ngff import PyramidBuildRefused, bounded_ngff_build_resources, build_and_publish
from .preprocess_status import set_stage
from .pyramid_authority import (
    begin_attempt,
    record_attempt_failure,
    record_import_success,
    record_terminal_failure,
)
from .roi_state import get_active_roi_for_asset
from .utils import create_roi_image_from_image, save_plane_as_canonical_png

logger = logging.getLogger(__name__)

ROI_SIZE_DEFAULT = int(os.environ.get("ROI_SIZE", "1024"))
ROI_MIN_IMAGE_SIZE = int(os.environ.get("ROI_MIN_IMAGE_SIZE", "512"))


def _get_asset_or_none(asset_id: str) -> Asset | None:
    try:
        asset = Asset.objects.get(id=asset_id)
    except Asset.DoesNotExist:
        logger.warning(
            "Asset %s not found - likely deleted after task was queued. Skipping.",
            asset_id,
        )
        return None
    if asset.preprocess_stage == "CANCELLED":
        logger.info("Preprocessing for asset %s was cancelled, skipping", asset_id)
        return None
    return asset


def _is_canonical_image_file(path: Path) -> bool:
    """True when ``path`` already lives in the canonical image store.

    The encode step is re-entrant: on a job retry the FULL rendition already
    points at ``IMAGES_DIR/<asset>/<stem>.png``, and re-encoding it would be
    pointless work. A *staged* upload (``TMP_DIR/uploads``) is never canonical,
    even when it is itself a PNG.
    """

    try:
        Path(path).resolve().relative_to(IMAGES_DIR.resolve())
    except (ValueError, OSError):
        return False
    return True


def prepare_asset_renditions_task(asset_id: str) -> None:
    # This entry point is run by the upload worker. Keep Zarr/Blosc's nested
    # thread pools inside the machine budget so the request server and viewer
    # remain interactive during a large encode.
    with bounded_ngff_build_resources():
        prepare_asset_renditions(asset_id)


def _canonical_png_target(asset: Asset) -> Path:
    original_stem = (asset.original_filename or asset.display_name or str(asset.id)).split(".")[0]
    return IMAGES_DIR / str(asset.id) / f"{original_stem}.png"


def _fail(asset: Asset, message: str) -> None:
    """Conclude this import as failed, before anything else can be told otherwise."""

    record_terminal_failure(asset, message)
    set_stage(asset, "FAILED", progress=0.0, error=message)


def _named_stage_failure(stage: str, exc: BaseException) -> ValueError:
    """A stage-naming ValueError that keeps the original sentence verbatim."""

    if isinstance(exc, MemoryError):
        return ValueError(f"Out of memory: Image is too large to process. {exc}")
    return ValueError(f"{stage}: {exc}")


def prepare_asset_renditions(asset_id: str) -> None:
    """Decode the staged upload once; build the pyramid and the canonical PNG from it.

    This replaces the encode -> NGFF pair for 2D imports. The two steps used to
    be strictly serial and each began by decoding the whole image: the encoder
    read the source and wrote a PNG that existed only so the NGFF builder could
    immediately read it back. On the 475 MP test image that intermediate cost
    10.4 s to write and 10.0 s to read, for an artifact nothing else in the
    import path wants.

    Now there is one decode. The pyramid is written from the resulting array
    and the canonical PNG is encoded from the same array on a worker thread
    (Pillow's zlib encoder releases the GIL), so the two run side by side
    instead of end to end.

    The pyramid is registered as soon as it is complete and valid, which is
    what makes the asset openable -- the PNG may still be landing. That is
    honest: ``ngff_ready`` promises a viewable pyramid and nothing else, the
    asset stays in the ``ENCODING`` stage until both artifacts exist, and every
    consumer of the FULL rendition reads the *same pixels* through that window
    as it will after ``DONE`` (see ``task_utils``, which resolves every uint8
    read against level 0 first). The pipeline only reports ``DONE`` once this
    function returns.

    If anything fails, the asset is left closed rather than half-open. That is
    now a property rather than a cleanup step: the failure handler calls
    :func:`~quantem.assets.pyramid_authority.record_attempt_failure`, which in
    one transaction clears the published pointer *and* bumps the attempt token
    -- so an NGFF job that was enqueued a second earlier and is still building
    cannot publish over the failure afterwards, and there is no instant at
    which a guard has to fire.
    """

    asset = _get_asset_or_none(asset_id)
    if asset is None:
        return
    # A new attempt, and therefore a new fence. Everything already running for
    # the previous token is now unable to publish; the previous generation
    # keeps serving until this attempt has one of its own.
    begin_attempt(asset)
    set_stage(asset, "ENCODING", progress=0.0, error="")

    try:
        openable = get_asset_openable(asset)
    except Exception as exc:
        _fail(asset, f"Missing full image rendition: {exc}")
        return

    if openable.rendition.type != Rendition.TYPE_FULL:
        _fail(asset, "Upload preprocessing requires a FULL rendition.")
        return

    source_path = openable.path
    if not source_path.exists():
        _fail(asset, "Missing source file for upload.")
        return

    source_is_canonical_png = source_path.suffix.lower() == ".png" and _is_canonical_image_file(
        source_path
    )
    target_png_path = source_path if source_is_canonical_png else _canonical_png_target(asset)

    # Everything from here on can leave the asset in a state the UI has to be
    # able to describe, so it all unwinds through one handler. That includes
    # the disk-space refusal: it mutates nothing itself, but a *retry* of an
    # asset that was openable from an earlier run would otherwise go FAILED
    # with the old pyramid still advertising it as ready.
    try:
        if not source_is_canonical_png:
            disk_usage = shutil.disk_usage(str(IMAGES_DIR))
            estimated_bytes = int(openable.width * openable.height)
            min_free = int(os.environ.get("ENCODE_MIN_FREE_BYTES", str(512 * 1024 * 1024)))
            required = int(estimated_bytes * 1.2)
            if disk_usage.free < max(min_free, required):
                raise ValueError("Insufficient free disk space")

        metadata = {
            "width": int(openable.width),
            "height": int(openable.height),
            "channels": int(openable.channels),
            "bit_depth": int(openable.bit_depth),
        }

        decode_start = time.time()
        # The one decode in the tree. Dispatches on magic bytes, takes band 0 in
        # every container, and refuses complex/negative-signed data by name
        # instead of clipping it silently.
        canonical = decode_canonical_plane(source_path, declared=metadata)
        trim_on_this_pass = should_trim_initial_import(
            source_is_canonical_png=source_is_canonical_png,
            rendition_metadata=openable.rendition.metadata,
        )
        trimmed_plane, border_trim = (
            trim_black_or_white_border(canonical.array)
            if trim_on_this_pass
            else (canonical.array, None)
        )
        if border_trim is not None:
            canonical = replace(canonical, array=trimmed_plane)
            # AssetOpenable reads these model objects dynamically. Update the
            # in-memory geometry before building so the pyramid validates the
            # cropped plane; persist the same dimensions only after success.
            openable.rendition.stored_width = canonical.width
            openable.rendition.stored_height = canonical.height
            asset.logical_width = canonical.width
            asset.logical_height = canonical.height
        plane = canonical.array
        logger.info(
            "Asset %s: decoded %s in %.2fs (shape=%s, %s)",
            asset_id,
            source_path.name,
            time.time() - decode_start,
            plane.shape,
            canonical.provenance,
        )
        set_stage(asset, "ENCODING", progress=5.0, error="")

        png_writer = None
        if not source_is_canonical_png or border_trim is not None:
            png_writer = _BackgroundCall(
                lambda: _write_canonical_png(plane, target_png_path),
                name=f"canonical-png-{asset_id}",
            )
            png_writer.start()

        try:
            last_update = 0.0

            def pyramid_progress(fraction: float, message: str) -> None:
                del message
                nonlocal last_update
                now = time.time()
                if now - last_update < 1.0 and fraction < 1.0:
                    return
                last_update = now
                set_stage(asset, "ENCODING", progress=5.0 + 45.0 * fraction, error="")

            try:
                generation_root = build_and_publish(
                    openable,
                    canonical,
                    progress_callback=pyramid_progress,
                )
            except Exception as exc:
                raise _named_stage_failure("Error building the image pyramid", exc) from exc
            logger.info(
                "Asset %s: published pyramid %s; asset is openable",
                asset_id,
                generation_root.name,
            )
        except BaseException:
            # The pyramid failed. Still wait for the PNG thread -- leaving it
            # writing into the image store after the job has given up is how you
            # get a half-written canonical PNG that the retry then trusts -- but
            # report the pyramid's failure, not the thread's. A raise from a
            # `finally` would silently replace the real cause.
            if png_writer is not None:
                try:
                    png_writer.join()
                except BaseException:
                    logger.warning(
                        "Asset %s: the canonical PNG also failed while unwinding",
                        asset_id,
                        exc_info=True,
                    )
            raise
        if png_writer is not None:
            png_writer.join()

        relative_path = normalize_stored_path_value(target_png_path, relative_to=IMAGES_DIR.parent)
        rendition_metadata = dict(openable.rendition.metadata or {})
        rendition_metadata["upload_state"] = "canonical"
        if border_trim is not None:
            rendition_metadata["border_trim"] = border_trim.as_metadata()
        # The canonical file becomes authoritative in one database commit.
        # Keep the staged source until after that commit: if either model write
        # fails, the retry still has the original bytes instead of a rendition
        # row pointing at a file this attempt already deleted.
        with transaction.atomic():
            asset.channels = 1
            asset.bit_depth = 8
            asset.logical_width = canonical.width
            asset.logical_height = canonical.height
            asset.save(
                update_fields=[
                    "channels",
                    "bit_depth",
                    "logical_width",
                    "logical_height",
                    "updated_at",
                ]
            )
            Rendition.objects.filter(id=openable.rendition.id).update(
                storage_root="DATA_DIR",
                stored_path=relative_path,
                path_exists=target_png_path.exists(),
                is_directory=False,
                stored_width=canonical.width,
                stored_height=canonical.height,
                stored_channels=1,
                stored_bit_depth=8,
                metadata=rendition_metadata,
            )

        if png_writer is not None and source_path != target_png_path and source_path.exists():
            try:
                source_path.unlink()
            except OSError:
                # The rendition no longer references the staging file. A
                # Windows scanner may briefly hold it open; the staging sweep
                # can reclaim this unreferenced copy later.
                logger.warning("Could not remove staged upload %s", source_path)
    except BaseException as exc:
        # One transaction does three things, and the order does not matter
        # because they commit together: the published pointer is cleared (the
        # whole of withdrawal, one column write, cannot fail), the attempt
        # token is bumped (so a build already running for the previous token
        # can never publish afterwards), and the *real* cause is recorded on
        # ``failure_detail`` -- a field the job layer's retry note does not
        # write, which is why a lease conflict can no longer replace it.
        record_attempt_failure(asset, f"{type(exc).__name__}: {exc}")
        raise
    record_import_success(asset)
    set_stage(asset, "ENCODING", progress=55.0, error="")


def _write_canonical_png(plane, target_png_path: Path) -> Path:
    """``save_plane_as_canonical_png``, with the stage named on the way out."""

    try:
        return save_plane_as_canonical_png(plane, target_png_path)
    except Exception as exc:
        raise _named_stage_failure("Error writing the canonical PNG", exc) from exc


class _BackgroundCall:
    """Run one callable on a thread and re-raise whatever it raised on join.

    Deliberately not a ThreadPoolExecutor: there is exactly one job, and a
    failure in it has to reach the caller's ``finally`` so the import fails
    rather than reporting success with a missing canonical PNG.
    """

    def __init__(self, operation, *, name: str):
        self._operation = operation
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._error: BaseException | None = None

    def _run(self) -> None:
        try:
            self._operation()
        except BaseException as exc:  # noqa: BLE001 - re-raised on join
            self._error = exc

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join()
        if self._error is not None:
            raise self._error


def ensure_ngff_for_asset_task(asset_id: str) -> dict:
    """Build this asset's pyramid if it has none, and say what happened.

    **A refused or superseded build is a successful no-op, not a failure.** A
    job that fails invokes ``jobs.failure_reconcile``, which marks the asset
    FAILED and overwrites ``preprocess_error`` -- so before this, a redundant
    NGFF job losing a race told the user about a storage lease instead of the
    real WinError 5 that had actually broken their import. There is nothing
    wrong when a build discovers it is stale; the correct report is "nothing to
    do", and the reconciler is never reached.

    The lazy rebuild itself is unconditional in the sense that matters: it asks
    the authority for a ticket rather than testing the filesystem for a
    complete-looking store, so the ``suffix == ".png"`` test that let round 3
    rebuild a *staged 16-bit upload* as though it were the canonical PNG -- and
    publish an all-white pyramid over a FAILED asset -- does not exist here.
    """

    asset = _get_asset_or_none(asset_id)
    if asset is None:
        return {"asset_id": asset_id, "status": "noop", "reason": "asset is gone or cancelled"}

    from .pyramid_authority import Intent, PublishedPyramid, resolve_pyramid

    resolved = resolve_pyramid(asset, intent=Intent.SERVE)
    if isinstance(resolved, PublishedPyramid):
        return {
            "asset_id": asset_id,
            "status": "noop",
            "generation": resolved.generation_id,
        }

    logger.info("Asset %s: NGFF stage started", asset_id)
    openable = get_asset_openable(asset)
    try:
        with bounded_ngff_build_resources():
            if openable.has_stored_z_stack:
                generation_root = build_and_publish(openable, volume_source=openable.path)
            else:
                canonical = decode_canonical_plane(
                    openable.path,
                    declared={
                        "width": int(openable.width),
                        "height": int(openable.height),
                        "channels": int(openable.channels),
                        "bit_depth": int(openable.bit_depth),
                    },
                )
                generation_root = build_and_publish(openable, canonical)
    except PyramidBuildRefused as refused:
        logger.info(
            "Asset %s: NGFF build declined (%s); nothing was published and nothing failed.",
            asset_id,
            refused.unavailable.reason.value,
        )
        return {
            "asset_id": asset_id,
            "status": "refused",
            "reason": refused.unavailable.reason.value,
            "detail": refused.unavailable.detail,
        }
    logger.info("Asset %s: published pyramid %s", asset_id, generation_root.name)
    return {"asset_id": asset_id, "status": "published", "generation": generation_root.name}


def ensure_roi_for_asset_task(asset_id: str) -> ImageROI | None:
    asset = _get_asset_or_none(asset_id)
    if asset is None:
        return None

    existing_roi = get_active_roi_for_asset(asset)
    if existing_roi:
        return existing_roi

    openable = get_asset_openable(asset)
    roi_size = ROI_SIZE_DEFAULT
    if openable.width * openable.height >= ROI_MIN_IMAGE_SIZE**2:
        roi_result = select_roi_for_image(
            image=openable,
            roi_size=roi_size,
        )
        return create_roi_image_from_image(
            openable,
            x=roi_result.x,
            y=roi_result.y,
            width=roi_result.width,
            height=roi_result.height,
            source="AUTO",
        )
    return None
