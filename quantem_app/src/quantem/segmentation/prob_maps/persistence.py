"""Keep the probability map a run produced: the run's own record, and the dial.

What the stored map is
----------------------
It is the **same bytes the run thresholded**. Inference carries the probability
field back to the image's own pixel coordinates and quantises it to uint8
*before* deciding anything (:class:`quantem.inference.resample.NativeProbabilityMap`),
and the foreground decision is taken on that array. Writing it here is therefore
storing the authority, not exporting a picture of it.

This supersedes an earlier note in this module, which refused replay on the
record: *"The stored map is uint8 at native scale, while the foreground decision
is made on the model's own resampled grid and only the resulting binary mask is
brought back. Replaying a stored map would re-threshold a quantised array on the
wrong grid and quietly change every candidate."* That was true of the old
ordering and is the reason the ordering was inverted. There is no longer a wrong
grid: the map, the threshold, the closing, the labeling and the area filter are
all in native coordinates, so re-thresholding the stored map at ``T`` is the
same arithmetic as the threshold step of a fresh run at ``T``, on the same
bytes, in the same function
(:func:`quantem.inference.resample.binarize_quantized`). Replay is exact by
construction rather than by careful matching, and
:func:`quantem.seg_core.db.inference.replay_stored_probability_map` is the
operation that uses it.

Two things that follow, and are worth stating rather than discovering:

* **Replaying is not re-running.**
  :attr:`~quantem.inference.segmenter.DinoOrganelleSegmenter.persist_probability_maps`
  stays ``False``, so "run the model" always runs the model. A stored map is
  never silently substituted for one. Moving the threshold is a different verb
  and takes the replay path.
* **A re-run is required after fine-tuning, when the map is gone, or when the
  map predates this pipeline.** The model changed, the bytes did, or the bytes
  were written under a different set of conventions and cannot be re-decided
  under these ones. The third case is checked on read rather than assumed away:
  see :func:`replay_provenance_problem` and :data:`LEGACY_MAP_MESSAGE`.

*Writing* the map was already necessary for guided fine-tuning, and remains so.
Without it, threshold calibration (``mode="threshold_only"``) was unreachable:
it sweeps a threshold against what the model currently predicts inside the
user's completed ROI, so
:func:`quantem.segmentation.services.adapt.collect_crops` refused with "No
probability map covers the completed area. Run the model on this image first" --
and running the model did not change that, because nothing ever wrote one.

Storage policy
--------------
* **8-bit, and that is the working precision, not a compression of it.**
  :func:`quantem.seg_core.db.prob_maps.save_probability_map` writes the uint8
  array through unchanged (or quantises a float caller's map with the same
  round-to-nearest rule). 255 levels is a fine dial: sweeping every level
  produces 255 distinct foreground areas on a real image, and the quantisation
  costs at most one object anywhere on the dial -- less than one step of the
  dial's own resolution. A 4096x4096 map costs single-digit MB on disk.
* **One file per (segmentation, model).** A full-image re-run overwrites the same
  path, and the superseded ``ProbabilityMap`` rows pointing at it are deleted, so
  N runs cost one file and one row rather than N.
* **An ROI run stores only the window it ran**, plus the full-image composite that
  :func:`save_probability_map` already maintains. The window is recorded in
  ``metadata["roi"]`` so :func:`~quantem.segmentation.services.adapt.collect_crops`
  can place it at the right offset instead of falling back to the composite (which
  reads as confident background everywhere the model never ran).
* **A size ceiling**, :data:`MAX_MEGAPIXELS_ENV` (default
  :data:`DEFAULT_MAX_MEGAPIXELS`). Above it a full-image map is skipped. Note
  what that now costs: the run itself is unaffected (it holds the map in memory
  and thresholds it there), but with nothing on disk the threshold cannot be
  moved afterwards without re-running the model, and guided fine-tuning still
  needs an ROI run. The ceiling was chosen when the map was an optional artifact
  for fine-tuning; at 1 byte/px a 2-3 GB image needs a 2-3 GB map, which is the
  size question this number now really asks. Flagged in the R11 report rather
  than changed here, because raising it is a storage decision and lowering the
  stored map's resolution would change what the dial can do.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from quantem.assets.models import ImageROI
from quantem.core.config import STORAGE_DIR
from quantem.inference.resample import (
    QUANTIZATION_ID,
    NativeProbabilityMap,
    quantize_probability,
)
from quantem.seg_core.db.inference import StoredMapUnavailable
from quantem.seg_core.db.prob_maps import (
    get_prob_map_file_path,
    load_prob_map_from_path,
    prob_map_file_exists,
    save_probability_map,
)
from quantem.segmentation.models import ImageSegmentation, ProbabilityMap

logger = logging.getLogger(__name__)

#: Full-image maps larger than this are not written. Chosen so every ordinary EM
#: field fits (a 22k x 22k image is 484 MP) while a stitched gigapixel montage
#: does not silently consume hundreds of MB per organelle.
DEFAULT_MAX_MEGAPIXELS = 512.0

#: Override for :data:`DEFAULT_MAX_MEGAPIXELS`. ``0`` disables the ceiling.
MAX_MEGAPIXELS_ENV = "QUANTEM_PROB_MAP_MAX_MEGAPIXELS"

#: ``metadata["run_scope"]`` values, so a reader can tell what a map covers
#: without reconstructing it from the file path.
SCOPE_FULL = "full"
SCOPE_ROI = "roi"


def max_megapixels() -> float:
    """Full-image ceiling in megapixels; ``0`` (or a bad value) means no ceiling."""
    raw = os.environ.get(MAX_MEGAPIXELS_ENV)
    if raw is None:
        return DEFAULT_MAX_MEGAPIXELS
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_MEGAPIXELS
    return max(0.0, value)


def _scalar_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """JSON-safe copy: scalars kept, ``None`` dropped, anything else stringified.

    Unlike the normaliser in :mod:`quantem.seg_core.db.inference` this keeps a
    nested dict of scalars, because ``metadata["roi"]`` has to survive as a dict
    for the crop reader to use its offsets.
    """
    out: dict[str, object] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, dict):
            nested = _scalar_metadata(value)
            if nested:
                out[key] = nested
        else:
            out[key] = str(value)
    return out


def _prune_superseded(kept: ProbabilityMap) -> int:
    """Delete earlier rows for the same file, so re-runs do not accumulate rows.

    The file itself was overwritten in place, so every other row with this
    ``(segmentation, name, file_path)`` describes bytes that no longer exist.
    Composites are left alone: they are maintained by
    :func:`save_probability_map` under their own path and lifecycle.
    """
    stale = ProbabilityMap.objects.filter(
        segmentation_id=kept.segmentation_id,
        name=kept.name,
        file_path=kept.file_path,
    ).exclude(id=kept.id)
    deleted, _ = stale.delete()
    return int(deleted)


def persist_run_probability_maps(
    *,
    segmentation: ImageSegmentation,
    segmenter,
    prob_maps: dict[str, np.ndarray],
    roi: ImageROI | None = None,
    on_detail: Callable[[str], None] | None = None,
) -> list[ProbabilityMap]:
    """Store the maps a completed run produced. Never raises.

    Args:
        segmentation: the segmentation that was run.
        segmenter: the segmenter that ran, for its prefix, generated flag and
            per-model provenance metadata.
        prob_maps: ``InferenceResult.prob_maps`` — model name -> native-scale
            float array in ``[0, 1]``.
        roi: the ROI the run was scoped to, or ``None`` for a full-image run.
        on_detail: optional job-log callback, used to say when a map was skipped.

    Returns:
        The ``ProbabilityMap`` rows written (empty when nothing was stored).
    """
    report = on_detail or (lambda _message: None)

    # A segmenter that persists its own maps is already served by
    # run_inference_for_segmentation; writing again would duplicate the row.
    if bool(getattr(segmenter, "persist_probability_maps", True)):
        return []

    prefix = str(getattr(segmenter, "prob_map_prefix", "") or "")
    generated_flag = str(getattr(segmenter, "generated_flag", "") or "")
    if not prefix or not generated_flag:
        return []

    roi_id = str(roi.id) if roi is not None else None
    ceiling_px = max_megapixels() * 1e6
    written: list[ProbabilityMap] = []

    model_names = list(segmenter.get_dl_model_names())
    # The array the run actually thresholded, when the segmenter kept one. Its
    # dequantised float in ``prob_maps`` requantises to the same bytes -- there
    # is a test -- but storing the bytes themselves means the file is the
    # authority by construction rather than by a numeric coincidence that a
    # future change to the store could break silently. Only for a
    # single-output segmenter: with several maps there is no way to tell which
    # one the stored array belongs to.
    authority = getattr(segmenter, "native_probability_map", None)
    if len(model_names) != 1:
        authority = None

    for model_name in model_names:
        data = prob_maps.get(model_name)
        if data is None:
            continue
        array = np.asarray(data)
        if array.ndim != 2 or array.size == 0:
            continue
        if authority is not None and authority.shape == array.shape:
            array = authority.data

        if roi is None and ceiling_px and array.size > ceiling_px:
            message = (
                f"Probability map not stored: this image is "
                f"{array.size / 1e6:.0f} MP, above the {max_megapixels():.0f} MP "
                f"ceiling ({MAX_MEGAPIXELS_ENV}). The objects from this run are "
                "unaffected, but changing the threshold later will need a full "
                "re-run rather than being instant. Guided fine-tuning needs a "
                "map, so run the model over an ROI that covers the area you "
                "annotated."
            )
            logger.info(
                "Skipping probability map for segmentation %s (%s): %d px > %d px",
                segmentation.id,
                model_name,
                array.size,
                int(ceiling_px),
            )
            report(message)
            continue

        try:
            metadata: dict[str, object] = dict(
                segmenter.get_probability_map_metadata(model_name) or {}
            )
            metadata["run_scope"] = SCOPE_ROI if roi is not None else SCOPE_FULL
            if roi is not None:
                # The offset the crop reader needs; without it an ROI-sized map
                # is unusable and only the composite (which reads unrun area as
                # background) is left.
                metadata["roi"] = {
                    "x": int(roi.x),
                    "y": int(roi.y),
                    "width": int(roi.width),
                    "height": int(roi.height),
                }
            saved = save_probability_map(
                segmentation,
                model_name,
                array,
                prefix,
                generated_flag,
                roi_id,
                extra_metadata=_scalar_metadata(metadata),
            )
        except Exception:
            # The candidates are the run's product; losing the map costs guided
            # fine-tuning, not the segmentation, so say so and carry on.
            logger.warning(
                "Failed to store probability map for segmentation %s (%s)",
                segmentation.id,
                model_name,
                exc_info=True,
            )
            report(
                f"Probability map for {model_name} could not be stored; guided "
                "fine-tuning will ask you to run the model again."
            )
            continue

        _prune_superseded(saved)
        written.append(saved)
        path = get_prob_map_file_path(segmentation, model_name, prefix, roi_id)
        logger.info(
            "Stored %s probability map for segmentation %s at %s (%d px)",
            model_name,
            segmentation.id,
            path,
            array.size,
        )

    if written:
        scope = "ROI" if roi is not None else "full image"
        report(f"Stored {len(written)} probability map(s) for the {scope} run")
    return written


# --- Reading a stored map back, for replay ----------------------------------


@dataclass(frozen=True)
class StoredProbabilityMap:
    """A probability map read back off disk, ready to be thresholded again.

    ``native`` carries the bytes plus the provenance recorded when they were
    written -- which interpolator brought the field to these coordinates, and
    how it was quantised -- so a replay can record the same facts about itself
    as the run did.
    """

    model_name: str
    native: NativeProbabilityMap
    roi_id: str | None
    roi_window: dict[str, int] | None
    metadata: dict[str, object]

    @property
    def shape(self) -> tuple[int, int]:
        return self.native.shape


#: ``metadata["thresholded_on"]`` for a map whose run took its foreground
#: decision on the stored bytes themselves. Anything else means the decision was
#: taken somewhere the dial cannot reproduce.
THRESHOLDED_ON_STORED_MAP = "stored_native_uint8"

#: What a user is told when the map on disk was written by an older build.
#:
#: It has to say that the objects need re-running, because that is the only way
#: out: the bytes are real but the conventions behind them are not recorded, and
#: re-thresholding them under today's conventions would move object boundaries
#: and object *counts* by around a percent while presenting the result as the
#: same computation a fresh run performs. Under the two quantisation rules that
#: have existed here, about 18-40% of stored bytes hold a different level.
LEGACY_MAP_MESSAGE = (
    "The stored result for this image was written by an earlier version of "
    "QuantEM, which recorded probabilities differently. Reading it at a new "
    "threshold would give objects that do not match a run at the same "
    "threshold, so the model has to run on this image again before the "
    "threshold can be moved."
)


def replay_provenance_problem(metadata: dict[str, object]) -> str | None:
    """Why these stored bytes may not be re-thresholded, or ``None`` if they may.

    The five markers below are written **together**, by
    :meth:`quantem.inference.resample.NativeProbabilityMap.provenance`, every
    time a run stores a map. A map carrying some of them and not others did not
    come from that writer, so the honest reading of a missing marker is "unknown
    convention", not "the current convention".

    That distinction is the whole point. :meth:`NativeProbabilityMap.from_stored`
    *defaults* a missing interpolator to no-resample, a missing factor to 1.0 and
    a missing quantiser to the current one, which is the right thing for a caller
    that knows what it is holding and exactly the wrong thing for a file of
    unknown origin: the replay would not only re-decide the pixels, it would then
    describe itself in provenance as the pipeline that never touched it.

    Returns a short technical phrase naming the first marker that fails, for the
    log. The sentence a user reads is :data:`LEGACY_MAP_MESSAGE`, which is the
    same whichever marker it was: from where the user sits there is one fact,
    and it is that this map has to be made again.

    Note that the ``ProbabilityMap`` provenance columns are deliberately *not*
    consulted here. ``quantisation`` defaults to ``"uint8"`` and nothing on the
    write path sets it, so a map stored a minute ago carries the same column
    value as one from the previous build; only ``metadata`` distinguishes them.
    """
    if metadata.get("native_coordinates") is not True:
        return "no record that the map is in the image's own pixel coordinates"
    if metadata.get("thresholded_on") != THRESHOLDED_ON_STORED_MAP:
        return (
            "the run that wrote it did not record taking its foreground "
            f"decision on the stored map ({metadata.get('thresholded_on')!r})"
        )
    quantization = metadata.get("quantization")
    if quantization != QUANTIZATION_ID:
        return (
            f"quantisation recorded as {quantization!r}, not {QUANTIZATION_ID!r}"
        )
    interpolation = metadata.get("resample_interpolation")
    if not isinstance(interpolation, str) or not interpolation.strip():
        return "no interpolator recorded for the crossing back to native pixels"
    back_factor = metadata.get("resample_back_factor")
    if isinstance(back_factor, bool) or not isinstance(back_factor, (int, float)):
        return f"no usable resample factor recorded ({back_factor!r})"
    return None


def stored_map_is_replayable(metadata: dict[str, object] | None) -> bool:
    """``True`` when a stored map's own record says today's rules produced it.

    The predicate behind :func:`replay_provenance_problem`, exported because
    every reader of a stored map needs it, not only the dial. In particular
    :func:`quantem.segmentation.services.adapt.collect_crops` selects maps by
    file geometry alone (``_prob_sources``) and would happily fit a threshold
    against bytes from the previous build.
    """
    if not isinstance(metadata, dict):
        return False
    return replay_provenance_problem(metadata) is None


#: The dial can move: a map is stored and today's rules produced it.
REPLAY_READY = "ready"
#: Nothing was ever stored here, or it has been reclaimed.
REPLAY_NOT_STORED = "not_stored"
#: Bytes are on disk, but their own record does not say how they were made.
REPLAY_FROM_OLDER_BUILD = "from_older_build"

#: What a user is told when no map was ever written for this image.
#:
#: Deliberately a different sentence from :data:`LEGACY_MAP_MESSAGE`, and the
#: difference is the point. Both end in "run the model again", but they are not
#: the same situation and do not have the same future: this one is fixed for
#: good by running once, and the other says the stored result is an old one that
#: will keep being refused until it is replaced. A user told only "run it again"
#: cannot tell which of those they are in.
NO_STORED_MAP_MESSAGE = (
    "No stored result is kept for this image, so the include level cannot be "
    "moved without running the model again. Running it once saves one, and the "
    "level can be moved freely from then on."
)


@dataclass(frozen=True)
class StoredMapReadiness:
    """Whether the include level can be moved here, and what to say if not.

    Answered **without decoding the map**. The two cheap facts -- is there a
    file, and does its row record today's conventions -- are exactly the two
    that decide it, and a stored map for a 484 MP image is not something to read
    off disk in order to grey out a control.

    ``detail`` is empty when :attr:`ready`; otherwise it is the sentence for
    *this* case, never a generic one.
    """

    status: str
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == REPLAY_READY


def stored_map_readiness(
    *,
    segmentation: ImageSegmentation,
    segmenter,
    model_name: str,
    roi: ImageROI | None = None,
) -> StoredMapReadiness:
    """Can the include level be moved for this (segmentation, model)?

    The same two checks :func:`load_stored_native_map` makes before it reads
    anything, in the same order, so a "yes" here and a failure there can only be
    a genuine race -- the file reclaimed between the question and the answer --
    rather than two implementations of one rule that drifted apart.

    Exists so the endpoint that queues a dial move can refuse **before** queuing
    it. A job that is certain to fail is worse than a refusal: the user waits,
    watches a task appear and go red, and reads the reason a minute after the
    moment they could have acted on it.
    """
    prefix = str(getattr(segmenter, "prob_map_prefix", "") or "")
    if not prefix:
        return StoredMapReadiness(REPLAY_NOT_STORED, NO_STORED_MAP_MESSAGE)

    roi_id = str(roi.id) if roi is not None else None
    if not prob_map_file_exists(segmentation, model_name, prefix, roi_id):
        return StoredMapReadiness(REPLAY_NOT_STORED, NO_STORED_MAP_MESSAGE)

    file_path = get_prob_map_file_path(segmentation, model_name, prefix, roi_id)
    relative_path = str(file_path.relative_to(STORAGE_DIR)).replace("\\", "/")
    record = (
        ProbabilityMap.objects.filter(
            segmentation=segmentation,
            name=f"{prefix.upper()}_{model_name}",
            file_path=relative_path,
        )
        .order_by("-updated_at")
        .first()
    )
    metadata: dict[str, object] = {}
    if record is not None and isinstance(record.metadata, dict):
        metadata = dict(record.metadata)

    if replay_provenance_problem(metadata) is not None:
        return StoredMapReadiness(REPLAY_FROM_OLDER_BUILD, LEGACY_MAP_MESSAGE)
    return StoredMapReadiness(REPLAY_READY)


def _roi_window(metadata: dict[str, object]) -> dict[str, int] | None:
    window = metadata.get("roi")
    if not isinstance(window, dict):
        return None
    try:
        return {key: int(window[key]) for key in ("x", "y", "width", "height")}
    except (KeyError, TypeError, ValueError):
        return None


def load_stored_native_map(
    *,
    segmentation: ImageSegmentation,
    segmenter,
    model_name: str,
    roi: ImageROI | None = None,
) -> StoredProbabilityMap | None:
    """The stored uint8 map for one (segmentation, model), or ``None``.

    ``None`` means there is nothing to replay -- the run predates map storage,
    the image was above the size ceiling, or the file has been reclaimed -- and
    the caller must re-run the model rather than invent a map.

    :class:`~quantem.seg_core.db.inference.StoredMapUnavailable` means something
    different and is raised rather than returned: there *are* bytes on disk, but
    their own record does not say they were produced by this pipeline, so they
    are refused instead of reinterpreted (:func:`replay_provenance_problem`).
    Both outcomes end in "run the model again"; keeping them apart is what lets
    the sentence on screen say which of the two happened, and stops a legacy map
    being quietly re-decided under conventions it was never written under.

    **Why the bytes are recovered by requantising, not by a second decoder.**
    The stored PNG is uint8 and :func:`load_prob_map_from_path` reads it as
    ``uint8 / 255``; running that back through
    :func:`~quantem.inference.resample.quantize_probability` returns the
    original bytes exactly, for all 256 levels (there is a test). Opening the
    file again here would add a fifth place in the tree that turns a file into
    pixels, which
    ``quantem/assets/tests/test_ngff_decode_chokepoint.py`` exists to prevent --
    and would buy nothing, since the round trip is lossless.
    """
    prefix = str(getattr(segmenter, "prob_map_prefix", "") or "")
    if not prefix:
        return None
    roi_id = str(roi.id) if roi is not None else None

    # Both this and `load_prob_map_from_path` below reject a stale composite
    # standing in for a full-image map, which is exactly the artifact that must
    # never be thresholded: it reads as confident background everywhere the
    # model did not run. Asked first so the provenance check below can refuse a
    # map without decoding it -- a legacy map is the ordinary case on an
    # upgraded install, and it can be hundreds of megapixels.
    if not prob_map_file_exists(segmentation, model_name, prefix, roi_id):
        return None

    # Matched on the file path, not just the name: an ROI run also maintains a
    # full-image *composite* row under the same name, and reading provenance off
    # that would describe a different artifact than the bytes just loaded.
    file_path = get_prob_map_file_path(segmentation, model_name, prefix, roi_id)
    relative_path = str(file_path.relative_to(STORAGE_DIR)).replace("\\", "/")
    record = (
        ProbabilityMap.objects.filter(
            segmentation=segmentation,
            name=f"{prefix.upper()}_{model_name}",
            file_path=relative_path,
        )
        .order_by("-updated_at")
        .first()
    )
    metadata: dict[str, object] = {}
    if record is not None and isinstance(record.metadata, dict):
        metadata = dict(record.metadata)

    # A file with no row at all is the same situation as a row with no markers:
    # bytes whose conventions nobody wrote down. Refused, not assumed.
    problem = replay_provenance_problem(metadata)
    if problem is not None:
        logger.info(
            "Refusing to reuse the stored probability map for segmentation %s "
            "(%s) at %s: %s",
            segmentation.id,
            model_name,
            relative_path,
            problem,
        )
        raise StoredMapUnavailable(LEGACY_MAP_MESSAGE)

    data = load_prob_map_from_path(segmentation, model_name, prefix, roi_id)
    if data is None:  # deleted between the existence check and the read
        return None

    native = NativeProbabilityMap.from_stored(
        quantize_probability(data),
        interpolation=metadata.get("resample_interpolation"),
        back_factor=metadata.get("resample_back_factor"),
        quantization=metadata.get("quantization"),
    )
    return StoredProbabilityMap(
        model_name=model_name,
        native=native,
        roi_id=roi_id,
        roi_window=_roi_window(metadata),
        metadata=metadata,
    )
