"""Keep the probability map a run produced, so guided fine-tuning can reach it.

Why this lives here and not in the segmenter
--------------------------------------------
:attr:`quantem.inference.segmenter.DinoOrganelleSegmenter.persist_probability_maps`
is ``False`` and stays that way. That flag answers one question inside
:func:`quantem.seg_core.db.inference.run_inference_for_segmentation`: *may a
stored map be replayed instead of running the model?* For these models the
answer has to be no. The stored map is uint8 at native scale, while the
foreground decision is made on the model's own resampled grid and only the
resulting binary mask is brought back (see :mod:`quantem.inference.resample`).
Replaying a stored map would re-threshold a quantised array on the wrong grid
and quietly change every candidate.

*Writing* the map is a different question, and the answer to it is yes. Without
it, guided fine-tuning was unreachable: threshold calibration
(``mode="threshold_only"``) sweeps a threshold against what the model currently
predicts inside the user's completed ROI, so
:func:`quantem.segmentation.services.adapt.collect_crops` refused with "No
probability map covers the completed area. Run the model on this image first" —
and running the model did not change that, because nothing ever wrote one.
``GET .../probability-maps/`` returned ``[]`` forever.

So the run writes the map; nothing reads it back as an inference cache.

Storage policy
--------------
* **8-bit, not float32.** :func:`quantem.seg_core.db.prob_maps.save_probability_map`
  quantises to uint8 and deflates it into a PNG. A 4096x4096 map costs single-digit
  MB on disk where the float32 array it came from was 64 MB. The quantisation is
  1/255 of the probability range, which is far below the granularity any threshold
  sweep resolves.
* **One file per (segmentation, model).** A full-image re-run overwrites the same
  path, and the superseded ``ProbabilityMap`` rows pointing at it are deleted, so
  N runs cost one file and one row rather than N.
* **An ROI run stores only the window it ran**, plus the full-image composite that
  :func:`save_probability_map` already maintains. The window is recorded in
  ``metadata["roi"]`` so :func:`~quantem.segmentation.services.adapt.collect_crops`
  can place it at the right offset instead of falling back to the composite (which
  reads as confident background everywhere the model never ran).
* **A size ceiling**, :data:`MAX_MEGAPIXELS_ENV` (default
  :data:`DEFAULT_MAX_MEGAPIXELS`). Above it a full-image map is skipped and the
  reason is reported on the job, because a 500 MP uint8 PNG is a real cost on a
  desktop and an ROI run gives calibration everything it needs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import numpy as np

from quantem.assets.models import ImageROI
from quantem.seg_core.db.prob_maps import get_prob_map_file_path, save_probability_map
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

    for model_name in segmenter.get_dl_model_names():
        data = prob_maps.get(model_name)
        if data is None:
            continue
        array = np.asarray(data)
        if array.ndim != 2 or array.size == 0:
            continue

        if roi is None and ceiling_px and array.size > ceiling_px:
            message = (
                f"Probability map not stored: this image is "
                f"{array.size / 1e6:.0f} MP, above the {max_megapixels():.0f} MP "
                f"ceiling ({MAX_MEGAPIXELS_ENV}). Guided fine-tuning needs a map, "
                "so run the model over an ROI that covers the area you annotated."
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
