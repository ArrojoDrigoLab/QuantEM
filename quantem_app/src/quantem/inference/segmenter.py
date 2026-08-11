"""QuantEM / OmniEM organelle segmenters.

One segmenter class per organelle, registered as ``dino_<organelle>`` by
:func:`quantem.seg_core.registry.register_default_segmenters`. The model family
(``quantem`` / ``omniem``) is chosen per run from the ``source_model`` kwarg, so
``quantem:mito`` and ``omniem:mito`` share a class and differ only in the pack
they load.

This is the concrete implementation of the
:class:`~quantem.seg_core.base_segmenter.BaseSegmenter` contract. It is a thin
assembly of the modules around it:

* :mod:`.specs`        -- which model, what canonical nm/px, what tile size
* :mod:`.engine`       -- load the pack once, run sliding windows over a region
* :mod:`.resample`     -- native <-> canonical, and the stored native uint8 map
* :mod:`.postprocess`  -- threshold -> close -> fill -> label -> min area
* :func:`quantem.seg_core.extraction.build_segment_from_region` -- the shape
  measurements every downstream analysis reads

Where the foreground decision is made
-------------------------------------
The model predicts on its own resampled grid, and **nothing is decided there**.
The probability field is carried back to the image's own pixel coordinates
(:func:`quantem.inference.resample.probability_to_native`), quantised to uint8
(:func:`~quantem.inference.resample.quantize_probability`), and *that* array is
thresholded -- in native coordinates, by
:func:`~quantem.inference.resample.binarize_quantized` -- after which the
closing, the hole fill, the labeling and the minimum-area filter all run on
native pixels too.

The consequence worth stating plainly: a fresh run thresholds the **stored
uint8 map**, not the in-memory float it came from. That is what makes a later
threshold movement (the accuracy dial) arithmetically the same operation as the
run that preceded it instead of a near-miss. See :mod:`.resample` for the
measured effect of this ordering and for the interpolator it requires.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from skimage.measure import regionprops

from quantem.seg_core.base_segmenter import BaseSegmenter
from quantem.seg_core.extraction import build_segment_from_region
from quantem.seg_core.types import ExtractedSegment

from . import engine, postprocess, resample, tiling
from .specs import (
    DEFAULT_FAMILY,
    ORGANELLES,
    get_model_spec,
    parse_family,
    source_model_value,
)

logger = logging.getLogger(__name__)

#: Single DL output name; also the progress-stage key seen by
#: ``quantem.seg_core.db.inference``.
DL_MODEL_NAME = "DINO"

#: ``progress_stage`` values this module writes. They are a subset of
#: :data:`quantem.jobs.models.PROGRESS_STAGES`, kept as plain strings here so
#: that :mod:`quantem.inference` stays importable with no Django settings; a
#: test asserts the two lists agree.
STAGE_LOADING_MODEL = "loading_model"
STAGE_INFERENCE = "inference"
STAGE_EXTRACTING = "extracting"

#: ``progress_unit_label`` for sliding-window inference.
UNIT_TILE = "tile"


def _job_progress():
    """The running job's progress API, or None when nothing is watching.

    Inference is reached from a CLI call and from tests as well as from a
    worker, so this is allowed to find nothing. The import is lazy because
    :mod:`quantem.jobs.reporter` needs Django configured and this module must
    not.
    """
    try:
        from quantem.jobs import reporter as job_reporter  # noqa: PLC0415
    except Exception:  # no Django settings, or jobs not installed
        return None
    return job_reporter


def _report_stage(stage: str, **detail) -> None:
    progress = _job_progress()
    if progress is not None:
        progress.report_stage(stage, detail={k: v for k, v in detail.items() if v is not None})


class _NullTileScope:
    """:class:`~quantem.jobs.reporter.UnitProgressScope`'s surface, no database.

    What inference gets when it is not running under a job.
    """

    total = 0
    done = 0
    write_stats: dict = {}

    def set(self, done: int, *, total: int | None = None) -> None:
        self.done = int(done)
        if total is not None:
            self.total = int(total)

    def advance(self, count: int = 1) -> None:
        self.done += int(count)

    def finish(self) -> None:
        return

    def __enter__(self) -> _NullTileScope:
        return self

    def __exit__(self, *_exc) -> None:
        return


def _tile_scope(total: int, **detail):
    """A unit-progress scope over ``total`` tiles for the running job.

    Returns a no-op scope with the same surface when there is no job, so the
    inference path below carries no "is anyone watching" branch.
    """
    progress = _job_progress()
    if progress is None:
        return _NullTileScope()
    return progress.unit_scope(
        total=int(total),
        label=UNIT_TILE,
        stage=STAGE_INFERENCE,
        detail={k: v for k, v in detail.items() if v is not None},
    )


class DinoOrganelleSegmenter(BaseSegmenter):
    """Base foundation-model segmenter; subclasses set ``ORGANELLE``."""

    ORGANELLE: str = ""

    def __init__(
        self,
        *,
        source_model: str | None = None,
        device: str | None = None,
        fg_threshold: float | None = None,
        min_area: int | None = None,
        pixel_size_nm: float | None = None,
        overlap: float = tiling.DEFAULT_OVERLAP,
        **_ignored,
    ) -> None:
        if self.ORGANELLE not in ORGANELLES:
            raise ValueError(f"Invalid organelle: {self.ORGANELLE!r}")
        self._organelle = ORGANELLES[self.ORGANELLE]
        self._family = parse_family(source_model, default=DEFAULT_FAMILY)
        self._source_model = source_model or source_model_value(
            self._family, self.ORGANELLE
        )
        self._spec = get_model_spec(self._family, self.ORGANELLE)
        self._device = device
        self._fg_threshold = (
            float(fg_threshold) if fg_threshold is not None else self._spec.threshold
        )
        self._min_area = (
            int(min_area) if min_area is not None else self._organelle.default_min_area
        )
        # Supplied by quantem.segmentation.organelle_tasks from
        # ``Asset.pixel_size_nm``. When it is None the image is genuinely
        # uncalibrated: a model with a ``canonical_nm`` then runs at native
        # scale, which is a real caveat and is warned about at predict time
        # rather than silently assumed to be correct.
        self._pixel_size_nm = float(pixel_size_nm) if pixel_size_nm else None
        self._overlap = float(overlap)
        self._model: engine.LoadedModel | None = None
        # The stored uint8 probability map, in the image's own pixel
        # coordinates, from the most recent run_dl_inference() or from
        # adopt_native_probability_map(). extract_instances() thresholds *this*
        # -- see the module docstring.
        self._native_prob: resample.NativeProbabilityMap | None = None
        # Set by apply_adapter(); see there.
        self._adapter_id: str | None = None
        self._adapter_head: Path | None = None
        # Plain-language sentences about where this run actually ran -- see
        # ``device_notices``. Empty on the ordinary path.
        self._device_notices: list[str] = []

    # --- Identity ---

    @property
    def name(self) -> str:
        return self._organelle.key

    @property
    def generated_flag(self) -> str:
        return self._organelle.generated_flag

    @property
    def prob_map_prefix(self) -> str:
        return self._organelle.key

    @property
    def source_model(self) -> str:
        return self._source_model

    @property
    def persist_probability_maps(self) -> bool:
        # This flag answers exactly one question inside
        # run_inference_for_segmentation: *may a stored map be substituted for
        # running the model?* The answer stays no, and it is not the same
        # question as "is the stored map trustworthy" -- under the native-
        # coordinate ordering it is, which is what
        # replay_stored_probability_map exists to use. "Run" means run: a user
        # who asks for a re-run after fine-tuning, or after changing anything
        # about the image, must get the model, not last week's bytes. Changing
        # the *threshold* is a different verb and takes the replay path.
        return False

    @property
    def model_spec(self):
        return self._spec

    @property
    def fg_threshold(self) -> float:
        """Foreground probability threshold this run will use."""
        return self._fg_threshold

    def set_fg_threshold(self, threshold: float) -> None:
        """Move the threshold for the next extraction.

        The accuracy dial's backend: with the stored native map in hand
        (:meth:`adopt_native_probability_map`), changing this and extracting
        again is the whole operation -- no model, no resampling, no second
        decision procedure. Setting it does not touch the stored map, which is
        the point: every threshold reads the same bytes.

        The value lands in provenance through
        :meth:`get_probability_map_metadata`, alongside the level it becomes and
        the cut that level actually applies.
        """
        value = float(threshold)
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"threshold must be a probability in [0, 1]; got {threshold!r}"
            )
        self._fg_threshold = value

    @property
    def adapter_id(self) -> str | None:
        """The applied adapter, if this run is wearing one."""
        return self._adapter_id

    # --- Adaptation ---

    supports_adapters = True

    def apply_adapter(
        self,
        *,
        adapter_id: str,
        base_model: str,
        calibrated_threshold: float | None = None,
        head_file: Path | str | None = None,
    ) -> None:
        """Run this segmenter with a user-fitted threshold and head.

        Called by :func:`quantem.seg_core.db.inference.run_inference_for_segmentation`
        for the adapter the user applied to the segmentation. Without it a
        calibrated threshold and a trained head are computed, verified, and
        never used.

        ``base_model`` is checked against the pack this segmenter was built for
        and a mismatch is refused. An adapter is a *threshold and head fitted to
        one model's probability distribution*; a threshold of 0.31 calibrated on
        ``quantem:mito`` says nothing about where ``omniem:mito``'s foreground
        starts, and applying it across families would produce a plausible,
        wrong, and silently mislabelled segmentation.

        Raises:
            ValueError: when ``base_model`` is not the pack being run.
        """
        if base_model != self._spec.pack_id:
            raise ValueError(
                f"adapter {adapter_id} was fitted on {base_model!r}, but this run "
                f"uses {self._spec.pack_id!r}. A calibrated threshold and a trained "
                "head belong to the model they were fitted on."
            )
        self._adapter_id = str(adapter_id)
        if calibrated_threshold is not None:
            self._fg_threshold = float(calibrated_threshold)
        self._adapter_head = Path(head_file) if head_file else None
        # Force a reload: load_models() may already have run, and the module in
        # hand is the released one.
        self._model = None
        logger.info(
            "Adapter %s applied to %s (threshold=%.3f, head=%s)",
            self._adapter_id,
            self._spec.pack_id,
            self._fg_threshold,
            self._adapter_head or "released",
        )

    # --- DL inference ---

    def load_models(self) -> None:
        """Resolve and load the pack. Cheap after the first call in a process.

        The first call in a process is *not* cheap -- 4 to 20 s for an exported
        encoder, minutes for the eager fallback -- and for that whole time the
        run used to report a hard-coded 5 % and nothing else. Naming the stage
        before the load starts is what turns that silence into a sentence.
        """
        _report_stage(
            STAGE_LOADING_MODEL,
            model=self._spec.pack_id,
            adapter=self._adapter_id,
        )
        if self._adapter_head is not None:
            self._model = engine.load_adapted_model(
                self._spec.pack_id,
                self._adapter_head,
                self._device,
                adapter_id=self._adapter_id,
            )
            return
        self._model = engine.load_model(self._spec.pack_id, self._device)

    @property
    def encoder_tier(self) -> str | None:
        """How the loaded model's encoder was built, or None before loading.

        ``"exported"`` is the fast shipping path; ``"timm"`` / ``"dinov3"`` mean
        the run fell back to rebuilding the encoder from raw weights (minutes,
        not seconds). Public so the run task can surface the slow fallback on
        the job record instead of the user only seeing an unexplained 4-minute
        load.
        """
        return getattr(self._model, "encoder_tier", None) if self._model else None

    @property
    def inference_device(self) -> str | None:
        """The device the last run finished on: ``cpu`` / ``cuda`` / ``mps``.

        The device the run *finished* on, which is not always the one it was
        offered: a model that cannot execute on the graphics card, or a card
        that ran out of memory, moves the run to the processor. Provenance must
        record what happened, not what was asked for.
        """
        return getattr(self._model, "device", None) if self._model else None

    @property
    def device_notices(self) -> list[str]:
        """Sentences about where this run ran, for the run record. Usually empty.

        Populated when something changed the arithmetic's location or size --
        the graphics card could not run this model, or ran short of memory and
        the batch shrank, or the run moved to the processor part-way through.
        Each is one plain sentence naming what happened and what it cost, in the
        app's own vocabulary; none of them asks the user to do anything, because
        there is nothing for them to do.

        Read by the run task after inference, exactly as ``encoder_tier`` is: a
        run that took twenty minutes when the estimate said one is the surprise
        that destroys trust in the estimate, and a silent fallback is how that
        happens.
        """
        return list(self._device_notices)

    def get_dl_model_names(self) -> list[str]:
        return [DL_MODEL_NAME]

    def estimate_dl_tile_count(self, image_shape: tuple[int, int]) -> int | None:
        return engine.estimate_tiles(
            self._spec,
            image_shape,
            pixel_size_nm=self._pixel_size_nm,
            overlap=self._overlap,
        )

    def run_dl_inference(
        self,
        image: np.ndarray,
        cached_prob_maps: dict[str, np.ndarray | None],
        on_progress: Callable[[str, float], None] | None = None,
        *,
        pixel_size_nm: float | None = None,
        **_kwargs,
    ) -> dict[str, np.ndarray]:
        report = on_progress or (lambda _stage, _fraction: None)
        self._device_notices = []
        cached = cached_prob_maps.get(DL_MODEL_NAME)
        if cached is not None:
            # A cached map is already in native coordinates (it was written by
            # this pipeline). Re-adopting it through the same quantiser is what
            # makes "thresholds run uniformly on the same stored map" true of
            # the cached path as well as the fresh one.
            self.adopt_native_probability_map(
                resample.NativeProbabilityMap(
                    data=resample.quantize_probability(cached)
                )
            )
            # The array itself is handed back untouched, as the BaseSegmenter
            # contract requires; the quantised copy above is what the threshold
            # will read. For a map this pipeline stored the two are the same
            # numbers -- the uint8 -> float -> uint8 round trip is exact.
            return {DL_MODEL_NAME: cached}

        if self._model is None:
            self.load_models()

        report(DL_MODEL_NAME, 0.0)
        effective_pixel_size = pixel_size_nm or self._pixel_size_nm
        # The denominator goes on the row *before* the first forward pass, so
        # the run can say "0 of 858 tiles" instead of nothing while the first
        # tile is in flight. ``on_tile`` then reports whole tiles, and carries
        # the plan's own total, which supersedes this one if they ever differ.
        planned_tiles = engine.estimate_tiles(
            self._spec,
            image.shape[:2],
            pixel_size_nm=effective_pixel_size,
            overlap=self._overlap,
        )
        with _tile_scope(
            planned_tiles,
            model=self._spec.pack_id,
            organelle=self.ORGANELLE,
            device=getattr(self._model, "device", None),
        ) as tiles:
            prediction = engine.predict_region(
                self._model,
                image,
                pixel_size_nm=effective_pixel_size,
                overlap=self._overlap,
                on_progress=lambda fraction: report(DL_MODEL_NAME, fraction),
                on_tile=lambda done, total: tiles.set(done, total=total),
            )
        # Where the run ended up, which a fallback may have changed while those
        # tiles were being blended. Collected here rather than left in the
        # engine because the model object is cached across runs and these
        # sentences belong to this one.
        self._device_notices = [
            *getattr(self._model, "load_notices", ()),
            *getattr(prediction, "notices", ()),
        ]
        # Step 3 of the pipeline: the field crosses to the image's own pixel
        # grid and is quantised, in one call, before anything is decided about
        # it. The model-scale float is dropped here -- it is not the authority
        # for anything downstream, and on a 50 MP image it is 200 MB.
        native = resample.NativeProbabilityMap.from_model_grid(
            prediction.prob, prediction.context
        )
        if native.shape != tuple(image.shape[:2]):
            raise ValueError(
                f"prob map shape {native.shape} != image {tuple(image.shape[:2])}"
            )
        self.adopt_native_probability_map(native)
        report(DL_MODEL_NAME, 1.0)
        return {DL_MODEL_NAME: native.as_float()}

    def combine_prob_maps(self, prob_maps: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(prob_maps[DL_MODEL_NAME], dtype=np.float32)

    # --- The stored native-coordinate map ---

    @property
    def native_probability_map(self) -> resample.NativeProbabilityMap | None:
        """The stored uint8 map this segmenter will threshold, or None.

        Public because it is the run's authoritative artifact, not an
        implementation detail: it is what gets written to disk and what a later
        threshold movement re-reads.
        """
        return self._native_prob

    def adopt_native_probability_map(
        self, native: resample.NativeProbabilityMap
    ) -> None:
        """Use this stored map as the authority for the next extraction.

        Called by :meth:`run_dl_inference` with what the model just produced,
        and by the replay path
        (:func:`quantem.seg_core.db.inference.replay_stored_probability_map`)
        with bytes read back off disk. Both then take the identical threshold
        path, which is what makes a dial movement and a fresh run agree.
        """
        if native.data.dtype != np.uint8:
            raise TypeError("a stored probability map is uint8")
        self._native_prob = native

    # --- Instance extraction ---

    def _foreground_mask(self, prob: np.ndarray) -> np.ndarray:
        """Binary foreground in the image's own pixel coordinates.

        Thresholds the **stored uint8 map** whenever there is one covering this
        region -- never the float in ``prob``, which is a dequantised copy of
        those same bytes and would re-decide them at a different precision.

        The fallback quantises what the caller supplied and thresholds that, so
        a caller who hands in its own probabilities still gets the one threshold
        operation this app performs rather than a second, slightly different
        one.
        """
        native = self._native_prob
        native_shape = tuple(prob.shape[:2])
        if native is not None and native.shape == native_shape:
            return native.foreground(self._fg_threshold)
        return resample.binarize_quantized(
            resample.quantize_probability(prob), self._fg_threshold
        )

    def extract_instances(
        self,
        prob: np.ndarray,
        image: np.ndarray,
        prob_maps: dict[str, np.ndarray],
        *,
        min_area: int | None = None,
        coordinate_offset: tuple[float, float] | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[ExtractedSegment]:
        report = on_progress or (lambda _fraction: None)
        report(0.0)
        # Tiles are done; what follows is morphology and measurement, which is
        # a different kind of work and takes a different amount of time. Saying
        # so is what stops the tile bar sitting at 100 % looking hung.
        _report_stage(STAGE_EXTRACTING, organelle=self.ORGANELLE)

        area_floor = int(min_area) if min_area is not None else self._min_area
        dx, dy = coordinate_offset or (0.0, 0.0)

        # Every object-level decision is made here, on native pixels: the
        # threshold (inside _foreground_mask, on the stored uint8 map), then the
        # closing radius, the hole fill, the labeling and the area floor -- all
        # of which are defined in native pixels and none of which now sees the
        # model's grid at all.
        labels = postprocess.postprocess_mask(
            self._foreground_mask(prob),
            close_radius=self._organelle.close_radius,
            min_area=area_floor,
        )
        intensity = image if image is not None and image.ndim == 2 else None

        segments: list[ExtractedSegment] = []
        regions = regionprops(labels, intensity_image=intensity)
        total = max(len(regions), 1)
        for idx, region in enumerate(regions):
            segment = build_segment_from_region(
                region,
                labels,
                prob_maps,
                prob,
                self.generated_flag,
                float(dx),
                float(dy),
                image,
            )
            if segment is not None:
                segments.append(segment)
            if idx % 64 == 0:
                report(idx / total)
        report(1.0)
        return segments

    # --- Provenance ---

    def get_probability_map_metadata(self, model_name: str) -> dict[str, object]:
        # ``adapter_id`` and ``adapted_head`` are recorded whenever they are set:
        # a map made at a user-calibrated threshold is not the same artifact as
        # one made at the published default, and the difference has to travel
        # with the pixels rather than live only in a log line.
        #
        # The map's own provenance is merged in below: which interpolator
        # carried the field back to native pixels, how it was quantised, the
        # level the threshold became and the cut that level actually applies,
        # and that the decision was taken on the stored native map. Those are
        # the degrees of freedom in the result; a reader who has only the pack
        # id and the requested threshold cannot reconstruct the boundary.
        metadata: dict[str, object] = {
            "model_name": model_name,
            "pack_id": self._spec.pack_id,
            "family": self._family,
            "organelle": self.ORGANELLE,
            "canonical_nm": self._spec.canonical_nm,
            "tile_size": self._spec.tile_size,
            "overlap": self._overlap,
            "threshold": self._fg_threshold,
            "default_threshold": self._spec.threshold,
            "adapter_id": self._adapter_id,
            "adapted_head": str(self._adapter_head) if self._adapter_head else None,
        }
        native = self._native_prob
        if native is not None:
            metadata.update(native.provenance(self._fg_threshold))
        return metadata


class DinoMitoSegmenter(DinoOrganelleSegmenter):
    ORGANELLE = "mito"


class DinoErSegmenter(DinoOrganelleSegmenter):
    ORGANELLE = "er"


class DinoLdSegmenter(DinoOrganelleSegmenter):
    ORGANELLE = "ld"


class DinoNucleusSegmenter(DinoOrganelleSegmenter):
    ORGANELLE = "nucleus"
