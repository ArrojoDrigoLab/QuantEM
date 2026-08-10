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
* :mod:`.resample`     -- native <-> canonical, mask back with NEAREST
* :mod:`.postprocess`  -- threshold -> close -> fill -> label -> min area
* :func:`quantem.seg_core.extraction.build_segment_from_region` -- the shape
  measurements every downstream analysis reads

The one behavioural subtlety: the model predicts on a resampled grid, so the
foreground decision is made *there* and only the resulting binary mask is
brought back to native pixels. The native-scale probability map returned in
``InferenceResult`` is for display and per-object confidence; it is never
re-thresholded. See :mod:`.resample` for why.
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
        # Model-scale prediction from the most recent run_dl_inference(), so
        # extract_instances() can threshold on the grid the model predicted on.
        self._last_prediction: engine.RegionPrediction | None = None
        # Set by apply_adapter(); see there.
        self._adapter_id: str | None = None
        self._adapter_head: Path | None = None

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
        # Probability maps are re-runnable and large; the proofreading UI reads
        # segments, not maps.
        return False

    @property
    def model_spec(self):
        return self._spec

    @property
    def fg_threshold(self) -> float:
        """Foreground probability threshold this run will use."""
        return self._fg_threshold

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
        """Resolve and load the pack. Cheap after the first call in a process."""
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
        cached = cached_prob_maps.get(DL_MODEL_NAME)
        if cached is not None:
            self._last_prediction = None
            return {DL_MODEL_NAME: cached}

        if self._model is None:
            self.load_models()

        report(DL_MODEL_NAME, 0.0)
        prediction = engine.predict_region(
            self._model,
            image,
            pixel_size_nm=pixel_size_nm or self._pixel_size_nm,
            overlap=self._overlap,
            on_progress=lambda fraction: report(DL_MODEL_NAME, fraction),
        )
        self._last_prediction = prediction

        native_prob = resample.probability_to_native(
            prediction.prob, prediction.context
        )
        if native_prob.shape != image.shape[:2]:
            raise ValueError(
                f"prob map shape {native_prob.shape} != image {image.shape[:2]}"
            )
        report(DL_MODEL_NAME, 1.0)
        return {DL_MODEL_NAME: np.asarray(native_prob, dtype=np.float32)}

    def combine_prob_maps(self, prob_maps: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(prob_maps[DL_MODEL_NAME], dtype=np.float32)

    # --- Instance extraction ---

    def _foreground_mask(self, prob: np.ndarray) -> np.ndarray:
        """Binary foreground at native scale.

        Threshold on the model's own grid when the last prediction is still the
        one that produced ``prob``, then map the mask back NEAREST. Falls back to
        thresholding the native map (cached maps, or a caller that supplied its
        own probabilities).
        """
        prediction = self._last_prediction
        native_shape = tuple(prob.shape[:2])
        if prediction is not None and prediction.context.native_shape == native_shape:
            mask = postprocess.binarize(prediction.prob, self._fg_threshold)
            return resample.mask_to_native(mask, prediction.context)
        return postprocess.binarize(prob, self._fg_threshold)

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

        area_floor = int(min_area) if min_area is not None else self._min_area
        dx, dy = coordinate_offset or (0.0, 0.0)

        # Morphology and the area filter run at native scale, where
        # close_radius and min_area are defined.
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
        return {
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


class DinoMitoSegmenter(DinoOrganelleSegmenter):
    ORGANELLE = "mito"


class DinoErSegmenter(DinoOrganelleSegmenter):
    ORGANELLE = "er"


class DinoLdSegmenter(DinoOrganelleSegmenter):
    ORGANELLE = "ld"


class DinoNucleusSegmenter(DinoOrganelleSegmenter):
    ORGANELLE = "nucleus"
