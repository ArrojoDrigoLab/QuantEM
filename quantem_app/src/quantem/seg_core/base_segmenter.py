"""
Abstract Base Segmenter
========================

Central abstraction for organelle segmentation pipelines. A segmenter owns the
whole path from an image array to a list of extracted instances.

Subclasses implement the abstract methods for their specific model, probability
map naming, and extraction strategy. The concrete ``predict()`` provides the
shared orchestration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from .types import ExtractedSegment, InferenceResult

logger = logging.getLogger(__name__)


class BaseSegmenter(ABC):
    """Interface implemented by every QuantEM segmenter.

    This is the contract the in-process inference module
    (:mod:`quantem.inference`) must satisfy. Nothing outside a segmenter knows
    what a model is: the job handler resolves a segmenter through
    :mod:`quantem.seg_core.registry`, and :mod:`quantem.seg_core.db` drives it.

    Lifecycle of one run (see :func:`quantem.seg_core.db.inference.
    run_inference_for_segmentation`)::

        segmenter = get_segmenter(internal_name, source_model=...)   # cheap, no I/O
        segmenter.load_models()                                      # weights resolved here
        result = segmenter.predict(image, cached_prob_maps, on_progress=...)
        segments = segmenter.extract_instances(result.prob, image,
                                               result.prob_maps, min_area=...,
                                               coordinate_offset=(roi.x, roi.y))

    Required of an implementation
    -----------------------------
    * ``__init__`` must be cheap and must not touch the filesystem or a GPU.
      Everything expensive belongs in :meth:`load_models`, which is called once
      per run and is expected to hit a process-level model cache.
    * :meth:`run_dl_inference` returns one float32 probability map per name in
      :meth:`get_dl_model_names`, each in ``[0, 1]`` and the **same shape as the
      input image** (native pixels). Any resampling to a model's canonical
      nm/px, tiling, and blending happens inside the implementation and must be
      undone before returning.
    * A cached map (``cached_prob_maps[name] is not None``) must be returned
      unchanged rather than recomputed.
    * :meth:`extract_instances` consumes ``prob`` and returns
      :class:`~quantem.seg_core.types.ExtractedSegment` in **parent-image pixel
      coordinates** -- add ``coordinate_offset`` to every polygon, centroid and
      bbox so ROI results land in the right place on the full image.
    * ``on_progress(stage, fraction)`` may be called freely; ``stage`` is a
      short lowercase key and ``fraction`` is in ``[0, 1]``.

    Progress, and why it does not come back through ``on_progress``
    ---------------------------------------------------------------
    ``on_progress`` carries a fraction, and a fraction cannot say "531 of 858
    tiles" without being rounded back into a count that may not be the one the
    loop reached. Countable work is therefore reported *sideways*, straight
    onto the job row, by :func:`quantem.jobs.reporter.unit_scope`: the running
    job's reporter registers itself on its thread, and the tiling loop writes
    ``progress_units_done`` / ``progress_units_total`` as it goes. An
    implementation that runs sliding windows should open a scope around its
    loop (:mod:`quantem.inference.segmenter` is the worked example); one that
    does not is simply absent from the tile numbers, which is honest.

    Not part of the contract any more
    ---------------------------------
    There is no per-call retraining hook, and no ensemble: nothing trains a
    per-segmentation model on top of the DL output inside ``predict()``.
    Adaptation happens
    offline in :mod:`quantem.finetune`, which fits a threshold and (optionally)
    a head once; a run either wears that result from the start
    (:meth:`apply_adapter`, called before :meth:`load_models`) or does not.
    Inference remains a pure function of (image, model).
    """

    # --- Identity ---

    @property
    @abstractmethod
    def name(self) -> str:
        """Short lowercase name, e.g. 'er', 'mito'."""
        ...

    @property
    @abstractmethod
    def generated_flag(self) -> str:
        """Feature key marking segments from this segmenter, e.g. 'er_generated'."""
        ...

    @property
    @abstractmethod
    def prob_map_prefix(self) -> str:
        """Prefix for probability map filenames, e.g. 'er', 'mito'."""
        ...

    @property
    def source_model(self) -> str:
        """Stable source-model key used for SegmentObject provenance."""
        return self.name

    @property
    def job_resource_class(self) -> str:
        """Preferred queue resource class for inference jobs."""
        return "gpu"

    @property
    def min_area(self) -> int | None:
        """Native-pixel area floor this segmenter applies, or None for no opinion.

        The area floor is a property of the *organelle* -- 60 px for
        mitochondria, 8000 for nuclei -- and of the resolution the model was
        tuned at, so the segmenter is the only thing that can state it. The
        driver (:func:`quantem.seg_core.db.extraction.resolve_min_area`) asks
        for it rather than imposing one generic number on every organelle, and
        records the answer in the object's run identity.

        The default reads ``self._min_area``, which is where the shipped
        segmenters keep the value resolved in ``__init__`` from an explicit
        override or the organelle spec. Override the property outright if an
        implementation computes it some other way.
        """
        return getattr(self, "_min_area", None)

    @property
    def persist_probability_maps(self) -> bool:
        """Whether this segmenter writes probability-map artifacts."""
        return True

    @property
    def supports_image_file_prediction(self) -> bool:
        """Whether full-image inference can stream tiles from the source image.

        True means :meth:`predict_from_image_file` is implemented and the caller
        must not load the whole image into memory. This is the bounded-memory
        path for gigapixel assets.
        """
        return False

    def estimate_image_file_prediction_units(
        self,
        image_shape: tuple[int, int],
    ) -> int | None:
        """Optional full-image work estimate for progress reporting."""
        _ = image_shape
        return None

    def estimate_dl_tile_count(self, image_shape: tuple[int, int]) -> int | None:
        """How many windows this segmenter will run over ``image_shape``.

        Called before any model is loaded so the run has a denominator to show
        while the weights are still coming off disk. An implementation that
        returns a number must return the number its loop will actually count
        to -- the window layout is fully determined by the region shape, the
        pack's canonical nm/px, its tile size and its patch size, so there is
        nothing to guess and no excuse for an approximation the user then
        watches overshoot. (:func:`quantem.inference.engine.estimate_tiles` is
        the reference implementation and is exact.)

        Return ``None`` to let the caller fall back to a generic estimate.

        Progress does not depend on this being called: the tiling loop reports
        whole tiles with the plan's own total as it goes (see
        :func:`quantem.inference.tiling.blend_region_streaming`), and that total
        wins if the two ever disagree. This is what makes the count exist
        *before* tile 1.
        """
        _ = image_shape
        return None

    # --- Adaptation ---

    #: True when :meth:`apply_adapter` does something. The driver checks this
    #: rather than ``hasattr``, so a segmenter that cannot use an adapter is a
    #: deliberate statement rather than an accident of naming.
    supports_adapters: bool = False

    def apply_adapter(
        self,
        *,
        adapter_id: str,
        base_model: str,
        calibrated_threshold: float | None = None,
        head_file=None,
    ) -> None:
        """Use a guided fine-tuning result for this run.

        Called by :func:`quantem.seg_core.db.inference.run_inference_for_segmentation`
        **before** :meth:`load_models`, for the adapter the user applied to the
        segmentation. An implementation must refuse an adapter whose
        ``base_model`` is not the model it is about to run: a calibrated
        threshold describes one model's probability distribution and means
        nothing about another's.

        The default raises, because silently ignoring an applied adapter is how
        a user ends up with a run they believe is calibrated and is not.
        """
        _ = (adapter_id, calibrated_threshold, head_file)
        # The Python class name used to lead this sentence. It lands in the
        # Tasks & Queues panel through the failed job's message, where it is
        # I-12's exception/class class and means nothing to the reader; the
        # model the adapter was fitted on is the fact that matters.
        raise NotImplementedError(
            f"This model cannot use the adapter fitted on {base_model!r}: it "
            "does not support adaptation."
        )

    def predict_from_image_file(
        self,
        image_file,
        cached_prob_maps: dict[str, np.ndarray | None] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
        **kwargs,
    ) -> InferenceResult:
        """Optional full-image prediction path that streams from the source image."""
        _ = (image_file, cached_prob_maps, on_progress, kwargs)
        raise NotImplementedError(
            "This model cannot run over a whole image file."
        )

    # --- DL Inference ---

    @abstractmethod
    def load_models(self) -> None:
        """Resolve and load model weights. Call once, reuse for many predictions."""
        ...

    @abstractmethod
    def run_dl_inference(
        self,
        image: np.ndarray,
        cached_prob_maps: dict[str, np.ndarray | None],
        on_progress: Callable[[str, float], None] | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Run DL inference, returning named probability maps.

        Must skip models whose cached value is not None.

        Args:
            image: Input image (uint8 grayscale).
            cached_prob_maps: Dict of cached maps. None means not cached.
            on_progress: Optional progress callback.

        Returns:
            Dict of model_name -> float32 probability map, same shape as image.
        """
        ...

    @abstractmethod
    def combine_prob_maps(self, prob_maps: dict[str, np.ndarray]) -> np.ndarray:
        """Combine the named probability maps into one foreground map.

        Single-model segmenters pass their one map through.
        """
        ...

    @abstractmethod
    def get_dl_model_names(self) -> list[str]:
        """DL model names for prob map caching, e.g. ['DINO']."""
        ...

    def get_probability_map_metadata(self, model_name: str) -> dict[str, object]:
        """Optional per-model metadata to persist on probability maps."""
        return {}

    # --- Instance Extraction ---

    @abstractmethod
    def extract_instances(
        self,
        prob: np.ndarray,
        image: np.ndarray,
        prob_maps: dict[str, np.ndarray],
        *,
        min_area: int,
        coordinate_offset: tuple[float, float] | None,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[ExtractedSegment]:
        """Extract segment instances from the foreground probability map."""
        ...

    # --- Concrete Orchestration ---

    def predict(
        self,
        image: np.ndarray,
        cached_prob_maps: dict[str, np.ndarray | None] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
        **kwargs,
    ) -> InferenceResult:
        """Full prediction pipeline. Shared orchestration.

        Args:
            image: Input image (uint8 grayscale).
            cached_prob_maps: Cached probability maps (None = not cached).
            on_progress: Optional progress callback.

        Returns:
            InferenceResult with prob_maps and prob.
        """
        cached = cached_prob_maps or {}
        prob_maps = self.run_dl_inference(image, cached, on_progress, **kwargs)

        if on_progress is not None:
            on_progress("combine", 0.0)
        prob = self.combine_prob_maps(prob_maps)
        if on_progress is not None:
            on_progress("combine", 1.0)

        return InferenceResult(prob_maps=prob_maps, prob=prob)
