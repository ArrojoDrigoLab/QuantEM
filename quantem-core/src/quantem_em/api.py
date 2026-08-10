"""The public surface of ``quantem-core``.

Everything a front-end needs, and nothing about how it is displayed. Progress and cancellation are
plain callables so the same code drives a napari worker, a CLI, and the desktop application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import device as _device
from .registry import (
    DEFAULT_MODEL_FOR_ORGANELLE,
    ORGANELLE_LABELS,
    get_model_spec,
    list_models,
)
from .spec import ModelSpec

__all__ = [
    "list_models",
    "get_model_spec",
    "DEFAULT_MODEL_FOR_ORGANELLE",
    "ORGANELLE_LABELS",
    "ProbabilityResult",
    "SegmentationResult",
    "QuantEMModel",
    "load_model",
    "segment",
]


@dataclass(frozen=True)
class ProbabilityResult:
    probability: np.ndarray  # float32 [H, W] on the caller's grid
    affinities: np.ndarray | None
    contract: dict


@dataclass(frozen=True)
class SegmentationResult:
    """A segmentation plus everything needed to reproduce it."""

    labels: np.ndarray  # int32 [H, W], 0 = background
    mask: np.ndarray  # bool  [H, W]
    probability: np.ndarray  # float32 [H, W]
    n_objects: int
    contract: dict = field(default_factory=dict)

    def measure(self, intensity_image=None, pixel_size_nm=None):
        from .measure import measure_objects

        return measure_objects(self.labels, intensity_image, pixel_size_nm=pixel_size_nm)

    def summary(self, pixel_size_nm=None, tissue_mask=None):
        from .measure import summarize

        return summarize(self.labels, pixel_size_nm=pixel_size_nm, tissue_mask=tissue_mask)


class QuantEMModel:
    """A loaded released model. **Not thread-safe** — use one instance per worker.

    (The affinity decoder writes its auxiliary output onto itself during ``forward``.)
    """

    def __init__(self, spec: ModelSpec, module, torch_device):
        self.spec = spec
        self.module = module
        self.device = torch_device
        self._adapted_from: str | None = None
        self._calibrated_threshold: float | None = None

    # -- introspection -------------------------------------------------------
    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def threshold(self) -> float:
        """The threshold to use: a calibrated one if fine-tuning produced it, else the default."""
        return (
            self._calibrated_threshold
            if self._calibrated_threshold is not None
            else self.spec.fg_threshold
        )

    def estimate(self, shape, pixel_size_nm=None) -> dict:
        """Window count, working size and peak memory, before running anything."""
        from .inference.predict import plan_resample
        from .inference.tiling import round_up, stride_for, window_count

        factors, info = plan_resample(
            self.spec, tuple(shape[-2:]), pixel_size_nm, allow_extreme=True
        )
        work = info.get("working_shape", tuple(shape[-2:])) if factors else tuple(shape[-2:])
        t = round_up(self.spec.tile_size, self.spec.encoder.patch_size)
        stride = stride_for(t, self.spec.overlap)
        padded = (max(work[0], t), max(work[1], t))
        n = window_count(padded, t, stride)
        # 2 x float32 accumulator + float32 weight sum + float32 input + uint8 resampled
        peak = int(padded[0] * padded[1] * 17)
        return {
            "windows": n,
            "tile": t,
            "stride": stride,
            "working_shape": work,
            "peak_bytes": peak,
            "resample_factor": info.get("resample_factor"),
            "warnings": info.get("warnings", []),
        }

    # -- inference -----------------------------------------------------------
    def predict_probability(
        self,
        image,
        *,
        pixel_size_nm=None,
        invert=False,
        return_affinities=False,
        allow_extreme_resample=False,
        progress=None,
        cancel=None,
    ) -> ProbabilityResult:
        from .inference.predict import predict_image

        want_aux = bool(return_affinities) and self.spec.task == "instance"
        fg, contract = predict_image(
            self.module,
            image,
            self.spec,
            self.device,
            pixel_size_nm=pixel_size_nm,
            invert=invert,
            collect_aux=want_aux,
            allow_extreme_resample=allow_extreme_resample,
            progress=progress,
            cancel=cancel,
        )
        contract["quantem_core_version"] = _version()
        if self._adapted_from:
            contract["adapted_from"] = self._adapted_from
        return ProbabilityResult(fg, None, contract)

    def segment(
        self,
        image,
        *,
        pixel_size_nm=None,
        invert=False,
        threshold=None,
        fill_holes=True,
        allow_extreme_resample=False,
        progress=None,
        cancel=None,
    ) -> SegmentationResult:
        """Segment a 2-D image. Instances are connected components (see ``postprocess``)."""
        from .inference.postprocess import segment_from_probability

        prob = self.predict_probability(
            image,
            pixel_size_nm=pixel_size_nm,
            invert=invert,
            allow_extreme_resample=allow_extreme_resample,
            progress=progress,
            cancel=cancel,
        )
        thr = self.threshold if threshold is None else float(threshold)
        labels, mask, n = segment_from_probability(
            prob.probability,
            fg_threshold=thr,
            close_radius=self.spec.close_radius,
            min_area=self.spec.min_area,
            fill_holes=fill_holes,
        )
        contract = dict(prob.contract)
        contract.update(
            threshold=thr,
            instances="connected",
            min_area=self.spec.min_area,
            close_radius=self.spec.close_radius,
            n_objects=n,
            threshold_source="calibrated" if self._calibrated_threshold is not None else "default",
        )
        return SegmentationResult(labels, mask, prob.probability, n, contract)

    def segment_stack(self, stack, *, z_range=None, **kw):
        """Segment a ZYX stack slice by slice. Yields ``(z, SegmentationResult)``.

        The models are 2-D; this is independent per-slice segmentation, not 3-D instance
        segmentation. Use ``postprocess.link_across_z`` if you want ids joined through z.
        """
        stack = np.asarray(stack)
        if stack.ndim != 3:
            raise ValueError(f"expected a 3-D ZYX stack, got {stack.ndim}-D")
        zs = range(*z_range) if z_range else range(stack.shape[0])
        for z in zs:
            yield z, self.segment(stack[z], **kw)

    # -- adaptation ----------------------------------------------------------
    def finetune(self, examples, **kw):
        """Head-only fine-tuning. See :mod:`quantem_em.adapt`."""
        from .adapt import finetune_head

        return finetune_head(self, examples, **kw)

    def save_adapted(self, path, *, note: str = "") -> Path:
        from .adapt import save_adapted_head

        return save_adapted_head(self, path, note=note)


def _version() -> str:
    from . import __version__

    return __version__


def load_model(
    model_id: str,
    *,
    device: str = "auto",
    weights: dict | None = None,
    allow_network: bool = True,
    progress=None,
) -> QuantEMModel:
    """Load a released model, fetching its artifacts if needed.

    ``weights`` may be an explicit ``{artifact_name: path}`` mapping (used by tests and by the
    side-load path); otherwise the registry cache is consulted and, with permission, the files are
    downloaded.
    """
    from safetensors.torch import load_file

    from .models.build import build_from_artifacts
    from .weights import fetch

    spec = get_model_spec(model_id)
    names = fetch.artifacts_for(spec)
    paths = weights or fetch.ensure(names, progress=progress, allow_network=allow_network)

    merged: dict = {}
    for name in names:  # trunk first, model artifact second (overwrites)
        merged.update(load_file(str(paths[name])))

    dev = _device.resolve(device)
    module = build_from_artifacts(spec, merged, device=str(dev))
    return QuantEMModel(spec, module, dev)


def segment(image, *, organelle: str, family: str | None = None, device: str = "auto", **kw):
    """One-liner: pick the default model for an organelle and segment."""
    model_id = f"{family}/{organelle}" if family else DEFAULT_MODEL_FOR_ORGANELLE[organelle]
    return load_model(model_id, device=device).segment(image, **kw)
