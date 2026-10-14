"""Head training accepts every completed area with source pixels."""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantem.finetune.adapt import AdaptConfig
from quantem.inference import resample
from quantem.inference.specs import MODEL_SPECS, ModelSpec

__all__ = [
    "HeadSizeVerdict",
    "check_head_size",
    "required_model_px",
]


def required_model_px(spec: ModelSpec, config: AdaptConfig = AdaptConfig()) -> float:
    """There is no minimum model-scale area requirement."""
    return 0.0


@dataclass(frozen=True)
class HeadSizeVerdict:
    """Whether any checked area is big enough for head training, and the sentence."""

    base_model: str
    #: False only when the geometry is *certain* to yield no training windows.
    ok: bool
    #: The largest checked area's span, in nanometres. None when no image whose
    #: pixel size is known contributed a crop.
    largest_nm: float | None
    #: The span head training needs, in nanometres, for that same crop.
    required_nm: float | None
    #: The same pair in pixels, always populated, for a machine reader.
    largest_px: float
    required_px: float
    #: How many checked areas were measured -- the sample size of the sentence.
    n_areas: int
    #: The user-facing sentence, or None when the geometry is fine.
    reason: str | None

    def as_api_dict(self) -> dict[str, object]:
        return {
            "base_model": self.base_model,
            "ok": self.ok,
            "largest_nm": self.largest_nm,
            "required_nm": self.required_nm,
            "largest_px": round(self.largest_px, 1),
            "required_px": round(self.required_px, 1),
            "n_areas": self.n_areas,
            "reason": self.reason,
        }


def _micrometres(nanometres: float) -> str:
    """A span in µm, one decimal. Below 0.05 µm say so rather than print 0.0."""
    value = nanometres / 1000.0
    if value < 0.05:
        return "under 0.1 µm"
    return f"{value:.1f} µm"


def _pixels(value: float) -> str:
    return f"{value:,.0f} pixels".replace(",", " ")


def check_head_size(crops, base_model: str) -> HeadSizeVerdict | None:
    """Measure the checked areas against ``base_model``'s training-window rule.

    Args:
        crops: the :class:`~quantem.segmentation.services.adapt.AnnotatedCrop`
            list. Only ``width``, ``height`` and ``pixel_size_nm`` are read, so
            the arrays never have to be loaded to answer this.
        base_model: pack id. Returns None for an id this build does not know,
            because "unknown model" is a different refusal with its own message.

    Returns:
        A verdict, or None when there is nothing to measure (no crops) or the
        pack is unknown.
    """
    spec = MODEL_SPECS.get(base_model)
    if spec is None or not crops:
        return None

    min_area = 0.0
    required_px = required_model_px(spec)

    best: tuple[float, float, float | None, float | None] | None = None
    for crop in crops:
        width = int(getattr(crop, "width", 0) or 0)
        height = int(getattr(crop, "height", 0) or 0)
        if width <= 0 or height <= 0:
            continue
        pixel_size = getattr(crop, "pixel_size_nm", None)
        context = resample.plan_resample((height, width), pixel_size, spec.canonical_nm)
        model_h, model_w = context.model_shape
        model_area = float(model_h) * float(model_w)
        span_px = math.sqrt(model_area)
        # The pack's canonical size when it has one; otherwise the model runs at
        # native resolution and the image's own pixel size is the model's.
        nm_per_model_px = spec.canonical_nm or (float(pixel_size) if pixel_size else None)
        span_nm = span_px * nm_per_model_px if nm_per_model_px else None
        needed_nm = required_px * nm_per_model_px if nm_per_model_px else None
        if model_area >= min_area:
            # One usable region is all head training needs.
            return HeadSizeVerdict(
                base_model=base_model,
                ok=True,
                largest_nm=span_nm,
                required_nm=needed_nm,
                largest_px=span_px,
                required_px=required_px,
                n_areas=len(crops),
                reason=None,
            )
        if best is None or span_px > best[0]:
            best = (span_px, required_px, span_nm, needed_nm)

    if best is None:
        return None

    span_px, _, span_nm, needed_nm = best
    subject = "Your checked area is" if len(crops) == 1 else "Your largest checked area is"
    if span_nm is not None and needed_nm is not None:
        reason = (
            f"{subject} {_micrometres(span_nm)} across; this needs about {_micrometres(needed_nm)}."
        )
    else:
        # No pixel size on the image, so the span cannot be stated as a length.
        # The comparison is still exact -- with no pixel size nothing is
        # resampled, so model pixels and image pixels are the same pixels.
        reason = (
            f"{subject} {_pixels(span_px)} across; this needs about "
            f"{_pixels(required_px)}. This image has no pixel size set, so the "
            "size can only be given in pixels."
        )
    return HeadSizeVerdict(
        base_model=base_model,
        ok=False,
        largest_nm=span_nm,
        required_nm=needed_nm,
        largest_px=span_px,
        required_px=required_px,
        n_areas=len(crops),
        reason=reason,
    )
