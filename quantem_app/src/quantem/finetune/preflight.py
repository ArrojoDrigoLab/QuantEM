"""Will this actually run? Answered before anything is queued.

Two of the three ways guided fine-tuning fails are knowable from data the crops
endpoint already has, and both used to be discovered minutes later by a job that
died in the queue:

* **Threshold calibration with no stored probability map.** There is nothing to
  sweep, so the job raises seconds in. :meth:`CropSet.mode_blockers` already
  computes this; the refusal simply has to happen at the door.
* **Head training on a checked area too small to cut a training window out of.**
  This one is pure geometry and is the subject of this module.

The geometry rule, derived rather than guessed
----------------------------------------------
:func:`quantem.finetune.adapt.build_patches` pads a crop up to one ``tile`` and
keeps a window only when at least ``min_valid_fraction`` of it is inside a
checked area -- the padding is marked invalid, so it contributes shape and
nothing else. A crop therefore yields **no training windows at all** unless

    (checked area, in model pixels) >= min_valid_fraction x tile x tile

That is a *necessary* condition for every crop shape, because any single
window's valid count is bounded by the crop's total valid area. Necessary, not
sufficient -- a long thin region can clear it on area and still have no window
that clears it -- which is the right direction for a refusal: it fires only when
head training is certain to produce zero steps.

For the shipped packs the number works out at ``tile x sqrt(0.2)``, i.e. about
229 model pixels for the 512 px packs and 232 for the 518 px ones. Expressed as
a physical span (``model pixels x canonical nm``) it becomes something a
microscopist can act on, which is the form the user is shown: *"Your checked
area is 1.1 µm across; this needs about 1.9 µm."*

Nothing here re-implements the resampling. :func:`quantem.inference.resample.
plan_resample` decides the model-scale shape for the real run and it decides it
here too, so the pre-flight cannot drift away from the trainer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantem.finetune.adapt import AdaptConfig, tile_for
from quantem.inference import resample
from quantem.inference.specs import MODEL_SPECS, ModelSpec

__all__ = [
    "HeadSizeVerdict",
    "check_head_size",
    "required_model_px",
]


def required_model_px(spec: ModelSpec, config: AdaptConfig = AdaptConfig()) -> float:
    """Side of the smallest square checked area head training can use, in model px.

    The rule is on area; this is its square root, which is what a length-based
    sentence has to quote. Deliberately not rounded -- the comparison is done on
    area, and rounding here would make the sentence and the verdict disagree at
    the boundary.
    """
    tile = tile_for(spec.tile_size, spec.patch_size)
    return float(tile) * math.sqrt(config.min_valid_fraction)


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

    tile = tile_for(spec.tile_size, spec.patch_size)
    min_area = AdaptConfig.min_valid_fraction * tile * tile
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
