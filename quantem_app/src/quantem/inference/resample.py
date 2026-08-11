"""Native <-> canonical nm/px resampling, and the native-coordinate probability map.

Every released model was trained on data resampled to a per-organelle canonical
pixel size (mito 8 nm, lipid droplet 8 nm, nucleus 25 nm, ER native). Feeding a
2 nm/px asset straight to the 8 nm mito head shows it organelles four times
larger than anything it was trained on. So an asset's true ``pixel_size_nm``
gates inference, which is why it is a required numeric field rather than a
free-text resolution string.

Order of operations
-------------------
1. resample the image to the model's ``canonical_nm``  (:func:`to_model_scale`)
2. run inference on that grid
3. **resample the probability map back to the image's own pixel grid**
   (:func:`probability_to_native`) and **quantise it to uint8**
   (:func:`quantize_probability`); the pair is :func:`probability_to_native_uint8`
4. **threshold that stored uint8 map, in native coordinates**
   (:func:`binarize_quantized`)
5. extract objects and apply every object-level filter -- closing radius,
   minimum area, everything -- in native coordinates

The probability field crosses back to native pixels *before* any decision is
made about it. That is the whole rule, and everything below follows from it.

Why the field comes back, not the mask
--------------------------------------
The stored native-coordinate probability map is a **substrate for later
refinement**. A finer correction model can combine it with native image
features and with known organelle-shape priors to improve a boundary, because
the confidence structure is still there to combine with. A binary mask
upsampled with nearest-neighbour has already thrown that away: every pixel of
the k x k block it replicated carries the same single bit, and no later stage
can recover which parts of that block the model was actually sure about. The
option is preserved by keeping the continuous field; thresholding early
forecloses it permanently. That is a statement about what the system can become,
and it is why this order is the one the project takes.

It is also the only order under which a **threshold dial** can exist. Both a
fresh run and a later dial movement threshold the *same stored uint8 array*
with the same code, so re-thresholding is arithmetically the same operation as
the threshold step of the run that preceded it. There is no float-versus-
quantised and no model-grid-versus-native discrepancy left to drift.

This reverses an earlier, deliberate decision
---------------------------------------------
This module previously argued the opposite order -- threshold on the model's
own grid, bring the *binary mask* back with nearest-neighbour -- on the grounds
that interpolating probabilities and then re-deciding invents boundary values
the model never produced. That argument is real, and it was measured rather
than waved away:

* **Ordering-A is ordering-B with a nearest interpolator.** At integer
  back-factors the two produce byte-identical results -- same foreground pixel
  count, same object count, same perimeter. The entire question reduces to
  which interpolator carries the field back.
* **Dice barely moves.** Over 24 real cases with ground truth (back-factors
  0.50x to 5.41x, four organelles, three thresholds), the largest change
  anywhere is **0.0077 Dice**, median -0.00016 at t = 0.5, sign mixed. Object
  counts are unchanged in 17 of 24 cases and move by at most 4.
* **Geometry is what changes, and it changes towards the annotation.** Before
  the closing, median Crofton perimeter falls 5 % at a 1.33x back-factor, 9-10 %
  at 2x and 12-14 % at 4-5x (fit: ``dP = -2.88 % * ln(back) - 6.46 %``), and
  circularity rises 0.02-0.16. Measured against ground truth on the same native
  grid, ordering B's circularity is **closer to the truth in 12 of 12
  upsample-back cases**. The staircase the old note called "the honest artifact"
  was inflating perimeter on exactly the metric this project moved to a Crofton
  estimator to protect.
* **What reaches a published number is smaller**, because the per-organelle
  closing launders most of it: mitochondria (r = 3) keep **-2.6 % to -5.2 %**
  perimeter and **+0.03 to +0.05** circularity, while nucleus (r = 12) and
  lipid droplet (r = 2) move by well under 1 %. Total area falls 0.02-1.3 %.
  Old and new segmentations are therefore *not* the same computation and must
  not be pooled silently. Nothing already stored became wrong; it came from a
  different, documented pipeline.

Interpolator choice
-------------------
``to_model_scale`` (image, native -> model): ``cv2.INTER_AREA`` down,
``cv2.INTER_LINEAR`` up. Area averaging integrates every source pixel, so fine
EM texture becomes signal rather than aliasing.

``probability_to_native`` (field, model -> native) applies the **same policy in
the other direction**: ``cv2.INTER_LINEAR`` when the native grid is finer than
the model grid (the common case -- any image finer than the head's canonical
nm), ``cv2.INTER_AREA`` when it is coarser, and the result is clipped back into
``[0, 1]``.

  **Do not "just use INTER_AREA".** When upsampling, OpenCV's ``INTER_AREA``
  falls back to nearest-neighbour and returns byte-identical output to
  ``cv2.INTER_NEAREST`` -- measured on real maps: nucleus 5x, 5 540 475
  foreground px and mean circularity 0.5478 under both; mito 4x, 200.3 um
  perimeter under both against 191.6 um for bilinear. Naming "area" for the
  resample-back would silently reinstate the exact staircase artifact this
  ordering removes, while the code and the provenance both claimed a continuous
  interpolator. :func:`probability_interpolation` exists so that branch is
  written once and recorded.

``mask_to_native`` still uses ``INTER_NEAREST`` and always will -- interpolating
instance ids invents instances -- but it is no longer on the foreground path.

Quantisation
------------
The stored map is uint8, 255 levels, **round-to-nearest**
(:func:`quantize_probability`), and a threshold ``t`` becomes the integer level
``k = floor(255t + 1)`` (:func:`threshold_level`), which cuts at
``(k - 0.5) / 255`` (:func:`realised_threshold`). At the product default
``t = 0.5`` that is 127.5/255, i.e. exactly 0.5, so thresholding the uint8 map
and thresholding the float it came from give **the same pixels**; measured
across 27 images from 0.59 MP to 41 MP, zero pixels differ. Elsewhere on the
dial the realised cut is within 1/510 of the requested one, which costs at most
one object -- less than one step of the dial's own 1/255 resolution.

A truncating ``(p * 255).astype(uint8)`` does **not** have that property: it
biases every value down by up to 1/255 and flips up to 13 956 pixels (and one
object) at t = 0.5. Use :func:`quantize_probability`.

Pure numpy + cv2. No torch, no Django.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

#: Factors within this of 1.0 are treated as a no-op, to avoid needless blur.
UNIT_EPS = 1e-3

#: Interpolators for carrying a *continuous* probability field from the model's
#: grid back to native pixels. Never ``INTER_NEAREST`` (that is ordering A), and
#: never ``INTER_AREA`` in the upsample direction (OpenCV degrades it to
#: nearest); see the module docstring.
PROBABILITY_UPSCALE_INTERPOLATION = cv2.INTER_LINEAR
PROBABILITY_DOWNSCALE_INTERPOLATION = cv2.INTER_AREA

#: Levels in the stored map. uint8 holds 0..255; ``PROB_LEVELS`` is the
#: multiplier, so probability ``p`` stores as ``round(255 p)``.
PROB_LEVELS = 255

#: Name recorded in provenance for the value stored in each byte.
QUANTIZATION_ID = "uint8_255_round"

_INTERPOLATION_NAMES: dict[int, str] = {
    cv2.INTER_NEAREST: "INTER_NEAREST",
    cv2.INTER_LINEAR: "INTER_LINEAR",
    cv2.INTER_CUBIC: "INTER_CUBIC",
    cv2.INTER_AREA: "INTER_AREA",
    cv2.INTER_LANCZOS4: "INTER_LANCZOS4",
}

#: What :func:`interpolation_name` reports when no resampling happened. It is a
#: distinct answer from any interpolator: "the model predicted on this grid".
NO_RESAMPLE = "none"


def interpolation_name(interpolation: int | None) -> str:
    """Human- and machine-readable name of an OpenCV interpolation flag.

    ``None`` -> :data:`NO_RESAMPLE`. Used for provenance, so an unknown flag is
    reported as its number rather than silently as something else.
    """
    if interpolation is None:
        return NO_RESAMPLE
    return _INTERPOLATION_NAMES.get(int(interpolation), f"cv2:{int(interpolation)}")


def resample_factor(pixel_size_nm: float | None, canonical_nm: float | None) -> float:
    """Scale from native pixels to model pixels.

    ``factor > 1`` upsamples (the asset is coarser than canonical), ``< 1``
    downsamples. Returns 1.0 when either size is unknown or the model is
    native-resolution (``canonical_nm is None``, i.e. ER).
    """
    if not pixel_size_nm or not canonical_nm:
        return 1.0
    if pixel_size_nm <= 0 or canonical_nm <= 0:
        return 1.0
    return float(pixel_size_nm) / float(canonical_nm)


@dataclass(frozen=True)
class ResampleContext:
    """Everything needed to go native -> model and back for one region."""

    factor: float
    native_shape: tuple[int, int]
    model_shape: tuple[int, int]

    @property
    def is_identity(self) -> bool:
        return self.native_shape == self.model_shape

    @property
    def upsamples(self) -> bool:
        return self.factor > 1.0

    @property
    def back_factor(self) -> float:
        """Linear scale from the model grid to the native grid.

        ``> 1`` means the native grid is finer than the model's, so the
        probability map is **upsampled** on its way back -- the common case, and
        the one where the choice of interpolator matters (a 4 nm image on the
        8 nm mito head, a 5 nm image on the 25 nm nucleus head). ``< 1`` means
        the map is averaged down on the way back.
        """
        model_h, model_w = self.model_shape
        native_h, native_w = self.native_shape
        if model_h <= 0 or model_w <= 0:
            return 1.0
        return math.sqrt((native_h * native_w) / float(model_h * model_w))

    @property
    def back_upsamples(self) -> bool:
        """True when the resample *back* to native enlarges the map.

        Derived from the two shapes rather than from ``factor``, because the
        shapes are what ``cv2.resize`` is actually given and the rounding in
        :func:`plan_resample` can put a factor of exactly 1.0 either side of a
        one-pixel difference.
        """
        native_h, native_w = self.native_shape
        model_h, model_w = self.model_shape
        return (native_h * native_w) > (model_h * model_w)

    def scale_length(self, native_px: float) -> float:
        """Convert a native-pixel length (e.g. a closing radius) to model pixels."""
        return float(native_px) * self.factor

    def scale_area(self, native_px: float) -> float:
        """Convert a native-pixel area (e.g. min_area) to model pixels."""
        return float(native_px) * self.factor * self.factor


def plan_resample(
    native_shape: tuple[int, int],
    pixel_size_nm: float | None,
    canonical_nm: float | None,
) -> ResampleContext:
    """Compute the model-scale shape for a region, or an identity context."""
    factor = resample_factor(pixel_size_nm, canonical_nm)
    h, w = int(native_shape[0]), int(native_shape[1])
    if abs(factor - 1.0) < UNIT_EPS:
        return ResampleContext(1.0, (h, w), (h, w))
    model_shape = (max(1, int(round(h * factor))), max(1, int(round(w * factor))))
    return ResampleContext(factor, (h, w), model_shape)


def to_model_scale(
    image: np.ndarray,
    context: ResampleContext,
    *,
    downscale_interpolation: int = cv2.INTER_AREA,
    upscale_interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resample a grayscale EM image from native pixels to model pixels.

    Returns the input unchanged for an identity context (no interpolation, no
    copy, no blur).

    Note for reproducibility: the research pipeline that *built the training
    data* used ``scipy.ndimage.zoom(order=1)`` for EM and ``order=0`` for
    labels, i.e. bilinear in both directions with no area averaging. For
    upsampling the two agree; for downsampling ``INTER_AREA`` is the better
    antialiaser but is not bit-identical to how the training crops were
    produced. Pass ``downscale_interpolation=cv2.INTER_LINEAR`` to reproduce the
    training-time behaviour exactly.
    """
    if context.is_identity:
        return image
    interpolation = (
        upscale_interpolation if context.upsamples else downscale_interpolation
    )
    height, width = context.model_shape
    out = cv2.resize(image, (width, height), interpolation=interpolation)
    if image.dtype == np.uint8 and out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def mask_to_native(mask: np.ndarray, context: ResampleContext) -> np.ndarray:
    """Map a **label or instance** image from model pixels back to native pixels.

    NEAREST always: interpolating instance ids invents instances.

    This is no longer the foreground path. Bringing a *binary* mask back this
    way is ordering A, and ordering A is exactly this function's interpolator
    applied to the probability field instead -- which is what the tests use it
    to demonstrate. A foreground decision is made by
    :func:`binarize_quantized` on the stored native map.
    """
    if context.is_identity:
        return mask
    height, width = context.native_shape
    source = mask
    cast_back = None
    if source.dtype == np.bool_:
        source = source.astype(np.uint8)
        cast_back = np.bool_
    elif source.dtype not in (np.uint8, np.uint16, np.int32, np.float32):
        cast_back = source.dtype
        source = source.astype(np.int32)
    out = cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST)
    if cast_back is not None:
        out = out.astype(cast_back)
    return np.ascontiguousarray(out)


def probability_interpolation(context: ResampleContext) -> int | None:
    """The interpolator :func:`probability_to_native` will use, or ``None``.

    ``None`` means the map is already in native coordinates and nothing is
    resampled. Exposed separately from the resample itself so provenance records
    what actually happened rather than a restatement of the policy.
    """
    if context.is_identity:
        return None
    return (
        PROBABILITY_UPSCALE_INTERPOLATION
        if context.back_upsamples
        else PROBABILITY_DOWNSCALE_INTERPOLATION
    )


def probability_to_native(prob: np.ndarray, context: ResampleContext) -> np.ndarray:
    """Carry a probability map from the model's grid to the image's own pixels.

    **This is the one function that crosses that boundary**, so the interpolator
    is chosen in one place and can be recorded (see
    :func:`probability_interpolation`): ``INTER_LINEAR`` upsampling back,
    ``INTER_AREA`` downsampling back, never ``INTER_NEAREST``.

    The result is a float32 field. Quantise it with
    :func:`quantize_probability` before thresholding, or call
    :func:`probability_to_native_uint8` which does both -- the stored uint8 map
    is what a threshold reads, and a fresh run must not threshold this float.

    An identity context returns the model's own field with no copy and no clip:
    it is already a probability, and :func:`quantize_probability` clips anyway.
    A resampled result *is* clipped, because interpolation between values in
    ``[0, 1]`` can leave the interval by a rounding step.
    """
    field = np.asarray(prob, dtype=np.float32)
    if context.is_identity:
        return field
    height, width = context.native_shape
    out = cv2.resize(
        field,
        (width, height),
        interpolation=probability_interpolation(context),
    )
    np.clip(out, 0.0, 1.0, out=out)
    return np.ascontiguousarray(out, dtype=np.float32)


#: Values quantised per pass. Bounds the float64 working buffer to ~32 MB so a
#: 50+ MP map does not double its own footprint to be stored.
_QUANTIZE_CHUNK = 1 << 22


def quantize_probability(prob: np.ndarray) -> np.ndarray:
    """Quantise a ``[0, 1]`` field to the stored uint8 map, round-to-nearest.

    ``floor(255 p + 0.5)``, not ``(255 p).astype(uint8)``. The truncating cast
    is a biased quantiser -- it always rounds down, so the stored value sits up
    to 1/255 below the truth -- and that bias is what makes thresholding the
    stored map disagree with thresholding the float it came from: measured, up
    to 13 956 pixels and one object at the default threshold. Round-to-nearest
    makes the two agree exactly there.

    The arithmetic is done in float64. ``255 * p`` for a float32 ``p`` needs 32
    mantissa bits and is therefore *not* exact in float32, which would let a
    value one ULP from a quantisation boundary land on the wrong level; in
    float64 it is exact, so the level a probability stores at is a property of
    the number and not of the order of operations. Half-way values round up.
    """
    array = np.atleast_1d(np.asarray(prob))
    out = np.empty(array.shape, dtype=np.uint8)
    if array.size:
        rows = max(1, _QUANTIZE_CHUNK // max(int(array.shape[-1]), 1))
        for start in range(0, int(array.shape[0]), rows):
            # `copy=True`: a float64 input would otherwise be a view, and the
            # in-place arithmetic below would rewrite the caller's array.
            block = np.array(
                array[start : start + rows], dtype=np.float64, copy=True
            )
            np.clip(block, 0.0, 1.0, out=block)
            block *= PROB_LEVELS
            block += 0.5
            np.floor(block, out=block)
            out[start : start + rows] = block.astype(np.uint8)
    return out.reshape(np.shape(prob))


def dequantize_probability(stored: np.ndarray) -> np.ndarray:
    """The stored uint8 map read back as float32 probabilities in ``[0, 1]``.

    For display, for per-object confidence, and for the ``[0, 1]`` array the
    :class:`~quantem.seg_core.types.InferenceResult` contract promises. It is
    **not** what a threshold reads -- see :func:`binarize_quantized`.
    """
    return np.asarray(stored, dtype=np.uint8).astype(np.float32) / np.float32(
        PROB_LEVELS
    )


def probability_to_native_uint8(
    prob: np.ndarray, context: ResampleContext
) -> np.ndarray:
    """Step 3 of the pipeline in one call: back to native, then to uint8.

    The returned array is *the authority* for this image: the run thresholds it,
    it is what gets written to disk, and a later dial movement thresholds the
    same bytes. Nothing downstream may threshold the float this came from.
    """
    return quantize_probability(probability_to_native(prob, context))


def threshold_level(threshold: float) -> int:
    """The stored-map level a requested probability threshold becomes.

    ``k = floor(255 t + 1)``, so that ``stored >= k`` is ``p >= (k - 0.5)/255``
    under round-to-nearest quantisation. Returns ``0`` for ``t <= 0`` (keep
    everything) and ``256`` for a threshold no stored level can reach (keep
    nothing) rather than pretending either is representable.
    """
    value = float(threshold)
    if not (value == value):  # NaN
        raise ValueError("threshold must be a number")
    if value <= 0.0:
        return 0
    level = int(math.floor(PROB_LEVELS * value + 1.0))
    return min(max(level, 0), PROB_LEVELS + 1)


def realised_threshold(level: int) -> float:
    """The probability cut a stored-map level actually applies.

    ``(k - 0.5)/255``: the midpoint between the two float values that quantise
    either side of ``k``. Equal to the requested threshold whenever
    ``255 t + 0.5`` is an integer -- which includes the product default 0.5, and
    0.1/0.3/0.7/0.9 -- and within 1/510 of it otherwise. Record it beside the
    requested threshold, because they are not always the same number.
    """
    k = int(level)
    if k <= 0:
        return 0.0
    if k > PROB_LEVELS:
        return 1.0
    return (k - 0.5) / float(PROB_LEVELS)


def binarize_quantized(stored: np.ndarray, threshold: float) -> np.ndarray:
    """Foreground mask from the stored uint8 map, in native coordinates.

    The one place a foreground decision is made. Both a fresh run and a dial
    movement call this on the same bytes with the same level, which is what
    makes replay exact by construction rather than by careful matching.
    """
    array = np.asarray(stored, dtype=np.uint8)
    level = threshold_level(threshold)
    if level <= 0:
        return np.ones(array.shape, dtype=bool)
    if level > PROB_LEVELS:
        return np.zeros(array.shape, dtype=bool)
    return array >= np.uint8(level)


@dataclass(frozen=True)
class NativeProbabilityMap:
    """A probability map in the image's own pixel coordinates, uint8.

    **This is the authority for one region of one image.** The run that made it
    thresholds it, it is what is written to disk, and a later threshold movement
    thresholds the same bytes. Nothing may threshold the float it was built
    from: that float is on the model's grid, or is an unquantised copy, and
    either one re-decides pixels the stored map has already decided.

    ``interpolation`` and ``back_factor`` describe the crossing from the model's
    grid to this one, and travel into provenance with the threshold -- because
    "which interpolator" is a real degree of freedom in the result, not an
    implementation detail (``INTER_AREA`` upsampling would silently reproduce a
    nearest-neighbour staircase; see the module docstring).
    """

    data: np.ndarray
    interpolation: str = NO_RESAMPLE
    back_factor: float = 1.0
    quantization: str = QUANTIZATION_ID

    @classmethod
    def from_model_grid(
        cls, prob: np.ndarray, context: ResampleContext
    ) -> NativeProbabilityMap:
        """Build the stored map from what the model predicted. Steps 3 and 4."""
        return cls(
            data=probability_to_native_uint8(prob, context),
            interpolation=interpolation_name(probability_interpolation(context)),
            back_factor=float(context.back_factor),
        )

    @classmethod
    def from_stored(cls, stored: np.ndarray, **provenance: object) -> NativeProbabilityMap:
        """Re-adopt bytes that were previously stored, for replay.

        No requantisation: the bytes are taken as they are, which is the whole
        point -- re-deriving them from a float would be a second, different
        decision.
        """
        array = np.asarray(stored)
        if array.dtype != np.uint8:
            raise TypeError(
                f"a stored probability map is uint8; got {array.dtype}. Quantise "
                "with quantize_probability() rather than casting."
            )
        return cls(
            data=array,
            interpolation=str(provenance.get("interpolation") or NO_RESAMPLE),
            back_factor=float(provenance.get("back_factor") or 1.0),
            quantization=str(provenance.get("quantization") or QUANTIZATION_ID),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    def as_float(self) -> np.ndarray:
        """``[0, 1]`` float32 view of the stored levels, for display and export."""
        return dequantize_probability(self.data)

    def foreground(self, threshold: float) -> np.ndarray:
        """Native-coordinate foreground mask at ``threshold``."""
        return binarize_quantized(self.data, threshold)

    def provenance(self, threshold: float) -> dict[str, object]:
        """What has to be recorded beside the map: how it got here, and the cut.

        ``threshold`` is what was asked for; ``realised_threshold`` is what the
        255-level map could actually apply. They differ by at most 1/510 and are
        equal at the product default, but recording only the request would
        describe a cut that was not made.
        """
        level = threshold_level(threshold)
        return {
            "native_coordinates": True,
            "resample_interpolation": self.interpolation,
            "resample_back_factor": round(float(self.back_factor), 6),
            "quantization": self.quantization,
            "quantization_levels": PROB_LEVELS,
            "threshold": float(threshold),
            "threshold_level": level,
            "realised_threshold": realised_threshold(level),
            "thresholded_on": "stored_native_uint8",
        }
