"""Native <-> canonical nm/px resampling.

Every released model was trained on data resampled to a per-organelle canonical
pixel size (mito 8 nm, lipid droplet 8 nm, nucleus 25 nm, ER native). Feeding a
2 nm/px asset straight to the 8 nm mito head shows it organelles four times
larger than anything it was trained on. So an asset's true ``pixel_size_nm``
gates inference, which is why it is a required numeric field rather than a
free-text resolution string.

Order of operations
-------------------
1. resample the image to the model's ``canonical_nm``  (:func:`to_model_scale`)
2. run inference and **threshold at model scale**
3. map the resulting binary mask back to native with NEAREST
   (:func:`mask_to_native`)

Thresholding first and upsampling the mask second is deliberate. Upsampling a
probability map and thresholding it afterwards interpolates *between* the
model's decisions and then re-decides on invented intermediate values; the
boundary you get is not one the model ever produced. Nearest-neighbour on the
mask keeps every pixel a decision the model actually made -- at the cost of
blocky edges at large upsample factors, which is the honest artifact.

Interpolator choice
-------------------
Downsampling uses ``cv2.INTER_AREA`` (area averaging): it integrates every
source pixel, so fine EM texture becomes signal rather than aliasing.
Upsampling uses ``cv2.INTER_LINEAR``. Labels and masks always use
``cv2.INTER_NEAREST``.

Note for reproducibility: the research pipeline that *built the training data*
(``fig3/dataprep/resample.py``) used ``scipy.ndimage.zoom(order=1)`` for EM and
``order=0`` for labels, i.e. bilinear in both directions with no area
averaging. For upsampling the two agree; for downsampling INTER_AREA is the
better antialiaser but is not bit-identical to how the training crops were
produced. Pass ``downscale_interpolation=cv2.INTER_LINEAR`` to reproduce the
training-time behaviour exactly.

Pure numpy + cv2. No torch, no Django.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: Factors within this of 1.0 are treated as a no-op, to avoid needless blur.
UNIT_EPS = 1e-3


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
    """Map a binary or label image from model pixels back to native pixels.

    NEAREST always: interpolating a mask invents boundary values, and
    interpolating instance ids invents instances.
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


def probability_to_native(prob: np.ndarray, context: ResampleContext) -> np.ndarray:
    """Map a probability map back to native pixels, bilinear.

    Use for display, for persisted probability-map artifacts, and for reading a
    per-object confidence at a native coordinate. **Do not threshold the
    result** -- threshold at model scale and call :func:`mask_to_native`, or the
    boundary is one the model never produced.
    """
    if context.is_identity:
        return prob
    height, width = context.native_shape
    out = cv2.resize(
        np.asarray(prob, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.ascontiguousarray(out, dtype=np.float32)
