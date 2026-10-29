"""Probability map -> instance labels.

The published post-processing chain:

    threshold -> closing(disk(r)) -> binary_fill_holes -> label -> drop
    regions below min_area

The closing consolidates a compact organelle that internal probability texture
has fragmented (a nucleus at r=12; ER, which is genuinely thin, at r=1), and the
hole fill closes the interior the closing left ringed. Both are per-organelle
constants, see :mod:`quantem.inference.specs`.

What this module is *not*: it is connected components, not instance
segmentation. Two mitochondria that touch stay one object. The released
``affinity_mws`` decoders emit affinities that a mutex watershed would split
properly; wiring that up needs the decoder head code and is tracked in
README.md.

Pure numpy + scipy + skimage. No torch, no Django.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as sk_label
from skimage.morphology import closing, disk

from quantem.registry.manifest import DEFAULT_THRESHOLD

#: Re-exported so ``from quantem.inference.postprocess import DEFAULT_THRESHOLD``
#: keeps working. There is exactly one definition of this number
#: (:data:`quantem.registry.manifest.DEFAULT_THRESHOLD`) because it is the
#: setting behind every benchmark in the manuscript: a second copy that drifted
#: would silently make a reproduced number incomparable to a published one.
__all__ = [
    "DEFAULT_THRESHOLD",
    "binarize",
    "close_and_fill",
    "filter_min_area",
    "label_instances",
    "postprocess_mask",
    "postprocess_probability",
]

#: Labeling connectivity. None means skimage's default, which for a 2-D input
#: is full connectivity (8-connected: diagonal touches merge). The reference
#: called ``skimage.measure.label(mask)`` bare, so this preserves its behaviour.
_CONNECTIVITY = None

#: Maximum temporary selector used while removing undersized labels.  The
#: previous vectorized expression materialized one bool per image pixel; on a
#: 10.8-gigapixel image that was another 10.1 GiB allocation after the label
#: image had already been copied.  Row blocks keep the same lookup and write
#: semantics while bounding that scratch space.
_FILTER_SELECTOR_TARGET_BYTES = 64 * 1024 * 1024


def binarize(prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Foreground mask from a probability map (``>= threshold``)."""
    return np.asarray(prob, dtype=np.float32) >= float(threshold)


def close_and_fill(mask: np.ndarray, close_radius: int) -> np.ndarray:
    """Morphological closing with a disk, then fill enclosed holes.

    A radius of 0 (or an empty mask) is a no-op.
    """
    if close_radius <= 0 or not mask.any():
        return mask
    closed = closing(mask, disk(int(close_radius)))
    return ndi.binary_fill_holes(closed)


def label_instances(mask: np.ndarray) -> np.ndarray:
    """Connected-component labels; 0 is background."""
    return sk_label(mask, connectivity=_CONNECTIVITY)


def filter_min_area(
    labels: np.ndarray,
    min_area: int,
    *,
    in_place: bool = False,
) -> np.ndarray:
    """Zero out components strictly smaller than ``min_area`` pixels.

    Operates on an already-labeled image, so the label ids of the survivors are
    preserved and the connected-component decision is not made twice.

    Counted here rather than with ``skimage.morphology.remove_small_objects``,
    which changed this very comparison. Historically it removed ``size <
    min_size``; on 0.26 ``min_size`` is deprecated and an object of *exactly*
    ``min_size`` pixels is now discarded. QuantEM's dependency floor is
    ``scikit-image>=0.22``, so two users on either side of that change would
    get different object counts from the same image and the same QuantEM, with
    nothing to point at: the library version is the only thing that moved.

    Nothing in the manuscript is at stake -- the published pipeline
    (:mod:`quantem.inference._fig3`) has no min-area step; this is an
    application-layer filter QuantEM adds. So the rule is ours to state, and it
    is the one the parameter's name implies: an object of exactly ``min_area``
    pixels meets the minimum and is kept. ``area < min_area`` is removed.

    Args:
        labels: Integer connected-component image.
        min_area: Components below this many pixels are removed.
        in_place: Reuse ``labels`` instead of copying it.  The post-processing
            pipeline owns its newly-created label image and enables this for
            large images; the public default preserves the historical
            non-mutating behavior.
    """
    if min_area <= 1:
        return labels
    counts = np.bincount(labels.ravel())
    too_small = np.flatnonzero(counts < int(min_area))
    if too_small.size == 0:
        return labels
    drop = np.zeros(counts.size, dtype=bool)
    drop[too_small] = True
    out = labels if in_place else labels.copy()

    # ``drop[out]`` is the desired selector, but evaluating it for the whole
    # image recreates the image-sized bool allocation this path exists to
    # avoid.  A row block is a view of ``out``; only its selector is temporary.
    pixels_per_row = max(int(np.prod(out.shape[1:], dtype=np.int64)), 1)
    rows_per_block = max(1, _FILTER_SELECTOR_TARGET_BYTES // pixels_per_row)
    for start in range(0, int(out.shape[0]), rows_per_block):
        block = out[start : start + rows_per_block]
        block[drop[block]] = 0
    return out


def postprocess_mask(
    mask: np.ndarray,
    *,
    close_radius: int = 0,
    min_area: int = 0,
) -> np.ndarray:
    """Clean a foreground mask and label it.

    Returns:
        Int label image; 0 is background.
    """
    cleaned = close_and_fill(mask, close_radius)
    # ``_foreground_mask`` normally hands this function a temporary full-image
    # bool array.  Once closing/filling produced a distinct result, retaining
    # that original through labeling and filtering costs another byte per
    # image pixel for no purpose.
    if cleaned is not mask:
        mask = None
    labels = label_instances(cleaned)
    cleaned = None
    mask = None
    return filter_min_area(labels, min_area, in_place=True)


def postprocess_probability(
    prob: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    close_radius: int = 0,
    min_area: int = 0,
) -> np.ndarray:
    """Threshold a probability map and turn it into instance labels.

    Args:
        prob: foreground probability in ``[0, 1]``.
        threshold: foreground decision boundary.
        close_radius: disk radius for the consolidating closing, in pixels of
            ``prob``'s own grid.
        min_area: drop components below this many pixels of ``prob``'s grid.

    Returns:
        Int label image the same shape as ``prob``; 0 is background.
    """
    return postprocess_mask(
        binarize(prob, threshold),
        close_radius=close_radius,
        min_area=min_area,
    )
