"""Probability map -> labelled objects.

Threshold, binary closing, fill holes, connected components, drop small objects.

v1 ships connected components only. Mutex watershed is deliberately omitted: ``affogato`` was never
installed on any campaign box, so ``mutex_watershed_postproc`` always fell back to connected
components and **no published number was ever produced with real MWS**. CC is what reproduces the
paper.
"""

from __future__ import annotations

import numpy as np


def threshold(probability: np.ndarray, value: float) -> np.ndarray:
    return np.asarray(probability) >= float(value)


def clean_mask(mask: np.ndarray, *, close_radius: int = 0, fill_holes: bool = True) -> np.ndarray:
    """Morphological tidy-up before labelling.

    Closing goes through skimage, not scipy, and the difference is not cosmetic. ``ndi.binary_
    closing`` treats out-of-bounds as background during its erosion step, so it shaves a
    ``close_radius``-wide band off every object touching the image border. Measured on a full-
    foreground 100x100 field: scipy keeps 5776 px at the nucleus radius of 12 where skimage keeps
    all 10000 -- a 42 % loss, concentrated in exactly the objects a tiled or cropped EM workflow
    produces most. ``segmenter.py:201`` uses ``skimage.morphology.binary_closing``.
    """
    from scipy import ndimage as ndi

    m = np.asarray(mask, dtype=bool)
    if close_radius and close_radius > 0:
        m = _closing(m, _disk(close_radius))
    if fill_holes:
        m = ndi.binary_fill_holes(m)
    return m


def _closing(mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    from skimage import morphology as morph

    # `binary_closing` is deprecated in scikit-image 0.26 in favour of `closing`, which is
    # identical for a boolean input. Prefer the new name, fall back for older installs.
    fn = getattr(morph, "closing", None) or morph.binary_closing
    return np.asarray(fn(mask, footprint), dtype=bool)


def _disk(radius: int) -> np.ndarray:
    """Bit-identical to ``skimage.morphology.disk`` -- verified for r = 1, 2, 3 and 12."""
    r = int(radius)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def label_objects(mask: np.ndarray, *, min_area: int = 0) -> tuple[np.ndarray, int]:
    """Connected components with a minimum area, relabelled 1..N with 0 = background.

    **8-connected**, because the reference calls ``skimage.measure.label(mask)`` bare
    (``segmenter.py:203``) and skimage's 2-D default is full connectivity. ``scipy.ndimage.label``
    defaults to a 4-connected cross instead, which splits every diagonal one-pixel bridge into a
    separate object -- inflating the count and then deleting the fragments through ``min_area``,
    so both ``n_objects`` and the surviving mask change. The sibling implementation in
    ``quantem_app`` documents the same choice.
    """
    from skimage.measure import label as sk_label

    lab = sk_label(np.asarray(mask, dtype=bool)).astype(np.int32)
    n = int(lab.max())
    if n == 0:
        return lab, 0
    if min_area and min_area > 0:
        counts = np.bincount(lab.ravel())
        too_small = np.flatnonzero(counts < int(min_area))
        if too_small.size:
            drop = np.zeros(counts.size, dtype=bool)
            drop[too_small] = True
            drop[0] = True  # background is not an object
            lab[drop[lab]] = 0
        # compact ids so labels run 1..N with no gaps
        keep = np.flatnonzero(np.bincount(lab.ravel(), minlength=counts.size) > 0)
        keep = keep[keep != 0]
        remap = np.zeros(int(lab.max()) + 1, dtype=np.int32)
        remap[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
        lab = remap[lab]
        n = int(keep.size)
    return lab.astype(np.int32), int(n)


def segment_from_probability(
    probability: np.ndarray,
    *,
    fg_threshold: float,
    close_radius: int,
    min_area: int,
    fill_holes: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``probability -> (labels int32, mask bool, n_objects)``."""
    mask = clean_mask(
        threshold(probability, fg_threshold), close_radius=close_radius, fill_holes=fill_holes
    )
    labels, n = label_objects(mask, min_area=min_area)
    return labels, labels > 0, n


def link_across_z(volume_mask: np.ndarray, *, min_area: int = 0) -> tuple[np.ndarray, int]:
    """Label a stacked binary volume in 3-D.

    This is **not** 3-D instance segmentation: the models are 2-D and each slice was segmented
    independently. It simply joins slice-wise foreground that overlaps in z, which is useful for
    counting but should not be presented as volumetric instance segmentation.
    """
    from skimage.measure import label as sk_label

    # Full 26-connectivity, matching the 2-D choice in label_objects: a 6-connected cross would
    # break a track wherever an object shifts diagonally by one pixel between slices.
    lab = sk_label(np.asarray(volume_mask, dtype=bool)).astype(np.int32)
    n = int(lab.max())
    if n and min_area:
        counts = np.bincount(lab.ravel())
        drop = counts < int(min_area)
        drop[0] = True
        lab[drop[lab]] = 0
        keep = np.flatnonzero(np.bincount(lab.ravel(), minlength=counts.size) > 0)
        keep = keep[keep != 0]
        remap = np.zeros(int(lab.max()) + 1, dtype=np.int32)
        remap[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
        lab = remap[lab]
        n = int(keep.size)
    return lab.astype(np.int32), int(n)
