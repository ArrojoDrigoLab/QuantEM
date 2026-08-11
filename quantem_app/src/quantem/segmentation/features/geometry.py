"""
Geometric feature computation using scikit-image regionprops.

This module provides utilities to compute standard geometric shape features
from binary masks, suitable for storing in SegmentObject.features.

Every value produced here is in PIXELS (area in px^2, lengths in px). Converting
them to physical units is the job of the analysis layer and requires
``Asset.pixel_size_nm``.

The mask this is handed comes from :mod:`quantem.seg_core.rasterize`, so
``area`` is the polygon's area: a pixel is counted when its centre is inside the
outline. A model object's mask and a drawn object's mask are the same pixels for
the same shape, which is what makes the two comparable in one ``objects.csv``.

``perimeter`` is a different measurement of that mask, and it is worth knowing
which. ``regionprops`` walks the *centres* of the boundary pixels, so it reports
the outline of the region **inset by half a pixel**: an n x n square measures
``4(n - 1)``, not ``4n``. That is scikit-image's definition, it is what a
model-extracted object has always carried, and both provenances now get it off
the same mask -- but it does not share ``area``'s convention, and the shortfall
is a fixed half-pixel all round against a length that grows with the object.

Combining the two therefore puts ``circularity`` (``4 pi A / P^2``) **above**
the continuous value, by more the smaller the object, and it converges on that
value **from above**. For an n x n square, against a true ``pi/4 = 0.785``::

    3 px 1.767   5 px 1.227   8 px 1.026   10 px 0.970
   20 px 0.870  50 px 0.818  80 px 0.805  100 px 0.801

Every one of those is *above* ``pi/4``, and the first three are above **1.0**,
which is circularity's ceiling: ``4 pi A / P^2 <= 1`` is the isoperimetric
inequality, so a value over it describes the estimator and never the object. A
round outline does the same -- a drawn disc measures 1.154 at r=3, 1.111 at r=5
and 1.050 at r=7 -- and the crossover sits inside the default ``min_area`` of
60 px, so an 8x8 compact object clears the floor and reports 1.026. The other
end is biased too: a large rasterised disc settles near 0.90, not the 1.0 a
circle is owed.

:mod:`quantem.analysis.morphometrics` is where that is handled for anything a
reader sees. It refuses to export a value above the ceiling and attaches the
direction of the bias and its size dependence to every value it does export --
see :data:`~quantem.analysis.morphometrics.CIRCULARITY_MAX`. The estimator
is ``perimeter_crofton``, by owner ruling 2026-08-07 -- see the comment at
the call site for the numbers behind the choice.

The bias is a property of digitised perimeter, not of one provenance, and no
available estimator is free of it. Measured on the same masks:
``perimeter_crofton`` is much better on discs (-0.5% at r=100 against -9.8%
for this one) but settles ~12-14% high on squares from 20 px up, and *still*
exceeds 1.0 on a drawn disc (1.020 at r=5, 1.025 at r=7, and 79 of 149 radii
swept from 3 to 40 px, though never by more than 1.062 against this
estimator's 1.247); circularity taken from the
stored polygon is the best on squares (+0.6% at 100 px) and the worst on discs
(-10.6%), and it would reopen the model-versus-hand-drawn split that measuring
both off one mask closed. Changing which one is reported would rewrite every
stored ``perimeter``, model and hand-drawn alike -- which is what
``manage.py remeasure_segment_features --apply`` exists to do, and what the
ruling accepted; nothing *automatically* re-measures an object that already
carries an ``area``
(``jobs.handlers._unmeasured_segment_ids``), so it is a decision about what the
paper reports -- to be taken deliberately and backfilled -- and not a
rasterisation bug.
"""

import logging
import time

import numpy as np
from scipy import ndimage
from skimage import measure

logger = logging.getLogger(__name__)


def polygon_area(contour: np.ndarray) -> float:
    """
    Compute the area of a polygon using the shoelace formula.

    Args:
        contour: Nx2 numpy array of polygon vertices (row, col) or (x, y) coordinates.
                 Can be open or closed (will be closed if needed).

    Returns:
        Area of the polygon (always positive).
    """
    if len(contour) < 3:
        return 0.0

    # Ensure contour is closed
    if len(contour.shape) == 2 and contour.shape[1] == 2:
        # Check if first and last points are the same
        if not np.allclose(contour[0], contour[-1]):
            # Close the polygon
            closed_contour = np.vstack([contour, contour[0:1]])
        else:
            closed_contour = contour
    else:
        return 0.0

    # Shoelace formula: area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    x = closed_contour[:, 1]  # column (x)
    y = closed_contour[:, 0]  # row (y)

    area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return float(area)


def compute_regionprops_features(mask: np.ndarray) -> dict[str, float]:
    """
    Compute geometric shape features from a binary mask using regionprops.

    This function extracts standard geometric features from a single-object
    binary mask, including area, perimeter, eccentricity, axis lengths, and
    Feret diameter.

    Args:
        mask: 2D numpy array representing a binary mask (boolean or uint8).
              Should contain a single object (connected component).
              If empty (no foreground pixels), returns an empty dict.

    Returns:
        Dictionary mapping feature names to float values, all in PIXELS. Keys include:
        - area: Number of pixels in the mask (int as float)
        - perimeter: Approximate perimeter in pixels
        - eccentricity: Eccentricity of the ellipse with same second moments (0-1)
        - major_axis_length: Length of major axis of fitted ellipse, in pixels
        - minor_axis_length: Length of minor axis of fitted ellipse, in pixels
        - feret_diameter_max: Maximum distance between boundary points, in pixels
        - solidity: Region area over its convex-hull area (0-1)
        - elongation: major_axis_length / minor_axis_length

        ``solidity`` and ``elongation`` are here so this returns the same shape
        descriptors as :func:`quantem.seg_core.extraction.build_segment_from_region`,
        which is what model-extracted objects get. Without them a hand-drawn
        object had blank ``solidity``/``elongation`` columns in ``objects.csv``
        next to a model-extracted one that had them.

    Raises:
        ValueError: If mask is not 2D or has invalid shape

    Example:
        >>> mask = np.zeros((100, 100), dtype=bool)
        >>> mask[40:60, 40:60] = True  # 20x20 square
        >>> features = compute_regionprops_features(mask)
        >>> features['area']
        400.0
    """
    # Validate input
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask array, got {mask.ndim}D array with shape {mask.shape}")

    # Convert to boolean if needed
    binary_mask = (mask > 127).astype(bool) if mask.dtype != bool else mask

    # Check if mask is empty
    if not np.any(binary_mask):
        return {}

    t0 = time.time()
    # Use regionprops to get geometric features
    # Label the mask to get a single region
    # Use scipy.ndimage.label for consistent 2-value return (labeled_array, num_features)
    labeled_mask, num_features = ndimage.label(binary_mask)
    t_label = time.time() - t0

    if num_features == 0:
        return {}

    t0 = time.time()
    # Get regionprops for the largest component (or first if only one)
    regions = measure.regionprops(labeled_mask)
    t_regionprops = time.time() - t0

    if len(regions) == 0:
        return {}

    t0 = time.time()
    # Use the largest region by area
    props = max(regions, key=lambda r: r.area)

    # Extract features directly from regionprops (most are built-in)
    area = float(props.area)
    # perimeter_crofton, not perimeter, by owner ruling 2026-08-07.
    #
    # Both are digitised-perimeter estimators and neither is exact, but they
    # fail differently and only one of the failures matters here.
    # `props.perimeter` walks boundary-pixel centres: on a rasterised disc it
    # gives a circularity of 1.227 at r=3 falling to 0.910 at r=80 -- a
    # monotone drift with object size. That does not cancel between groups, so
    # any treatment that changes organelle size produces a circularity effect
    # out of a correct segmentation. Measured on eight real mitochondrial
    # outlines scaled to 0.6x, a pure size change: mean circularity 0.619 ->
    # 0.641, paired t = 3.596, p = 0.0088.
    #
    # `perimeter_crofton` integrates over four directions and lands within ~1%
    # of the true 1.0 on discs from r=5 upward, essentially flat with size. It
    # is still biased on squares (~0.88 against pi/4), but by a roughly constant
    # factor, which does cancel. Organelles are blob-like, so the disc column is
    # the one that describes this application.
    #
    # Rejected: the stored polygon's own length. Exact on polygons, ~10% low on
    # anything round, and it answers differently by provenance -- a smooth
    # hand-drawn circle reads 1.0000 where the identical model-found circle,
    # whose polygon is a marching-squares staircase, reads 0.90. That is the
    # provenance split this codebase already removed once.
    #
    # Costs 12 us/object more; 0.012 s per 1000 objects.
    perimeter = (
        float(props.perimeter_crofton)
        if hasattr(props, "perimeter_crofton")
        else float(getattr(props, "perimeter", 0.0))
    )
    eccentricity = float(props.eccentricity) if hasattr(props, "eccentricity") else 0.0
    if hasattr(props, "axis_major_length"):
        major_axis_length = float(props.axis_major_length)
    elif hasattr(props, "major_axis_length"):
        major_axis_length = float(props.major_axis_length)
    else:
        major_axis_length = 0.0

    if hasattr(props, "axis_minor_length"):
        minor_axis_length = float(props.axis_minor_length)
    elif hasattr(props, "minor_axis_length"):
        minor_axis_length = float(props.minor_axis_length)
    else:
        minor_axis_length = 0.0
    feret_diameter_max = (
        float(props.feret_diameter_max) if hasattr(props, "feret_diameter_max") else 0.0
    )
    try:
        solidity = float(props.solidity)
    except Exception:
        # A degenerate hull (a 1 px wide region) makes solidity undefined;
        # seg_core.extraction falls back to 1.0 in the same place.
        solidity = 1.0
    # max(minor, 1) matches seg_core.extraction so the two paths agree for a
    # sub-pixel minor axis instead of one of them producing inf.
    elongation = float(major_axis_length / max(minor_axis_length, 1.0))
    t_extract = time.time() - t0

    total_time = t_label + t_regionprops + t_extract
    if total_time > 0.1:  # Only log if it takes more than 100ms
        logger.debug(
            f"compute_regionprops_features timing: "
            f"label={t_label * 1000:.1f}ms, "
            f"regionprops={t_regionprops * 1000:.1f}ms, "
            f"extract={t_extract * 1000:.1f}ms, "
            f"total={total_time * 1000:.1f}ms, "
            f"mask_shape={mask.shape}, area={area}"
        )

    # Build result dictionary
    features = {
        "area": area,
        "perimeter": perimeter,
        "eccentricity": eccentricity,
        "major_axis_length": major_axis_length,
        "minor_axis_length": minor_axis_length,
        "feret_diameter_max": feret_diameter_max,
        "solidity": solidity,
        "elongation": elongation,
    }

    return features
