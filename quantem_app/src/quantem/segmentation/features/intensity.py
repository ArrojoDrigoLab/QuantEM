"""
Intensity-based feature computation from grayscale images.

This module provides utilities to compute intensity statistics (mean, percentiles)
inside masks and in ring regions outside masks.
"""

import logging
import time

import numpy as np
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)


def compute_intensity_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Compute intensity percentiles and statistics inside a binary mask.

    This function extracts pixel intensity values from the image where the mask
    is True (foreground) and computes mean and percentile statistics.

    Args:
        image: 2D numpy array representing a grayscale image (uint8 or other numeric dtype).
               Must have the same shape as mask.
        mask: 2D numpy array representing a binary mask (boolean or uint8).
              Must have the same shape as image.

    Returns:
        Dictionary mapping feature names to float values. Keys include:
        - intensity_mean: Mean intensity inside mask
        - intensity_p10: 10th percentile intensity
        - intensity_p50: 50th percentile (median) intensity
        - intensity_p90: 90th percentile intensity

        If the mask is empty (no foreground pixels), returns an empty dict.

    Raises:
        ValueError: If image and mask have different shapes, or if mask is empty
                    (if you want to enforce non-empty masks, uncomment the check).

    Example:
        >>> image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        >>> mask = np.zeros((100, 100), dtype=bool)
        >>> mask[40:60, 40:60] = True
        >>> features = compute_intensity_features(image, mask)
        >>> 'intensity_mean' in features
        True
    """
    # Validate shapes match
    if image.shape != mask.shape:
        raise ValueError(
            f"Image and mask must have the same shape. "
            f"Got image {image.shape} and mask {mask.shape}"
        )

    # Convert mask to boolean if needed
    binary_mask = (mask > 127).astype(bool) if mask.dtype != bool else mask

    t0 = time.time()
    # Extract pixel values where mask is True
    vals = image[binary_mask]
    t_extract = time.time() - t0

    # Handle empty mask
    if len(vals) == 0:
        # Return empty dict for empty masks (documented behavior)
        return {}

    t0 = time.time()
    # Convert to float array for percentile computation
    vals_float = vals.astype(np.float64)

    # Compute statistics
    mean = np.mean(vals_float)
    p10 = np.percentile(vals_float, 10)
    p50 = np.percentile(vals_float, 50)
    p90 = np.percentile(vals_float, 90)
    t_stats = time.time() - t0

    total_time = t_extract + t_stats
    if total_time > 0.1:  # Only log if it takes more than 100ms
        logger.debug(
            f"compute_intensity_features timing: "
            f"extract={t_extract * 1000:.1f}ms, "
            f"stats={t_stats * 1000:.1f}ms, "
            f"total={total_time * 1000:.1f}ms, "
            f"mask_pixels={len(vals)}"
        )

    # Return as dictionary with Python floats
    return {
        "intensity_mean": float(mean),
        "intensity_p10": float(p10),
        "intensity_p50": float(p50),
        "intensity_p90": float(p90),
    }


def compute_outside_ring_intensity_features(
    image: np.ndarray, mask: np.ndarray, ring_pixels: int
) -> dict[str, float]:
    """
    Compute intensity percentiles in a ring region outside the mask.

    The ring is defined as pixels within `ring_pixels` distance from the mask
    boundary but not inside the mask itself. This is useful for computing
    background/context intensity features around segments.

    Args:
        image: 2D numpy array representing a grayscale image (uint8 or other numeric dtype).
               Must have the same shape as mask.
        mask: 2D numpy array representing a binary mask (boolean or uint8).
              Must have the same shape as image.
        ring_pixels: Integer specifying the thickness of the ring in pixels.
                     Pixels within this distance from the mask boundary (but outside
                     the mask) are included in the ring.

    Returns:
        Dictionary mapping feature names to float values. Keys include:
        - outside_intensity_mean: Mean intensity in the ring
        - outside_intensity_p10: 10th percentile intensity in the ring
        - outside_intensity_p50: 50th percentile (median) intensity in the ring
        - outside_intensity_p90: 90th percentile intensity in the ring

        If the ring is empty (e.g., mask covers entire image or ring_pixels is too large),
        returns an empty dict.

    Raises:
        ValueError: If image and mask have different shapes, or if ring_pixels <= 0

    Example:
        >>> image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        >>> mask = np.zeros((100, 100), dtype=bool)
        >>> mask[40:60, 40:60] = True
        >>> features = compute_outside_ring_intensity_features(image, mask, ring_pixels=10)
        >>> 'outside_intensity_mean' in features
        True
    """
    # Validate inputs
    if image.shape != mask.shape:
        raise ValueError(
            f"Image and mask must have the same shape. "
            f"Got image {image.shape} and mask {mask.shape}"
        )

    if ring_pixels <= 0:
        raise ValueError(f"ring_pixels must be positive, got {ring_pixels}")

    # Convert mask to boolean if needed
    binary_mask = (mask > 127).astype(bool) if mask.dtype != bool else mask

    t0 = time.time()
    # Compute dilated mask to get the ring region
    # Dilate the mask by ring_pixels iterations
    dilated_mask = binary_dilation(binary_mask, iterations=ring_pixels)
    t_dilate = time.time() - t0

    t0 = time.time()
    # Ring is pixels in dilated mask but not in original mask
    ring_mask = dilated_mask & ~binary_mask

    # Extract pixel values in the ring
    ring_vals = image[ring_mask]
    t_extract = time.time() - t0

    # Handle empty ring
    if len(ring_vals) == 0:
        # Return empty dict for empty rings (documented behavior)
        return {}

    t0 = time.time()
    # Convert to float array for percentile computation
    ring_vals_float = ring_vals.astype(np.float64)

    # Compute statistics
    mean = np.mean(ring_vals_float)
    p10 = np.percentile(ring_vals_float, 10)
    p50 = np.percentile(ring_vals_float, 50)
    p90 = np.percentile(ring_vals_float, 90)
    t_stats = time.time() - t0

    total_time = t_dilate + t_extract + t_stats
    if total_time > 0.1:  # Only log if it takes more than 100ms
        logger.debug(
            f"compute_outside_ring_intensity_features timing: "
            f"dilate={t_dilate * 1000:.1f}ms, "
            f"extract={t_extract * 1000:.1f}ms, "
            f"stats={t_stats * 1000:.1f}ms, "
            f"total={total_time * 1000:.1f}ms, "
            f"ring_pixels={ring_pixels}, ring_pixels_count={len(ring_vals)}"
        )

    # Return as dictionary with Python floats
    return {
        "outside_intensity_mean": float(mean),
        "outside_intensity_p10": float(p10),
        "outside_intensity_p50": float(p50),
        "outside_intensity_p90": float(p90),
    }
