"""
Utility functions for segmentation app.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
from shapely.geometry import Polygon
from skimage import measure, transform

# Disable decompression bomb check for large SEM/TEM images
Image.MAX_IMAGE_PIXELS = None

from .models import SegmentationType
from .type_service import (
    get_or_create_lipid_droplet_type as _get_or_create_lipid_droplet_type,
)
from .type_service import (
    get_or_create_mitochondria_type as _get_or_create_mitochondria_type,
)
from .type_service import (
    get_or_create_nucleus_type as _get_or_create_nucleus_type,
)

logger = logging.getLogger(__name__)


def _rasterize_polygon(
    path_coords: list[tuple[float, float]], height: int, width: int
) -> np.ndarray:
    """Rasterize a polygon (x, y) ring to a mask that is deliberately generous.

    **Not a measurement.** ``cv2.fillPoly`` rounds each vertex to a pixel centre
    and paints both boundaries of every span, so this mask is the polygon plus
    up to a pixel all round. That is what its two callers want and neither
    counts the pixels:

    * :mod:`quantem.segmentation.edge_constraints` asks whether a point the user
      clicked is inside a candidate outline, and a click on the edge of the
      thing should read as inside -- the docstring there says so.
    * :mod:`quantem.segmentation.edge_utils` uses it as the working mask for
      edge refinement, an operation that then re-derives the polygon.

    Anything that *counts* pixels -- object features, the overlay label map the
    analysis compartments are read from, the fine-tuning ground truth -- goes
    through :mod:`quantem.seg_core.rasterize` instead, where a pixel is the
    shape's when its centre is inside it and the pixel count is the polygon's
    area. Do not reach for this one to measure something.

    Note: ``matplotlib.path.Path.contains_points`` over a full pixel grid would
    pull in matplotlib (~40 MB) for one class and is O(H*W) per polygon.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(np.asarray(path_coords, dtype=np.float64)).astype(np.int32)
    if pts.shape[0] < 3:
        return mask.astype(bool)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], color=1)
    return mask.astype(bool)


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


def get_or_create_mitochondria_type() -> SegmentationType:
    """Backward-compatible wrapper for canonical mito type creation."""
    return _get_or_create_mitochondria_type()


def get_or_create_nucleus_type() -> SegmentationType:
    """Backward-compatible wrapper for canonical nucleus type creation."""
    return _get_or_create_nucleus_type()


def get_or_create_lipid_droplet_type() -> SegmentationType:
    """Backward-compatible wrapper for canonical lipid droplet type creation."""
    return _get_or_create_lipid_droplet_type()


def polygon_rings(polygon: Polygon) -> list[list[tuple[float, float]]]:
    """Return a shapely Polygon's rings as ``[exterior, *holes]`` coordinate lists.

    ``GEOSGeometry.coords`` on a Polygon returned exactly this; shapely splits it
    into ``.exterior`` and ``.interiors``.
    """
    rings = [[(float(x), float(y)) for x, y, *_ in polygon.exterior.coords]]
    for interior in polygon.interiors:
        rings.append([(float(x), float(y)) for x, y, *_ in interior.coords])
    return rings


def polygon_to_mask(polygon: Polygon, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Rasterize a polygon to a boolean mask matching image_shape.

    Args:
        polygon: shapely Polygon geometry (image pixel coordinates)
        image_shape: Tuple of (height, width) for the target mask

    Returns:
        2D boolean numpy array of shape image_shape
    """
    height, width = image_shape

    # Extract polygon coordinates
    # Handle both 2D (x, y) and 3D (x, y, z) coordinates by taking only first two
    try:
        coords = polygon.exterior.coords
    except (AttributeError, IndexError, TypeError):
        # If coordinate access fails, return empty mask
        logger.warning("Failed to extract coordinates from polygon, returning empty mask")
        return np.zeros((height, width), dtype=bool)

    # Convert to matplotlib Path format (list of (x, y) tuples)
    # Handle both 2D (x, y) and 3D (x, y, z) coordinates by taking only first two
    try:
        path_coords = [(float(coord[0]), float(coord[1])) for coord in coords]
    except (IndexError, TypeError) as e:
        # If coordinate unpacking fails, log and return empty mask
        logger.warning(
            f"Failed to convert coordinates to path format: {str(e)}, returning empty mask"
        )
        return np.zeros((height, width), dtype=bool)
    return _rasterize_polygon(path_coords, height, width)


def polygon_to_mask_in_roi(
    polygon: Polygon,
    roi_x: int,
    roi_y: int,
    roi_shape: tuple[int, int],
) -> np.ndarray:
    """
    Rasterize a polygon into a mask within ROI coordinates.

    Args:
        polygon: shapely Polygon geometry (full image coordinates)
        roi_x: ROI left offset in full image coordinates
        roi_y: ROI top offset in full image coordinates
        roi_shape: Tuple of (height, width) for the ROI mask

    Returns:
        2D boolean numpy array of shape roi_shape
    """
    height, width = roi_shape
    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=bool)

    try:
        coords = polygon.exterior.coords
    except (AttributeError, IndexError, TypeError):
        return np.zeros((height, width), dtype=bool)

    try:
        path_coords = [(float(coord[0] - roi_x), float(coord[1] - roi_y)) for coord in coords]
    except (IndexError, TypeError):
        return np.zeros((height, width), dtype=bool)

    return _rasterize_polygon(path_coords, height, width)


def tile_mask_to_polygon(
    tile_mask: np.ndarray,
    x_offset: int,
    y_offset: int,
    full_height: int,
    full_width: int,
    downsample_factor: float = 1.0,
) -> tuple[Polygon, tuple[float, float], Polygon]:
    """
    Convert a tile mask directly to a polygon in full-resolution coordinates.

    This avoids creating a full-resolution mask array, which is memory-efficient
    for large images. The mask is converted to a polygon in tile coordinates,
    then the polygon coordinates are translated to full-resolution coordinates.

    Args:
        tile_mask: 2D binary mask array in tile coordinates
        x_offset: X offset of the tile in working-image coordinates
        y_offset: Y offset of the tile in working-image coordinates
        full_height: Full-resolution image height
        full_width: Full-resolution image width
        downsample_factor: Factor used to downsample the image before tiling

    Returns:
        Tuple of (polygon, centroid_point, bbox_polygon) in full-resolution coordinates
    """
    # Calculate scale factor for coordinate translation
    scale_factor = 1.0 / downsample_factor if downsample_factor != 1.0 else 1.0

    # Calculate actual tile region (without padding)
    tile_height, tile_width = tile_mask.shape
    actual_tile_height = min(tile_height, int((full_height * downsample_factor) - y_offset))
    actual_tile_width = min(tile_width, int((full_width * downsample_factor) - x_offset))

    # Extract actual tile region
    actual_tile_mask = tile_mask[:actual_tile_height, :actual_tile_width]

    # Scale mask if downsampling was used
    if downsample_factor != 1.0 and actual_tile_height > 0 and actual_tile_width > 0:
        scaled_height = int(actual_tile_height * scale_factor)
        scaled_width = int(actual_tile_width * scale_factor)
        if scaled_height > 0 and scaled_width > 0:
            actual_tile_mask = (
                transform.resize(
                    actual_tile_mask.astype(np.float32) / 255.0,
                    (scaled_height, scaled_width),
                    order=0,
                    anti_aliasing=False,
                    preserve_range=True,
                ).astype(np.uint8)
                * 255
            )

    # Convert tile mask to polygon (in tile coordinates)
    polygon_tile, centroid_tile, bbox_tile = mask_to_polygon(actual_tile_mask)

    # Translate coordinates to full-resolution
    # Scale offsets
    if downsample_factor != 1.0:
        x_offset_full = int(x_offset * scale_factor)
        y_offset_full = int(y_offset * scale_factor)
    else:
        x_offset_full = x_offset
        y_offset_full = y_offset

    max_x_full = float(full_width)
    max_y_full = float(full_height)

    def _translate_to_image_bounds(ring):
        translated_ring = []
        for coord in ring:
            translated_x = float(coord[0] + x_offset_full)
            translated_y = float(coord[1] + y_offset_full)
            translated_ring.append(
                (
                    min(max(translated_x, 0.0), max_x_full),
                    min(max(translated_y, 0.0), max_y_full),
                )
            )
        return translated_ring

    # Translate polygon coordinates
    translated_rings = []
    for ring in polygon_rings(polygon_tile):
        # Contours from padded edge masks can land at -0.5 on the top/left border.
        translated_rings.append(_translate_to_image_bounds(ring))

    polygon_full = Polygon(translated_rings[0], translated_rings[1:])

    # Translate centroid
    centroid_x_full = centroid_tile[0] + x_offset_full
    centroid_y_full = centroid_tile[1] + y_offset_full
    centroid_full = (centroid_x_full, centroid_y_full)

    # Translate bbox
    bbox_rings = []
    for ring in polygon_rings(bbox_tile):
        # Handle both 2D (x, y) and 3D (x, y, z) coordinates by taking only first two
        translated_ring = [
            (float(coord[0] + x_offset_full), float(coord[1] + y_offset_full)) for coord in ring
        ]
        bbox_rings.append(translated_ring)
    bbox_full = Polygon(bbox_rings[0], bbox_rings[1:])

    return polygon_full, centroid_full, bbox_full


def mask_to_polygon(mask: np.ndarray) -> tuple[Polygon, tuple[float, float], Polygon]:
    """
    Convert a binary mask to a shapely Polygon geometry.

    Uses the largest contour as the exterior polygon. Also computes centroid and bbox.

    Args:
        mask: 2D binary mask array (uint8, 0/255 or boolean)

    Returns:
        Tuple of (polygon, centroid_point, bbox_polygon):
        - polygon: shapely Polygon in image pixel coordinates
        - centroid_point: (x, y) tuple for centroid
        - bbox_polygon: shapely Polygon representing bounding box rectangle
    """
    # Ensure mask is boolean
    binary_mask = (mask > 127).astype(bool) if mask.dtype != bool else mask

    # Pad with a false border so components that touch the crop edge still
    # produce a closed exterior contour instead of an open edge fragment.
    padded_mask = np.pad(binary_mask, 1, mode="constant", constant_values=False)

    # Find contours
    contours = measure.find_contours(padded_mask, level=0.5)

    if len(contours) == 0:
        # Empty mask - return a tiny polygon at origin
        coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        polygon = Polygon(coords)
        centroid = (0.5, 0.5)
        bbox = Polygon(coords)
        return polygon, centroid, bbox

    # Use the largest contour (by area)
    largest_contour = max(contours, key=lambda c: polygon_area(c))

    # Convert contour to polygon coordinates
    # Contours are in (row, col) format, convert to (x, y) = (col, row)
    coords = [(float(col - 1.0), float(row - 1.0)) for row, col in largest_contour]
    # Close the polygon if not already closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    polygon = Polygon(coords)

    # Compute centroid from mask
    y_coords, x_coords = np.where(binary_mask)
    if len(y_coords) > 0:
        centroid_x = float(np.mean(x_coords))
        centroid_y = float(np.mean(y_coords))
    else:
        # Fallback to polygon centroid
        centroid_x = float(np.mean([c[0] for c in coords[:-1]]))
        centroid_y = float(np.mean([c[1] for c in coords[:-1]]))

    centroid = (centroid_x, centroid_y)

    # Compute bounding box
    if len(y_coords) > 0:
        min_x, max_x = float(np.min(x_coords)), float(np.max(x_coords))
        min_y, max_y = float(np.min(y_coords)), float(np.max(y_coords))
    else:
        # Fallback to polygon bounds
        xs = [c[0] for c in coords[:-1]]
        ys = [c[1] for c in coords[:-1]]
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))

    bbox_coords = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y),
    ]
    bbox = Polygon(bbox_coords)

    return polygon, centroid, bbox


def compute_regionprops_features(mask: np.ndarray) -> dict:
    """
    Compute shape features using scikit-image regionprops.

    Args:
        mask: 2D binary mask array

    Returns:
        Dictionary of feature names to values
    """
    # Ensure mask is boolean
    binary_mask = (mask > 127).astype(bool) if mask.dtype != bool else mask

    # Label connected components
    labeled_mask, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        # Empty mask - return default values
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "eccentricity": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "major_axis_length": 0.0,
            "minor_axis_length": 0.0,
            "orientation": 0.0,
        }

    # Get regionprops for the largest component
    regions = measure.regionprops(labeled_mask)
    if len(regions) == 0:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "eccentricity": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "major_axis_length": 0.0,
            "minor_axis_length": 0.0,
            "orientation": 0.0,
        }

    props = max(regions, key=lambda r: r.area)

    # Compute perimeter from contour
    contours = measure.find_contours(binary_mask, level=0.5)
    if len(contours) > 0:
        largest_contour = max(contours, key=lambda c: polygon_area(c))
        # Approximate perimeter as sum of edge lengths
        perimeter = float(
            np.sum(
                np.sqrt(np.diff(largest_contour[:, 0]) ** 2 + np.diff(largest_contour[:, 1]) ** 2)
            )
        )
    else:
        perimeter = 0.0

    return {
        "area": float(props.area),
        "perimeter": perimeter,
        "eccentricity": (float(props.eccentricity) if hasattr(props, "eccentricity") else 0.0),
        "solidity": float(props.solidity) if hasattr(props, "solidity") else 0.0,
        "extent": float(props.extent) if hasattr(props, "extent") else 0.0,
        "major_axis_length": (
            float(props.axis_major_length)
            if hasattr(props, "axis_major_length")
            else (float(props.major_axis_length) if hasattr(props, "major_axis_length") else 0.0)
        ),
        "minor_axis_length": (
            float(props.axis_minor_length)
            if hasattr(props, "axis_minor_length")
            else (float(props.minor_axis_length) if hasattr(props, "minor_axis_length") else 0.0)
        ),
        "orientation": (float(props.orientation) if hasattr(props, "orientation") else 0.0),
    }


def compute_intensity_features(
    image: np.ndarray, mask: np.ndarray, outside_ring_pixels: int = 10
) -> dict:
    """
    Compute intensity-based features inside and outside the mask.

    Args:
        image: 2D grayscale image array
        mask: 2D binary mask array
        outside_ring_pixels: Number of pixels to use for outside ring (default 10)

    Returns:
        Dictionary of feature names to values (percentiles, mean, etc.)
    """
    # Ensure mask is boolean
    binary_mask = (mask > 127).astype(bool) if mask.dtype != bool else mask

    features = {}

    # Inside mask features
    inside_pixels = image[binary_mask]
    if len(inside_pixels) > 0:
        features["intensity_inside_mean"] = float(np.mean(inside_pixels))
        features["intensity_inside_p10"] = float(np.percentile(inside_pixels, 10))
        features["intensity_inside_p50"] = float(np.percentile(inside_pixels, 50))
        features["intensity_inside_p90"] = float(np.percentile(inside_pixels, 90))
        features["intensity_inside_std"] = float(np.std(inside_pixels))
    else:
        features["intensity_inside_mean"] = 0.0
        features["intensity_inside_p10"] = 0.0
        features["intensity_inside_p50"] = 0.0
        features["intensity_inside_p90"] = 0.0
        features["intensity_inside_std"] = 0.0

    # Outside ring features
    # Create a dilated mask to get the ring region
    dilated_mask = ndimage.binary_dilation(binary_mask, iterations=outside_ring_pixels)
    ring_mask = dilated_mask & ~binary_mask

    outside_pixels = image[ring_mask]
    if len(outside_pixels) > 0:
        features["intensity_outside_mean"] = float(np.mean(outside_pixels))
        features["intensity_outside_p10"] = float(np.percentile(outside_pixels, 10))
        features["intensity_outside_p50"] = float(np.percentile(outside_pixels, 50))
        features["intensity_outside_p90"] = float(np.percentile(outside_pixels, 90))
        features["intensity_outside_std"] = float(np.std(outside_pixels))
    else:
        # If no ring pixels, use all non-mask pixels
        outside_mask = ~binary_mask
        outside_pixels = image[outside_mask]
        if len(outside_pixels) > 0:
            features["intensity_outside_mean"] = float(np.mean(outside_pixels))
            features["intensity_outside_p10"] = float(np.percentile(outside_pixels, 10))
            features["intensity_outside_p50"] = float(np.percentile(outside_pixels, 50))
            features["intensity_outside_p90"] = float(np.percentile(outside_pixels, 90))
            features["intensity_outside_std"] = float(np.std(outside_pixels))
        else:
            features["intensity_outside_mean"] = 0.0
            features["intensity_outside_p10"] = 0.0
            features["intensity_outside_p50"] = 0.0
            features["intensity_outside_p90"] = 0.0
            features["intensity_outside_std"] = 0.0

    # Contrast features
    if len(inside_pixels) > 0 and len(outside_pixels) > 0:
        features["intensity_contrast"] = float(
            features["intensity_inside_mean"] - features["intensity_outside_mean"]
        )
    else:
        features["intensity_contrast"] = 0.0

    return features


def convert_probability_map_to_uint8_png(
    input_path: Path, output_path: Path, channel_index: int = 0
) -> None:
    """
    Convert a probability map file (PNG or TIFF, any dtype) to compressed uint8 PNG.

    This function:
    - Accepts PNG or TIFF files
    - Handles any dtype (float32, uint16, etc.)
    - Normalizes to 0-255 uint8 range (preserving the probability distribution)
    - Saves as compressed PNG

    Args:
        input_path: Path to the input file (PNG or TIFF)
        output_path: Path where the uint8 PNG will be saved (should have .png extension)
        channel_index: Channel index to extract if input is multi-channel (default 0)

    Raises:
        ValueError: If input file format is not supported or cannot be read
        IOError: If file cannot be written
    """
    # Load image based on file extension
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix.lower() in [".tif", ".tiff"]:
        # Load TIFF using tifffile
        data = tifffile.imread(str(input_path))
    else:
        # Load PNG or other format using PIL
        pil_image = Image.open(input_path)
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")
        data = np.array(pil_image)

    # Handle multi-channel images
    if len(data.shape) == 3:
        # Select the specified channel
        if data.shape[0] < data.shape[2]:
            # Channels first: (C, H, W)
            data = data[channel_index, :, :]
        else:
            # Channels last: (H, W, C)
            data = data[:, :, channel_index]

    # Ensure 2D
    if data.ndim != 2:
        raise ValueError(f"Expected 2D probability map, got shape {data.shape}")

    # Convert to float32 for normalization
    data_float = data.astype(np.float32)

    # Normalize to 0-255 uint8 range
    # Handle different input ranges:
    # - If already in 0-1 range, scale to 0-255
    # - If in 0-255 range (uint8), keep as is
    # - If in 0-65535 range (uint16), scale to 0-255
    # - Otherwise, use min-max normalization
    data_min = np.min(data_float)
    data_max = np.max(data_float)

    if data_max > data_min:
        # Normalize to 0-1 first, then scale to 0-255
        if data_max <= 1.0:
            # Already in 0-1 range
            data_normalized = data_float
        elif data_max <= 255.0:
            # Likely in 0-255 range, normalize to 0-1
            data_normalized = data_float / 255.0
        elif data_max <= 65535.0:
            # Likely in 0-65535 range (uint16), normalize to 0-1
            data_normalized = data_float / 65535.0
        else:
            # Use min-max normalization
            data_normalized = (data_float - data_min) / (data_max - data_min)

        # Scale to 0-255 and convert to uint8
        data_uint8 = (data_normalized * 255.0).clip(0, 255).astype(np.uint8)
    else:
        # Constant image (all same value), set to 0
        data_uint8 = np.zeros_like(data_float, dtype=np.uint8)

    # Save as compressed PNG
    pil_output = Image.fromarray(data_uint8, mode="L")
    pil_output.save(output_path, format="PNG", compress_level=9, optimize=True)
