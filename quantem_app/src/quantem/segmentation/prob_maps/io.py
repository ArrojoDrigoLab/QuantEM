"""
Helpers for saving and validating probability map uploads.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import tifffile
from django.core.files.uploadedfile import UploadedFile
from PIL import Image

from quantem.core.config import PROB_MAPS_DIR
from quantem.core.local_storage import (
    ensure_cached_storage_path,
    storage_path,
    storage_relpath_for_path,
)
from quantem.segmentation.models import ImageSegmentation, ProbabilityMap
from quantem.segmentation.utils import convert_probability_map_to_uint8_png


def _load_prob_map_data(file_path: Path, channel_index: int) -> np.ndarray:
    if file_path.suffix.lower() in [".tif", ".tiff"]:
        data = tifffile.imread(str(file_path))
    else:
        pil_image = Image.open(file_path)
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")
        data = np.array(pil_image)

    if data.ndim == 3:
        if data.shape[0] < data.shape[2]:
            data = data[channel_index, :, :]
        else:
            data = data[:, :, channel_index]

    return data


def save_probability_map_upload(
    segmentation: ImageSegmentation,
    uploaded_file: UploadedFile,
    *,
    name: str,
    expected_height: int,
    expected_width: int,
    channel_index: int = 0,
    is_roi: bool = False,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> ProbabilityMap:
    prob_maps_dir = PROB_MAPS_DIR / str(segmentation.id)
    prob_maps_dir.mkdir(parents=True, exist_ok=True)

    temp_file_path = (
        prob_maps_dir / f"temp_{uuid.uuid4()}{Path(uploaded_file.name).suffix}"
    )
    with open(temp_file_path, "wb") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    try:
        data = _load_prob_map_data(temp_file_path, channel_index)
        if data.ndim != 2:
            raise ValueError(
                f"Probability map must be 2D or 3D (multi-channel). Got shape: {data.shape}"
            )

        prob_height, prob_width = data.shape
        if prob_height != expected_height or prob_width != expected_width:
            raise ValueError(
                "Probability map dimensions "
                f"({prob_height}x{prob_width}) do not match expected "
                f"({expected_height}x{expected_width})"
            )

        unique_filename = f"{uuid.uuid4()}.png"
        output_path = prob_maps_dir / unique_filename
        try:
            convert_probability_map_to_uint8_png(
                temp_file_path, output_path, channel_index=channel_index
            )
        except Exception:
            if output_path.exists():
                output_path.unlink()
            raise
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

    relative_path = storage_relpath_for_path(output_path)
    metadata: dict[str, object] = {}
    if is_roi and roi_bbox is not None:
        metadata["roi"] = {
            "x": int(roi_bbox[0]),
            "y": int(roi_bbox[1]),
            "width": int(roi_bbox[2]),
            "height": int(roi_bbox[3]),
        }

    prob_map = ProbabilityMap.objects.create(
        segmentation=segmentation,
        name=name,
        file_path=relative_path,
        channel_index=channel_index,
        metadata=metadata,
    )

    return prob_map


def clone_probability_map(
    prob_map: ProbabilityMap,
    segmentation: ImageSegmentation,
    *,
    name: str | None = None,
) -> ProbabilityMap:
    """
    Clone a probability map record to another segmentation using the same file path.
    """
    return ProbabilityMap.objects.create(
        segmentation=segmentation,
        name=name if name is not None else prob_map.name,
        file_path=prob_map.file_path,
        channel_index=prob_map.channel_index,
        metadata=prob_map.metadata,
    )


def resolve_probability_map_path(prob_map: ProbabilityMap) -> Path:
    """
    Resolve a ProbabilityMap.file_path to an absolute path.

    Uses STORAGE_DIR-relative paths (including tmp/ for ROI maps).
    """
    resolved = storage_path(prob_map.file_path)
    if not resolved.exists():
        ensure_cached_storage_path(prob_map.file_path, path_type="file")
    return resolved
