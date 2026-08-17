"""
Generic DB Layer for Organelle Segmentation
=============================================

Parameterized by a BaseSegmenter instance. Handles:
- Probability map persistence
- Inference orchestration with image loading
- Segment extraction and candidate replacement
"""

from .extraction import extract_and_save_segments
from .inference import run_inference_for_segmentation
from .prob_maps import (
    get_prob_map_file_path,
    load_prob_map_from_file,
    load_prob_map_from_path,
    load_prob_map_uint8_from_path,
    prob_map_file_exists,
    save_probability_map,
)

__all__ = [
    "run_inference_for_segmentation",
    "extract_and_save_segments",
    "get_prob_map_file_path",
    "prob_map_file_exists",
    "save_probability_map",
    "load_prob_map_from_path",
    "load_prob_map_uint8_from_path",
    "load_prob_map_from_file",
]
