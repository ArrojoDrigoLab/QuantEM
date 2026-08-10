"""ID-map segmentation overlay: build, serve, and render-time LUT.

The overlay is an integer label raster (``labels``) plus a baked border mask
(``border``); object colour/state lives in a render-time LUT, so state changes
never touch the raster. See :mod:`.constants` for the format overview.
"""

from __future__ import annotations

from . import labels_lut, render
from .dirty import (
    DirtyBBox,
    dirty_bbox_to_chunk_coords,
    full_image_dirty_bbox,
    merge_dirty_bboxes,
)
from .labels_lut import build_label_lut_binary, build_label_lut_json
from .manifest import build_overlay_manifest, ensure_overlay_manifest
from .mutations import (
    apply_partial_overlay_update,
    build_overlay_mutation_response,
    overlay_rebuild_policy,
    queue_full_overlay_rebuild,
    queue_overlay_rebuild,
    rebuild_overlay_full,
    register_overlay_mutation,
    register_overlay_mutation_all_bundles,
    register_state_mutation,
    run_overlay_rebuild_job,
)
from .paths import (
    OverlayStoreError,
    get_or_create_overlay_state,
    get_overlay_active_bundle_path,
    get_overlay_debug_manifest_path,
    get_overlay_root,
    get_overlay_stage_bundle_path,
    get_overlay_version_dir,
    normalize_overlay_source_model,
)
from .store import encode_zero_chunk, get_overlay_chunk_shape, parse_overlay_chunk_path

__all__ = [
    "DirtyBBox",
    "OverlayStoreError",
    "apply_partial_overlay_update",
    "build_label_lut_binary",
    "build_label_lut_json",
    "build_overlay_manifest",
    "build_overlay_mutation_response",
    "dirty_bbox_to_chunk_coords",
    "encode_zero_chunk",
    "ensure_overlay_manifest",
    "full_image_dirty_bbox",
    "get_or_create_overlay_state",
    "get_overlay_active_bundle_path",
    "get_overlay_chunk_shape",
    "get_overlay_debug_manifest_path",
    "get_overlay_root",
    "get_overlay_stage_bundle_path",
    "get_overlay_version_dir",
    "labels_lut",
    "merge_dirty_bboxes",
    "normalize_overlay_source_model",
    "overlay_rebuild_policy",
    "parse_overlay_chunk_path",
    "queue_full_overlay_rebuild",
    "queue_overlay_rebuild",
    "rebuild_overlay_full",
    "register_overlay_mutation",
    "register_overlay_mutation_all_bundles",
    "register_state_mutation",
    "render",
    "run_overlay_rebuild_job",
]
