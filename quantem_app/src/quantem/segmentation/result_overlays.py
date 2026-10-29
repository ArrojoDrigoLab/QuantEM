"""Full-resolution saved-result comparisons, without changing accepted data."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from django.db.models import Count, Max
from shapely.geometry.base import BaseGeometry

from quantem.segmentation.models import (
    CompletedROI,
    GlobalMask,
    RoiSegmentationStatus,
    SavedSegmentationResult,
    SegmentationOverlayState,
    SegmentObject,
)
from quantem.segmentation.overlay_ngff.constants import (
    MACRO_TILE_SIZE,
    OVERLAY_BORDER_WIDTH,
)
from quantem.segmentation.overlay_ngff.dimensions import segmentation_dimensions
from quantem.segmentation.overlay_ngff.paths import _close_overlay_arrays, _remove_tree
from quantem.segmentation.overlay_ngff.store import is_valid_label_store_cached
from quantem.segmentation.prob_maps.io import resolve_probability_map_path
from quantem.segmentation.prob_maps.preview import result_preview_cache_path

COMPARISON_FORMAT = 1


def comparison_identity(result: SavedSegmentationResult) -> str:
    """Cheap identity for the prediction and current annotation protection."""
    from quantem import __version__

    source = resolve_probability_map_path(result.probability_map)
    stat = source.stat()
    segmentation = result.segmentation
    identity = [
        COMPARISON_FORMAT,
        __version__,
        str(result.pk),
        result.threshold,
        result.is_canonical,
        result.updated_at,
        result.source_model,
        result.probability_map.updated_at,
        stat.st_mtime_ns,
        stat.st_size,
        segmentation_dimensions(segmentation),
        segmentation.manual_full_image,
    ]
    # Bulk label/geometry edits may intentionally omit per-object timestamps.
    # Their registered display revisions still invalidate the scientific view.
    identity.append(
        list(
            SegmentationOverlayState.objects.filter(
                segmentation=segmentation, candidate_source_model=""
            ).values_list("desired_revision", "lut_revision")
        )
    )
    for model in (SegmentObject, CompletedROI, RoiSegmentationStatus, GlobalMask):
        rows = model.objects.filter(segmentation=segmentation)
        if model is SegmentObject:
            rows = rows.filter(superseded_at__isnull=True)
        identity.append(rows.aggregate(count=Count("pk"), changed=Max("updated_at")))
    # ROI rectangles live on ImageROI rather than the status row.
    identity.append(
        list(
            RoiSegmentationStatus.objects.filter(segmentation=segmentation, is_complete=True)
            .order_by("pk")
            .values_list("image_roi__x", "image_roi__y", "image_roi__width", "image_roi__height")
        )
    )
    return hashlib.sha256(json.dumps(identity, default=str, sort_keys=True).encode()).hexdigest()[
        :32
    ]


def comparison_directory(result: SavedSegmentationResult, key: str) -> Path:
    if len(key) != 32 or any(char not in "0123456789abcdef" for char in key):
        raise ValueError("Invalid comparison revision")
    source = resolve_probability_map_path(result.probability_map)
    return result_preview_cache_path(source) / str(result.id) / key


def comparison_is_ready(directory: Path, key: str) -> bool:
    """A published marker alone cannot certify an intact tiled comparison."""
    try:
        metadata = json.loads((directory / "ready.json").read_text(encoding="utf-8"))
        return metadata["key"] == key and is_valid_label_store_cached(
            directory / "labels.zarr", width=metadata["width"], height=metadata["height"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def comparison_geometries(
    result: SavedSegmentationResult,
    *,
    cancel_check: Callable[[], None],
    progress: Callable[[float, str], None],
) -> tuple[list[BaseGeometry], np.ndarray | None]:
    """Use the confirmation extractor, polygon repair, and rejection rules."""
    from quantem.seg_core.db.candidate_protection import ProtectionIndex
    from quantem.seg_core.db.extraction import resolve_min_area
    from quantem.seg_core.db.inference import replay_stored_probability_map
    from quantem.seg_core.db.segment_writer import build_row, prepare_row, to_valid_polygon
    from quantem.seg_core.registry import get_segmenter_or_none
    from quantem.segmentation.coverage import completed_geometry
    from quantem.segmentation.geometry import extract_polygons
    from quantem.segmentation.global_masks import (
        load_global_mask,
        postprocess_global_prediction,
        preserve_completed_global_mask,
    )
    from quantem.segmentation.source_models import (
        SOURCE_MODEL_MANUAL,
        resolve_segmenter_internal_name,
    )

    segmentation = result.segmentation
    global_mode = segmentation.segmentation_type.measurement_mode == "global"
    live = SegmentObject.objects.filter(segmentation=segmentation, superseded_at=None)
    if result.is_canonical:
        return (
            ([], load_global_mask(segmentation))
            if global_mode
            else (
                [row.geometry for row in live.filter(label_state="CONFIRMED").only("geometry_wkb")],
                None,
            )
        )
    segmenter = get_segmenter_or_none(
        resolve_segmenter_internal_name(
            segmentation_type_internal_name=segmentation.segmentation_type.internal_name,
            source_model=result.source_model,
        ),
        source_model=result.source_model,
    )
    if segmenter is None:
        raise ValueError(
            "The model's postprocessing settings are unavailable. Install its model pack."
        )
    segmentation._probability_map_override = result.probability_map
    replay, _ = replay_stored_probability_map(
        segmenter, segmentation, threshold=result.threshold, geometry_only=True
    )
    cancel_check()
    if global_mode:
        mask, _ = postprocess_global_prediction(segmenter, replay.prob, result.threshold)
        cancel_check()
        return [], preserve_completed_global_mask(segmentation, mask)
    protected = completed_geometry(segmentation)

    def extraction_progress(fraction):
        cancel_check()
        progress(10 + 35 * fraction, "Finding postprocessed objects")

    extracted = segmenter.extract_instances_geometry(
        replay.prob, min_area=resolve_min_area(segmenter, None), on_progress=extraction_progress
    )
    preserved, rejected = [], []
    if not protected.is_empty:
        for row in live.only("geometry_wkb", "label_state", "source_model"):
            geometry = row.geometry
            kept = geometry.intersection(protected)
            if kept.is_empty:
                continue
            if row.label_state == "EXCLUDED" and row.source_model in {
                result.source_model,
                SOURCE_MODEL_MANUAL,
            }:
                rejected.extend(extract_polygons(kept))
            elif row.label_state == "CONFIRMED" or (
                row.label_state in {"CANDIDATE", "INFERRED"} and kept.equals(geometry)
            ):
                preserved.extend(extract_polygons(kept))
    protection = ProtectionIndex([], rejected)
    pairs = [(segment, to_valid_polygon(segment.polygon_coords)) for segment in extracted]
    pairs = [(segment, geometry) for segment, geometry in pairs if geometry is not None]
    suppressed = protection.suppressed_mask([geometry for _, geometry in pairs])
    for (segment, geometry), dropped in zip(pairs, suppressed, strict=True):
        cancel_check()
        if dropped:
            continue
        row = prepare_row(
            build_row(
                segmentation, segment, geometry, run_identity=None, source_model=result.source_model
            )
        )
        preserved.extend(extract_polygons(row.geometry.difference(protected)))
    return preserved, None


def build_comparison(
    result_id: str,
    key: str,
    *,
    cancel_check: Callable[[], None],
    progress: Callable[[float, str], None],
) -> None:
    from quantem.segmentation.overlay_ngff.mutations import bundle_write_lock

    segmentation_id = SavedSegmentationResult.objects.values_list("segmentation_id", flat=True).get(
        pk=result_id
    )
    with bundle_write_lock(str(segmentation_id), f"comparison:{result_id}"):
        return _build_comparison(result_id, key, cancel_check=cancel_check, progress=progress)


def _build_comparison(result_id, key, *, cancel_check, progress):
    from types import SimpleNamespace

    from quantem.segmentation.overlay_ngff import render
    from quantem.segmentation.overlay_ngff.mutations import (
        _build_draw_ops,
        _build_pyramid,
        _macro_tile_payloads,
        _rasterize_level0,
        _write_tile_result,
    )
    from quantem.segmentation.overlay_ngff.store import _create_empty_label_store

    cancel_check()
    result = SavedSegmentationResult.objects.select_related(
        "probability_map", "segmentation__asset", "segmentation__segmentation_type"
    ).get(pk=result_id, deleted_at=None)
    if comparison_identity(result) != key:
        raise ValueError("Annotations changed. Open the comparison again to refresh it.")
    directory = comparison_directory(result, key)
    if comparison_is_ready(directory, key):
        return
    if directory.exists():
        # An incomplete artifact is disposable; otherwise a retry would report
        # success forever while the serving endpoint correctly rejects it.
        _remove_tree(directory)
    # The per-result writer lock also makes abandoned-stage reclamation safe
    # after a killed worker. Never leave an unbounded collection of UUID stages.
    directory.parent.mkdir(parents=True, exist_ok=True)
    for abandoned in directory.parent.glob(".*.pending"):
        if abandoned.is_dir():
            _remove_tree(abandoned)
    stage = directory.with_name(f".{key}.pending")
    root = stage / "labels.zarr"
    stage.mkdir(parents=True)
    width, height = segmentation_dimensions(result.segmentation)
    try:
        progress(5, "Reading the saved result")
        geometries, mask = comparison_geometries(
            result, cancel_check=cancel_check, progress=progress
        )
        arrays = _create_empty_label_store(result.segmentation, root)
        bounds = []
        try:
            if mask is None:
                objects = [
                    SimpleNamespace(
                        id=i, geometry=geometry, label_state="CONFIRMED", refined=False, status=1
                    )
                    for i, geometry in enumerate(geometries)
                ]
                ops = _build_draw_ops(objects, label_map={obj.id: 1 for obj in objects})
                bounds = [op["bbox"] for op in ops]
                _rasterize_level0(
                    arrays,
                    _macro_tile_payloads(ops, width=width, height=height),
                    use_pool=False,
                    cancel_check=cancel_check,
                )
            else:
                halo = OVERLAY_BORDER_WIDTH
                for y in range(0, height, MACRO_TILE_SIZE):
                    for x in range(0, width, MACRO_TILE_SIZE):
                        cancel_check()
                        x1, y1 = min(width, x + MACRO_TILE_SIZE), min(height, y + MACRO_TILE_SIZE)
                        rx, ry = max(0, x - halo), max(0, y - halo)
                        block = mask[
                            ry : min(height, y1 + halo), rx : min(width, x1 + halo)
                        ].astype(np.uint32)
                        if not block.any():
                            continue
                        borders = render._compute_border(block, width=halo)
                        crop = (slice(y - ry, y1 - ry), slice(x - rx, x1 - rx))
                        _write_tile_result(arrays, (x, y, block[crop], borders[crop]))
                        bounds.append((x, y, x1, y1))
        finally:
            _close_overlay_arrays(arrays)
        progress(65, "Building comparison zoom levels")
        _build_pyramid(
            root, width=width, height=height, content_bboxes=bounds, cancel_check=cancel_check
        )
        cancel_check()
        result.refresh_from_db()
        # Never label a mixture of annotation revisions as a ready comparison.
        if result.deleted_at is not None or comparison_identity(result) != key:
            raise ValueError("Annotations changed while the comparison was building. Refresh it.")
        (stage / "ready.json").write_text(
            json.dumps({"key": key, "width": width, "height": height}), encoding="utf-8"
        )
        if not directory.exists():
            stage.rename(directory)
        progress(100, "Detailed comparison ready")
        try:
            prune_comparisons(directory.parent, keep=directory)
        except OSError:
            logging.getLogger(__name__).warning(
                "Could not prune older comparison generations", exc_info=True
            )
    finally:
        if stage.exists():
            _remove_tree(stage)


def prune_comparisons(parent: Path, *, keep: Path) -> None:
    """Keep three recent generations and one hour for open readers."""
    directories = sorted(
        (path for path in parent.iterdir() if (path / "ready.json").is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for directory in directories[3:]:
        if directory != keep and directory.stat().st_mtime < time.time() - 3600:
            _remove_tree(directory)
