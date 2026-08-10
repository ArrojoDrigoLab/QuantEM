"""Overlay bundle paths and filesystem state helpers."""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from quantem.core.config import SEGMENTATION_OVERLAYS_TMP_DIR
from quantem.segmentation.models import ImageSegmentation, SegmentationOverlayState

from .constants import (
    OVERLAY_STAGING_DIRNAME,
    OVERLAY_STORE_DIRNAME,
    OVERLAY_VERSIONED_DIRNAME,
)


class OverlayStoreError(RuntimeError):
    """Raised when the overlay bundle is missing or malformed."""


def normalize_overlay_source_model(source_model: str | None) -> str:
    return (source_model or "").strip().lower()


def _overlay_source_slug(source_model: str | None) -> str:
    normalized = normalize_overlay_source_model(source_model)
    if not normalized:
        return ""
    slug = re.sub(r"[^a-z0-9_.-]+", "_", normalized)
    return slug.strip("._-") or "source"


def get_overlay_root(segmentation_id: str, source_model: str | None = None) -> Path:
    base = SEGMENTATION_OVERLAYS_TMP_DIR / str(segmentation_id)
    slug = _overlay_source_slug(source_model)
    if not slug:
        return base
    return base / "sources" / slug


def get_overlay_debug_manifest_path(
    segmentation_id: str,
    source_model: str | None = None,
) -> Path:
    return get_overlay_root(segmentation_id, source_model) / "manifest.json"


def get_overlay_active_bundle_path(state: SegmentationOverlayState) -> Path:
    root = (
        get_overlay_root(str(state.segmentation_id))
        if not state.candidate_source_model
        else get_overlay_root(str(state.segmentation_id), state.candidate_source_model)
    )
    return (
        root
        / OVERLAY_VERSIONED_DIRNAME
        / str(state.bundle_version)
        / OVERLAY_STORE_DIRNAME
    )


def get_overlay_stage_bundle_path(
    segmentation_id: str,
    bundle_version: int,
    source_model: str | None = None,
) -> Path:
    return (
        get_overlay_root(segmentation_id, source_model)
        / OVERLAY_STAGING_DIRNAME
        / str(bundle_version)
        / OVERLAY_STORE_DIRNAME
    )


def get_overlay_version_dir(
    segmentation_id: str,
    bundle_version: int,
    source_model: str | None = None,
) -> Path:
    return (
        get_overlay_root(segmentation_id, source_model)
        / OVERLAY_VERSIONED_DIRNAME
        / str(bundle_version)
    )


def get_or_create_overlay_state(
    segmentation: ImageSegmentation,
    source_model: str | None = None,
) -> SegmentationOverlayState:
    normalized_source_model = normalize_overlay_source_model(source_model)
    state, _ = SegmentationOverlayState.objects.get_or_create(
        segmentation=segmentation,
        candidate_source_model=normalized_source_model,
    )
    return state


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for attempt in range(3):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if not path.exists():
                return
            time.sleep(0.05 * (attempt + 1))
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        raise OverlayStoreError(f"Failed to remove overlay path: {path}")


def _close_overlay_arrays(arrays) -> None:
    if isinstance(arrays, dict):
        flat: list = []
        for value in arrays.values():
            flat.extend(value)
    else:
        flat = list(arrays)
    seen_store_ids: set[int] = set()
    for array in flat:
        store = getattr(array, "store", None)
        if store is None:
            continue
        store_id = id(store)
        if store_id in seen_store_ids:
            continue
        seen_store_ids.add(store_id)
        close = getattr(store, "close", None)
        if callable(close):
            close()
