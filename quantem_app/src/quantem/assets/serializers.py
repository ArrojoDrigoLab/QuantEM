"""
Serialization for the local image library (assets, renditions, ROIs).

``ImageROISerializer`` is a plain DRF ``ModelSerializer``. Assets are serialized
by hand instead, because an asset payload is a join of the ``Asset`` row, its
``Rendition`` rows and the on-disk NGFF state - exactly the read model the
viewer needs to decide whether an image can be opened.
"""

from __future__ import annotations

from typing import Any

from rest_framework import serializers

from .asset_openable import asset_ngff_ready, get_asset_openable
from .models import ImageROI


def _positive_int(value) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _format_dimension_summary(
    width: int | None,
    height: int | None,
    depth: int | None = None,
) -> str:
    width_value = _positive_int(width)
    height_value = _positive_int(height)
    if width_value is None or height_value is None:
        return ""
    depth_value = _positive_int(depth)
    dimensions = [str(width_value), str(height_value)]
    if depth_value is not None:
        dimensions.append(str(depth_value))
    return "x".join(dimensions)


def _format_pixel_size(value: float | None) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        return ""
    text = f"{number:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_pixel_size_summary(
    pixel_size_nm: float | None,
    pixel_size_nm_z: float | None = None,
) -> str:
    """Render the numeric pixel size as ``"2x2 nm/pixel"`` / ``"2x2x50 nm/pixel"``.

    The components are real numbers, so the only formatting left is choosing how
    many decimals to show.
    """
    xy = _format_pixel_size(pixel_size_nm)
    if not xy:
        return ""
    components = [xy, xy]
    z = _format_pixel_size(pixel_size_nm_z)
    if z:
        components.append(z)
    return f"{'x'.join(components)} nm/pixel"


def format_image_metadata_summary(
    *,
    width: int | None,
    height: int | None,
    pixel_size_nm: float | None = None,
    depth: int | None = None,
    pixel_size_nm_z: float | None = None,
) -> str:
    dimension_summary = _format_dimension_summary(width, height, depth)
    pixel_size_summary = _format_pixel_size_summary(pixel_size_nm, pixel_size_nm_z)
    return ", ".join(part for part in [dimension_summary, pixel_size_summary] if part)


def _isoformat(value) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None


def _positive_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _renditions_full_first(asset) -> list:
    """The asset's renditions, FULL type first.

    FULL first because a volume import rewrites the same rendition row in place
    and only the FULL type is guaranteed to exist. Reads the prefetched
    renditions when the caller supplied them, so provenance costs no extra
    query per asset.
    """
    cache = getattr(asset, "_prefetched_objects_cache", {})
    prefetched = cache.get("renditions")
    renditions = list(prefetched) if prefetched is not None else list(asset.renditions.all())
    return [r for r in renditions if r.type == "FULL"] + [r for r in renditions if r.type != "FULL"]


def file_declared_pixel_size_nm(asset) -> float | None:
    """In-plane pixel size the *source file itself* declared, or None if silent.

    ``Asset.pixel_size_nm`` is the effective value and may have been typed by
    hand, so on its own it cannot say where the number came from. The file's own
    claim survives on the FULL rendition's metadata
    (``source_metadata.pixel_size_nm`` for a 2D import,
    ``volume_metadata.voxel_size_nm`` as (z, y, x) for a volume); comparing the
    two is how the UI separates "read from file" from "entered by hand".

    The detail payload could always do that because it embeds ``renditions``.
    The list payload cannot -- and without this field every calibrated image in
    the library resolved to "entered by hand", which is a provenance claim about
    a number that ends up in a figure caption. Hence one scalar here rather than
    the whole rendition list.

    Reads the prefetched renditions when the caller supplied them
    (``AssetListView`` does, via ``prefetch_related("renditions")``), so this
    costs no extra query per asset.
    """
    for rendition in _renditions_full_first(asset):
        metadata = rendition.metadata
        if not isinstance(metadata, dict):
            continue

        source_metadata = metadata.get("source_metadata")
        if isinstance(source_metadata, dict):
            declared = _positive_float(source_metadata.get("pixel_size_nm"))
            if declared is not None:
                return declared

        volume_metadata = metadata.get("volume_metadata")
        if isinstance(volume_metadata, dict):
            voxel = volume_metadata.get("voxel_size_nm")
            if isinstance(voxel, (list, tuple)) and len(voxel) >= 3:
                in_plane = _positive_float(voxel[2])
                if in_plane is not None:
                    return in_plane
    return None


def file_declared_pixel_size_source(asset) -> str | None:
    """Which tag or block in the file supplied the pixel size, or ``None``.

    Travels beside :func:`file_declared_pixel_size_nm`, which can only say
    *that* the file declared a scale. This says *who declared it*: the
    microscope (``"TIFF tag 51023 (FibicsXML)"``), Fiji (``"ImageJ
    ImageDescription unit (TIFF tag 270)"``), the baseline TIFF tags, or
    OME-XML. The vocabulary is
    ``volume_readers.PIXEL_SIZE_SOURCE_*``, and the
    reader records it at import (``source_metadata.pixel_size_source`` for a 2D
    import, ``volume_metadata.source.extra.calibration_source`` for a volume).

    Absent on assets imported before the vendor-tag reader landed, which is
    honest: nothing recorded where their number came from, so nothing here can
    invent it. ``None`` therefore means "not recorded", not "typed by hand" --
    that comparison is still :func:`file_declared_pixel_size_nm` against
    ``Asset.pixel_size_nm``.
    """
    for rendition in _renditions_full_first(asset):
        metadata = rendition.metadata
        if not isinstance(metadata, dict):
            continue

        source_metadata = metadata.get("source_metadata")
        if isinstance(source_metadata, dict):
            source = source_metadata.get("pixel_size_source")
            if isinstance(source, str) and source.strip():
                return source

        volume_metadata = metadata.get("volume_metadata")
        if isinstance(volume_metadata, dict):
            source_block = volume_metadata.get("source")
            extra = source_block.get("extra") if isinstance(source_block, dict) else None
            if isinstance(extra, dict):
                source = extra.get("calibration_source")
                if isinstance(source, str) and source.strip():
                    return source
    return None


def file_declared_pixel_size_caveat(asset) -> str | None:
    """Conflict note recorded when the file's pixel size was read, or ``None``.

    Travels beside :func:`file_declared_pixel_size_nm`: a file can declare its
    scale twice (an ImageJ/Fiji ``unit`` in the ImageDescription *and* a
    baseline TIFF ``ResolutionUnit`` tag) and the two can disagree. The reader
    picks one -- the rule lives on
    ``volume_readers._resolution_tag_nm_and_conflict`` -- but it never does so
    silently: the note it records in the rendition metadata
    (``source_metadata.pixel_size_caveat`` for a 2D import,
    ``volume_metadata.source.extra.calibration_conflict`` for a volume) is
    surfaced here so a number that ends up in a figure caption carries its
    doubt with it.
    """
    for rendition in _renditions_full_first(asset):
        metadata = rendition.metadata
        if not isinstance(metadata, dict):
            continue

        source_metadata = metadata.get("source_metadata")
        if isinstance(source_metadata, dict):
            caveat = source_metadata.get("pixel_size_caveat")
            if isinstance(caveat, str) and caveat.strip():
                return caveat

        volume_metadata = metadata.get("volume_metadata")
        if isinstance(volume_metadata, dict):
            source = volume_metadata.get("source")
            extra = source.get("extra") if isinstance(source, dict) else None
            if isinstance(extra, dict):
                caveat = extra.get("calibration_conflict")
                if isinstance(caveat, str) and caveat.strip():
                    return caveat
    return None


def _asset_datasets(asset) -> list:
    """The asset's datasets, from the prefetch when the caller supplied one.

    ``AssetListView`` prefetches them, so a library page costs one extra query
    for sixty cards rather than sixty.
    """
    cache = getattr(asset, "_prefetched_objects_cache", {})
    prefetched = cache.get("datasets")
    if prefetched is not None:
        return list(prefetched)
    return list(asset.datasets.all())


def serialize_asset_grouping(asset) -> dict[str, Any]:
    """Where this image sits in the library, if anywhere.

    Emitted on the **list** entry and not only on the detail payload, because
    the library groups and filters on it and a per-card round trip for sixty
    cards is not a grouping, it is a stampede.

    Every field is nullable or empty and that is the ordinary case: an
    unorganised library returns ``null`` and ``[]`` here for every image, and
    nothing downstream may read that as an error or as an incomplete setup.
    """
    datasets = _asset_datasets(asset)
    return {
        "experiment_id": (str(asset.experiment_id) if asset.experiment_id else None),
        "experiment_name": asset.experiment.name if asset.experiment_id else None,
        "dataset_ids": [str(dataset.id) for dataset in datasets],
        "dataset_names": [dataset.name for dataset in datasets],
    }


def serialize_asset_entry(asset) -> dict[str, Any]:
    """List-level payload for one Asset."""

    openable = get_asset_openable(asset, require=False)
    ngff_ready = asset_ngff_ready(asset)
    is_workable = openable is not None
    ngff_status = "ready" if ngff_ready else ("missing" if is_workable else "unavailable")
    return {
        "id": str(asset.id),
        "asset_id": str(asset.id),
        "display_name": asset.display_name,
        "original_filename": asset.original_filename,
        "notes": asset.notes,
        "width": asset.logical_width,
        "height": asset.logical_height,
        "depth": asset.logical_depth,
        "stored_depth": (openable.stored_depth if openable is not None else asset.logical_depth),
        "pixel_size_nm": asset.pixel_size_nm,
        "pixel_size_nm_z": asset.pixel_size_nm_z,
        # What the file declared, so the library card can tell "read from file"
        # from "entered by hand" without the whole rendition list.
        "file_declared_pixel_size_nm": file_declared_pixel_size_nm(asset),
        # The reader's conflict note when the file declared its scale twice and
        # the two declarations disagreed (ImageJ unit vs ResolutionUnit tag vs
        # the vendor's own tag).
        "file_declared_pixel_size_caveat": file_declared_pixel_size_caveat(asset),
        # Which tag supplied that number: the microscope's own record, Fiji, or
        # the baseline TIFF tags. Null on assets imported before the reader
        # recorded it.
        "file_declared_pixel_size_source": file_declared_pixel_size_source(asset),
        "metadata_summary": format_image_metadata_summary(
            width=asset.logical_width,
            height=asset.logical_height,
            depth=asset.logical_depth,
            pixel_size_nm=asset.pixel_size_nm,
            pixel_size_nm_z=asset.pixel_size_nm_z,
        ),
        "created_at": _isoformat(asset.created_at),
        "updated_at": _isoformat(asset.updated_at),
        "preprocess_stage": asset.preprocess_stage,
        "preprocess_progress": asset.preprocess_progress,
        "preprocess_error": asset.preprocess_error,
        "ngff_ready": ngff_ready,
        "ngff_url": f"/ngff/assets/{asset.id}.zarr" if ngff_ready else None,
        "ngff_status": ngff_status,
        "is_workable": is_workable,
        "can_open": is_workable,
        "can_view": ngff_ready,
        "can_segment": ngff_ready,
        **serialize_asset_grouping(asset),
    }


def serialize_rendition(rendition) -> dict[str, Any]:
    return {
        "id": str(rendition.id),
        "type": rendition.type,
        "derived_from": (str(rendition.derived_from_id) if rendition.derived_from_id else None),
        "storage_root": rendition.storage_root,
        "stored_path": rendition.stored_path,
        "path_exists": rendition.path_exists,
        "is_directory": rendition.is_directory,
        "stored_width": rendition.stored_width,
        "stored_height": rendition.stored_height,
        "stored_depth": rendition.stored_depth,
        "stored_channels": rendition.stored_channels,
        "stored_bit_depth": rendition.stored_bit_depth,
        "z_plane_indices": rendition.z_plane_indices or [],
        "metadata": rendition.metadata or {},
    }


def serialize_asset_detail(asset) -> dict[str, Any]:
    """Detail payload for one Asset: the list entry plus storage details."""

    openable = get_asset_openable(asset, require=False)
    payload = serialize_asset_entry(asset)
    payload.update(
        {
            "file_path": openable.file_path if openable is not None else "",
            "channels": asset.channels,
            "bit_depth": asset.bit_depth,
            "renditions": [serialize_rendition(rendition) for rendition in asset.renditions.all()],
        }
    )
    return payload


class ImageROISerializer(serializers.ModelSerializer):
    """Serializer for ROI images."""

    class Meta:
        model = ImageROI
        fields = [
            "id",
            "asset",
            "display_name",
            "x",
            "y",
            "width",
            "height",
            "source",
            "is_active",
            "is_complete",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]
