from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from django.http import Http404

from quantem.assets.models import Asset, Rendition
from quantem.core.config import DATA_DIR, NGFF_TMP_DIR, STORAGE_DIR
from quantem.core.local_storage import (
    path_value_is_absolute_like,
    resolve_stored_path,
)

LOCAL_RENDITION_TYPES = (Rendition.TYPE_FULL, Rendition.TYPE_SUBSET)


@dataclass(frozen=True)
class AssetOpenable:
    """Read-only local storage handle derived from canonical Asset/Rendition rows."""

    asset: Asset
    rendition: Rendition
    path: Path

    @property
    def id(self):
        return self.asset.id

    @property
    def asset_id(self):
        return self.asset.id

    @property
    def file_path(self) -> str:
        return self.rendition.stored_path

    @property
    def absolute_path(self) -> str:
        return str(self.path)

    @property
    def display_name(self) -> str:
        return self.asset.display_name

    @property
    def original_filename(self) -> str:
        return self.asset.original_filename

    @property
    def created_at(self):
        return self.asset.created_at

    @property
    def width(self) -> int:
        return int(self.rendition.stored_width or self.asset.logical_width or 0)

    @property
    def height(self) -> int:
        return int(self.rendition.stored_height or self.asset.logical_height or 0)

    @property
    def channels(self) -> int:
        return int(self.rendition.stored_channels or self.asset.channels or 1)

    @property
    def bit_depth(self) -> int:
        return int(self.rendition.stored_bit_depth or self.asset.bit_depth or 8)

    @property
    def depth(self) -> int | None:
        return self.asset.logical_depth

    @property
    def stored_depth(self) -> int | None:
        return self.rendition.stored_depth or self.asset.logical_depth

    @property
    def z_plane_indices(self) -> list[int]:
        values = self.rendition.z_plane_indices or []
        return [int(value) for value in values if str(value).strip()]

    @property
    def z_sampling(self) -> dict:
        metadata = self.rendition.metadata or {}
        volume_metadata = metadata.get("volume_metadata")
        if isinstance(volume_metadata, dict):
            sampling = volume_metadata.get("z_sampling")
            if isinstance(sampling, dict):
                return dict(sampling)
        return {}

    @property
    def volume_metadata(self) -> dict:
        metadata = self.rendition.metadata or {}
        volume_metadata = metadata.get("volume_metadata")
        return dict(volume_metadata) if isinstance(volume_metadata, dict) else {}

    @property
    def is_volume(self) -> bool:
        return bool(self.depth and self.depth > 1)

    @property
    def has_stored_z_stack(self) -> bool:
        return bool(self.stored_depth and self.stored_depth > 1)


def local_rendition_queryset(asset: Asset):
    return asset.renditions.filter(type__in=LOCAL_RENDITION_TYPES).exclude(stored_path="")


def asset_has_local_rendition(asset: Asset) -> bool:
    prefetched = _prefetched_renditions(asset)
    if prefetched is not None:
        return any(_is_local_rendition(rendition) for rendition in prefetched)
    return local_rendition_queryset(asset).exists()


def get_asset_openable(asset: Asset, *, require: bool = True) -> AssetOpenable | None:
    rendition = select_local_rendition(asset)
    if rendition is None:
        if require:
            raise Http404(f"Asset {asset.id} does not have a local rendition")
        return None
    return AssetOpenable(asset=asset, rendition=rendition, path=resolve_rendition_path(rendition))


def get_asset_ngff_rendition(asset: Asset) -> Rendition | None:
    prefetched = _prefetched_renditions(asset)
    renditions: Iterable[Rendition]
    if prefetched is None:
        renditions = asset.renditions.filter(type=Rendition.TYPE_NGFF).exclude(stored_path="")
    else:
        renditions = [
            rendition
            for rendition in prefetched
            if rendition.type == Rendition.TYPE_NGFF and rendition.stored_path
        ]
    return _first_existing_rendition(renditions)


def get_asset_ngff_path(asset: Asset) -> Path | None:
    """The published generation's directory, or ``None``.

    A one-line shim over :func:`quantem.assets.pyramid_authority.resolve_pyramid`,
    kept because ``segmentation`` and the job artifact registry read it. It no
    longer *derives* anything: "there is a path" and "the pyramid may be read"
    are now the same question, answered in one place.
    """

    from .pyramid_authority import Intent, PublishedPyramid, resolve_pyramid

    resolved = resolve_pyramid(asset, intent=Intent.SERVE)
    return resolved.root if isinstance(resolved, PublishedPyramid) else None


def asset_ngff_ready(asset: Asset) -> bool:
    """Whether the viewer may open this asset.

    The frontend contract (``ngff_ready``/``can_view``/``can_segment`` in
    ``serializers.py``) is unchanged; what changed is that this is no longer a
    separate opinion computed from ``path.exists()``. It is the authority's
    answer, so the card, the viewer route and every reader cannot disagree.
    """

    from .pyramid_authority import Intent, PublishedPyramid, resolve_pyramid

    return isinstance(resolve_pyramid(asset, intent=Intent.SERVE), PublishedPyramid)


def select_local_rendition(asset: Asset) -> Rendition | None:
    prefetched = _prefetched_renditions(asset)
    if prefetched is None:
        renditions = local_rendition_queryset(asset)
    else:
        renditions = [rendition for rendition in prefetched if _is_local_rendition(rendition)]
    return _first_existing_rendition(renditions)


def resolve_rendition_path(rendition: Rendition) -> Path:
    if path_value_is_absolute_like(rendition.stored_path):
        raise ValueError(
            f"Rendition {rendition.id} has absolute stored_path; stored_path must be "
            "relative to storage_root"
        )
    if rendition.storage_root == "DATA_DIR":
        path = resolve_stored_path(rendition.stored_path, relative_to=DATA_DIR)
    elif rendition.storage_root == "STORAGE_DIR":
        path = resolve_stored_path(rendition.stored_path, relative_to=STORAGE_DIR)
    elif rendition.storage_root == "NGFF_TMP_DIR":
        path = resolve_stored_path(rendition.stored_path, relative_to=NGFF_TMP_DIR)
    else:
        path = resolve_stored_path(rendition.stored_path, relative_to=DATA_DIR)

    return path


def _prefetched_renditions(asset: Asset) -> list[Rendition] | None:
    cache = getattr(asset, "_prefetched_objects_cache", {})
    renditions = cache.get("renditions")
    return list(renditions) if renditions is not None else None


def _is_local_rendition(rendition: Rendition) -> bool:
    return rendition.type in LOCAL_RENDITION_TYPES and bool(rendition.stored_path)


def _first_existing_rendition(renditions: Iterable[Rendition]) -> Rendition | None:
    ordered = sorted(
        renditions,
        key=lambda rendition: (
            0 if rendition.path_exists else 1,
            LOCAL_RENDITION_TYPES.index(rendition.type)
            if rendition.type in LOCAL_RENDITION_TYPES
            else 99,
            rendition.created_at,
            str(rendition.id),
        ),
    )
    return ordered[0] if ordered else None
