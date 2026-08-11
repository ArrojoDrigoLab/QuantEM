from __future__ import annotations

from django.http import Http404

from quantem.assets.asset_openable import AssetOpenable, get_asset_openable
from quantem.assets.models import Asset


def active_asset_queryset():
    return Asset.objects.exclude(lifecycle_status=Asset.LIFECYCLE_DELETED)


def get_active_asset(asset_id: str) -> Asset:
    try:
        return active_asset_queryset().get(id=asset_id)
    except Asset.DoesNotExist as exc:
        # No id in the sentence. This message reaches the user on nine served
        # routes, and a UUID identifies nothing they can see -- it is a datum,
        # so it belongs in a structured field, not in prose (ruling D7). The id
        # is still in the request they made and in the server log.
        raise Http404("That image is not in your library. It may have been deleted.") from exc


def get_openable_for_asset_id(asset_id: str, *, require: bool = True) -> AssetOpenable | None:
    return get_asset_openable(get_active_asset(asset_id), require=require)
