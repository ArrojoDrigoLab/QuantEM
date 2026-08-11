"""Payload helpers shared by every job handler.

Split out of ``quantem/jobs/handlers.py`` so the handler modules that grow can
grow independently. Nothing here reads a job type or touches the registry.
"""

from quantem.assets.models import Asset


def _asset_for_payload(payload: dict) -> Asset | None:
    asset_id = str(payload.get("asset_id") or "").strip()
    if asset_id:
        return Asset.objects.filter(id=asset_id).first()
    return None


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)
