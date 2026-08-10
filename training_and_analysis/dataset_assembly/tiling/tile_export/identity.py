from __future__ import annotations

import hashlib
from typing import Any

from .config import TileExportConfig, stable_json


def derive_run_id(
    *,
    source_keys: list[str],
    config: TileExportConfig,
) -> str:
    payload = {
        "version": 1,
        "source_keys": sorted(str(key) for key in source_keys),
        "config": config.to_json_dict(),
    }
    return "tiles_" + hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()[:16]


def normalization_config_hash(
    config: TileExportConfig,
    *,
    low: float | None,
    high: float | None,
    method: str | None = None,
    scope: str | None = None,
    low_percentile: float | None = None,
    high_percentile: float | None = None,
    raw_dtype: str | None = None,
    tile_storage_dtype: str | None = None,
    inverted: bool = False,
    low_dynamic_range: bool = False,
) -> str:
    payload = {
        "normalization": method or config.normalization,
        "scope": scope or getattr(config, "normalization_scope", ""),
        "low_percentile": (
            None if low_percentile is None else round(float(low_percentile), 6)
        ),
        "high_percentile": (
            None if high_percentile is None else round(float(high_percentile), 6)
        ),
        "low": None if low is None else round(float(low), 6),
        "high": None if high is None else round(float(high), 6),
        "raw_dtype": raw_dtype or "",
        "tile_storage_dtype": tile_storage_dtype or "",
        "inverted": bool(inverted),
        "low_dynamic_range": bool(low_dynamic_range),
    }
    return hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()[:12]


def make_tile_id(
    *,
    source_id: str,
    asset_id: str,
    z: int | None,
    x: int,
    y: int,
    tile_size: int,
    effective_nm_per_px: float | None,
    normalization: str,
) -> str:
    payload: dict[str, Any] = {
        "source_id": str(source_id),
        "asset_id": str(asset_id),
        "z": None if z is None else int(z),
        "x": int(x),
        "y": int(y),
        "tile_size": int(tile_size),
        "effective_nm_per_px": (
            None if effective_nm_per_px is None else round(float(effective_nm_per_px), 9)
        ),
        "normalization": str(normalization),
    }
    return hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()


def seeded_digest(value: str, *, seed: int) -> str:
    return hashlib.sha1(f"{int(seed)}:{value}".encode("utf-8")).hexdigest()
