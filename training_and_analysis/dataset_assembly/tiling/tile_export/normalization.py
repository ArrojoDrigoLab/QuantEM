from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .config import (
    INVERT_POLICY_AUTO_REPORT_ONLY,
    NORMALIZATION_NONE,
    NORMALIZATION_SCOPE_PLANE,
    NORMALIZATION_SCOPE_SOURCE,
    NORMALIZATION_SCOPE_TILE,
    NORMALIZATION_SOURCE_PERCENTILE_UINT8,
    NORMALIZATION_TILE_PERCENTILE_UINT8,
    TileExportConfig,
)
from .identity import normalization_config_hash


MAX_NORMALIZATION_SAMPLE_PIXELS = 1_000_000
MIN_MASKED_SAMPLE_PIXELS = 64


@dataclass(frozen=True)
class StorageNormalization:
    method: str
    scope: str
    tile_storage_dtype: str
    raw_dtype: str
    low_percentile: float | None
    high_percentile: float | None
    source_low_raw: float | None
    source_high_raw: float | None
    inverted: bool
    auto_reported_inverted: bool
    low_dynamic_range: bool
    normalization_sample_pixels: int
    normalization_support_fraction: float
    normalization_support_pixels: int
    normalization_excluded_padding_fraction: float
    normalization_excluded_artifact_fraction: float
    normalization_estimation_method: str
    normalization_warning: str
    normalization_hash: str

    def sidecar_payload(self) -> dict[str, Any]:
        return {
            "tile_storage_dtype": self.tile_storage_dtype,
            "method": self.method,
            "scope": self.scope,
            "low_percentile": self.low_percentile,
            "high_percentile": self.high_percentile,
            "source_low_raw": self.source_low_raw,
            "source_high_raw": self.source_high_raw,
            "inverted": self.inverted,
            "auto_reported_inverted": self.auto_reported_inverted,
            "low_dynamic_range": self.low_dynamic_range,
            "raw_dtype": self.raw_dtype,
            "normalization_sample_pixels": self.normalization_sample_pixels,
            "normalization_support_fraction": self.normalization_support_fraction,
            "normalization_support_pixels": self.normalization_support_pixels,
            "normalization_excluded_padding_fraction": self.normalization_excluded_padding_fraction,
            "normalization_excluded_artifact_fraction": self.normalization_excluded_artifact_fraction,
            "normalization_estimation_method": self.normalization_estimation_method,
            "normalization_warning": self.normalization_warning,
        }

    def flat_fields(self) -> dict[str, Any]:
        return {
            "normalization_method": self.method,
            "normalization_scope": self.scope,
            "tile_storage_dtype": self.tile_storage_dtype,
            "raw_dtype": self.raw_dtype,
            "low_percentile": self.low_percentile,
            "high_percentile": self.high_percentile,
            "source_low_raw": self.source_low_raw,
            "source_high_raw": self.source_high_raw,
            "inverted": self.inverted,
            "auto_reported_inverted": self.auto_reported_inverted,
            "low_dynamic_range": self.low_dynamic_range,
            "normalization_sample_pixels": self.normalization_sample_pixels,
            "normalization_support_fraction": self.normalization_support_fraction,
            "normalization_support_pixels": self.normalization_support_pixels,
            "normalization_excluded_padding_fraction": self.normalization_excluded_padding_fraction,
            "normalization_excluded_artifact_fraction": self.normalization_excluded_artifact_fraction,
            "normalization_estimation_method": self.normalization_estimation_method,
            "normalization_warning": self.normalization_warning,
            "normalization_hash": self.normalization_hash,
        }


def effective_normalization_scope(
    *,
    config: TileExportConfig,
    is_3d: bool,
) -> str:
    if config.normalization == NORMALIZATION_TILE_PERCENTILE_UINT8:
        return NORMALIZATION_SCOPE_TILE
    if is_3d and config.normalization_scope == NORMALIZATION_SCOPE_SOURCE:
        return NORMALIZATION_SCOPE_PLANE
    return config.normalization_scope


def explicit_inversion_enabled(
    *,
    config: TileExportConfig,
    source_id: str,
) -> bool:
    return bool(config.invert_all or str(source_id) in {str(value) for value in config.invert_source_ids})


def build_passthrough_normalization(
    *,
    config: TileExportConfig,
    raw_dtype: np.dtype | str,
    scope: str,
    inverted: bool,
) -> StorageNormalization:
    dtype_name = np.dtype(raw_dtype).name
    if dtype_name != "uint8":
        raise ValueError("normalization 'none' requires a raw uint8 source.")
    return _build_normalization(
        config=config,
        method=NORMALIZATION_NONE,
        scope=scope,
        raw_dtype=dtype_name,
        low_percentile=None,
        high_percentile=None,
        source_low_raw=None,
        source_high_raw=None,
        inverted=inverted,
        auto_reported_inverted=False,
        low_dynamic_range=False,
        sample_pixels=0,
        support_fraction=1.0,
        support_pixels=0,
        excluded_padding_fraction=0.0,
        excluded_artifact_fraction=0.0,
        estimation_method="none",
        warning="",
    )


def estimate_percentile_normalization(
    values: np.ndarray,
    *,
    config: TileExportConfig,
    raw_dtype: np.dtype | str,
    scope: str,
    inverted: bool,
    estimation_method_prefix: str,
) -> StorageNormalization:
    method = (
        NORMALIZATION_TILE_PERCENTILE_UINT8
        if scope == NORMALIZATION_SCOPE_TILE
        else NORMALIZATION_SOURCE_PERCENTILE_UINT8
    )
    dtype_name = np.dtype(raw_dtype).name
    support_values, warning, support_metrics = _estimation_values(values)
    if support_values.size == 0:
        low = 0.0
        high = 0.0
        sampled = support_values
        if support_metrics["finite_pixels"] == 0:
            warning = _join_warning(warning, "no_finite_pixels")
    else:
        sampled = _deterministic_sample(
            support_values,
            max_pixels=MAX_NORMALIZATION_SAMPLE_PIXELS,
            seed=config.seed,
        )
        low, high = np.percentile(sampled, [config.low_percentile, config.high_percentile])

    low_float = float(low)
    high_float = float(high)
    low_dynamic_range = high_float <= low_float + _dynamic_range_epsilon(dtype_name)
    if low_dynamic_range:
        warning = _join_warning(warning, "low_dynamic_range")
    if support_metrics["support_pixels"] < MIN_MASKED_SAMPLE_PIXELS:
        warning = _join_warning(warning, "insufficient_valid_support")

    auto_reported_inverted = False
    if (
        config.invert_policy == INVERT_POLICY_AUTO_REPORT_ONLY
        and not inverted
        and not low_dynamic_range
        and sampled.size > 0
    ):
        auto_reported_inverted = _appears_contrast_inverted(
            sampled,
            low=low_float,
            high=high_float,
        )
        if auto_reported_inverted:
            warning = _join_warning(warning, "auto_reported_contrast_inverted")

    estimation_method = f"{estimation_method_prefix}_masked_percentile"
    if support_values.size > sampled.size:
        estimation_method += "_sampled"

    return _build_normalization(
        config=config,
        method=method,
        scope=scope,
        raw_dtype=dtype_name,
        low_percentile=float(config.low_percentile),
        high_percentile=float(config.high_percentile),
        source_low_raw=low_float,
        source_high_raw=high_float,
        inverted=inverted,
        auto_reported_inverted=auto_reported_inverted,
        low_dynamic_range=low_dynamic_range,
        sample_pixels=int(sampled.size),
        support_fraction=float(support_metrics["support_fraction"]),
        support_pixels=int(support_metrics["support_pixels"]),
        excluded_padding_fraction=float(support_metrics["excluded_padding_fraction"]),
        excluded_artifact_fraction=float(support_metrics["excluded_artifact_fraction"]),
        estimation_method=estimation_method,
        warning=warning,
    )


def estimate_from_record(record: dict[str, Any]) -> StorageNormalization:
    return StorageNormalization(
        method=str(record.get("normalization_method") or record.get("normalization") or ""),
        scope=str(record.get("normalization_scope") or NORMALIZATION_SCOPE_SOURCE),
        tile_storage_dtype=str(record.get("tile_storage_dtype") or "uint8"),
        raw_dtype=str(record.get("raw_dtype") or "uint8"),
        low_percentile=_optional_float(record.get("low_percentile")),
        high_percentile=_optional_float(record.get("high_percentile")),
        source_low_raw=_optional_float(record.get("source_low_raw")),
        source_high_raw=_optional_float(record.get("source_high_raw")),
        inverted=bool(record.get("inverted")),
        auto_reported_inverted=bool(record.get("auto_reported_inverted")),
        low_dynamic_range=bool(record.get("low_dynamic_range")),
        normalization_sample_pixels=int(record.get("normalization_sample_pixels") or 0),
        normalization_support_fraction=float(record.get("normalization_support_fraction") or 0.0),
        normalization_support_pixels=int(record.get("normalization_support_pixels") or 0),
        normalization_excluded_padding_fraction=float(
            record.get("normalization_excluded_padding_fraction") or 0.0
        ),
        normalization_excluded_artifact_fraction=float(
            record.get("normalization_excluded_artifact_fraction") or 0.0
        ),
        normalization_estimation_method=str(record.get("normalization_estimation_method") or ""),
        normalization_warning=str(record.get("normalization_warning") or ""),
        normalization_hash=str(record.get("normalization_hash") or ""),
    )


def normalize_window_to_uint8(
    window: np.ndarray,
    *,
    normalization: StorageNormalization,
) -> np.ndarray:
    values = np.asarray(window)
    if values.ndim == 3:
        values = values[..., 0]

    if normalization.method == NORMALIZATION_NONE:
        if values.dtype != np.uint8:
            raise ValueError("normalization 'none' requires a raw uint8 source.")
        output = values.astype(np.uint8, copy=True)
    elif normalization.low_dynamic_range:
        output = np.zeros(values.shape, dtype=np.uint8)
    else:
        low = normalization.source_low_raw
        high = normalization.source_high_raw
        if low is None or high is None or high <= low:
            output = np.zeros(values.shape, dtype=np.uint8)
        else:
            scaled = (values.astype(np.float32) - float(low)) * (255.0 / float(high - low))
            output = np.rint(np.nan_to_num(np.clip(scaled, 0, 255), nan=0)).astype(np.uint8)

    if normalization.inverted:
        output = np.subtract(np.uint8(255), output, dtype=np.uint8)
    return output


def tile_uint8_stats(tile: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tile, dtype=np.uint8)
    if values.size == 0:
        return {
            "tile_mean_uint8": None,
            "tile_std_uint8": None,
            "tile_p01_uint8": None,
            "tile_p99_uint8": None,
        }
    return {
        "tile_mean_uint8": round(float(values.mean()), 6),
        "tile_std_uint8": round(float(values.std()), 6),
        "tile_p01_uint8": round(float(np.percentile(values, 1)), 6),
        "tile_p99_uint8": round(float(np.percentile(values, 99)), 6),
    }


def _build_normalization(
    *,
    config: TileExportConfig,
    method: str,
    scope: str,
    raw_dtype: str,
    low_percentile: float | None,
    high_percentile: float | None,
    source_low_raw: float | None,
    source_high_raw: float | None,
    inverted: bool,
    auto_reported_inverted: bool,
    low_dynamic_range: bool,
    sample_pixels: int,
    support_fraction: float,
    support_pixels: int,
    excluded_padding_fraction: float,
    excluded_artifact_fraction: float,
    estimation_method: str,
    warning: str,
) -> StorageNormalization:
    normalization_hash = normalization_config_hash(
        config,
        method=method,
        scope=scope,
        low=source_low_raw,
        high=source_high_raw,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        raw_dtype=raw_dtype,
        tile_storage_dtype="uint8",
        inverted=inverted,
        low_dynamic_range=low_dynamic_range,
    )
    return StorageNormalization(
        method=method,
        scope=scope,
        tile_storage_dtype="uint8",
        raw_dtype=raw_dtype,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        source_low_raw=source_low_raw,
        source_high_raw=source_high_raw,
        inverted=inverted,
        auto_reported_inverted=auto_reported_inverted,
        low_dynamic_range=low_dynamic_range,
        normalization_sample_pixels=sample_pixels,
        normalization_support_fraction=round(float(support_fraction), 6),
        normalization_support_pixels=int(support_pixels),
        normalization_excluded_padding_fraction=round(float(excluded_padding_fraction), 6),
        normalization_excluded_artifact_fraction=round(float(excluded_artifact_fraction), 6),
        normalization_estimation_method=estimation_method,
        normalization_warning=warning,
        normalization_hash=normalization_hash,
    )


def _estimation_values(values: np.ndarray) -> tuple[np.ndarray, str, dict[str, float | int]]:
    raw = np.asarray(values)
    if raw.ndim == 3:
        raw = raw[..., 0]
    values_float = raw.astype(np.float32, copy=False)
    finite_mask = np.isfinite(values_float)
    total_pixels = int(values_float.size)
    padding_mask = np.zeros(values_float.shape, dtype=bool)
    artifact_mask = ~finite_mask
    support_mask = finite_mask.copy()
    if values_float.ndim == 2:
        support_mask, padding_mask, artifact_mask = _content_support_masks(
            values_float,
            finite_mask=finite_mask,
        )

    support_values = values_float[support_mask]
    metrics = {
        "support_fraction": _fraction(int(np.count_nonzero(support_mask)), total_pixels),
        "support_pixels": int(support_values.size),
        "finite_pixels": int(np.count_nonzero(finite_mask)),
        "excluded_padding_fraction": _fraction(int(np.count_nonzero(padding_mask)), total_pixels),
        "excluded_artifact_fraction": _fraction(int(np.count_nonzero(artifact_mask)), total_pixels),
    }
    return support_values.astype(np.float32, copy=False), "", metrics


def _content_support_masks(
    values: np.ndarray,
    *,
    finite_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = values.shape
    support = finite_mask.copy()
    padding = np.zeros(values.shape, dtype=bool)
    artifact = ~finite_mask
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return support, padding, artifact

    low_ref, high_ref = np.percentile(finite_values, [0.5, 99.5])
    if high_ref <= low_ref:
        low_ref = float(np.min(finite_values))
        high_ref = float(np.max(finite_values))
    value_range = max(float(high_ref - low_ref), 1.0)
    near_black = values <= float(low_ref) + (0.01 * value_range)
    near_white = values >= float(high_ref) - (0.01 * value_range)
    extreme = near_black | near_white
    edge_seed = np.zeros(values.shape, dtype=bool)
    edge_seed[0, :] = extreme[0, :]
    edge_seed[-1, :] = extreme[-1, :]
    edge_seed[:, 0] |= extreme[:, 0]
    edge_seed[:, -1] |= extreme[:, -1]
    padding |= ndimage.binary_propagation(edge_seed, mask=extreme)

    row_extreme = extreme.mean(axis=1)
    col_extreme = extreme.mean(axis=0)
    row_std = np.nan_to_num(values.std(axis=1), nan=0.0)
    col_std = np.nan_to_num(values.std(axis=0), nan=0.0)
    constant_threshold = max(float(np.percentile(row_std, 10)) * 0.25, value_range * 0.001)
    border_rows = (row_extreme > 0.80) | (row_std <= constant_threshold)
    border_cols = (col_extreme > 0.80) | (col_std <= constant_threshold)

    y_indices = np.arange(height)
    x_indices = np.arange(width)
    top_border = border_rows & (y_indices < max(1, int(round(height * 0.10))))
    bottom_border = border_rows & (y_indices >= int(round(height * 0.90)))
    left_border = border_cols & (x_indices < max(1, int(round(width * 0.10))))
    right_border = border_cols & (x_indices >= int(round(width * 0.90)))
    padding[top_border | bottom_border, :] = True
    padding[:, left_border | right_border] = True

    bottom_start = int(round(height * 0.85))
    if bottom_start < height:
        bottom_extreme = extreme[bottom_start:, :]
        bottom_row_extreme = bottom_extreme.mean(axis=1)
        bottom_values = values[bottom_start:, :]
        bottom_row_std = np.nan_to_num(bottom_values.std(axis=1), nan=0.0)
        footer_rows = (bottom_row_extreme > 0.45) | (
            (bottom_row_extreme > 0.20) & (bottom_row_std <= constant_threshold * 2.0)
        )
        footer_indices = np.nonzero(footer_rows)[0] + bottom_start
        padding[footer_indices, :] = True

    if min(height, width) >= 8:
        low_norm, high_norm = np.percentile(finite_values, [1, 99])
        if high_norm > low_norm:
            normalized = np.clip((values - low_norm) / float(high_norm - low_norm), 0.0, 1.0)
            local_window = max(3, int(round(max(height, width) / 256.0)))
            if local_window % 2 == 0:
                local_window += 1
            mean = ndimage.uniform_filter(normalized, size=local_window, mode="nearest")
            mean2 = ndimage.uniform_filter(
                normalized * normalized,
                size=local_window,
                mode="nearest",
            )
            local_std = np.sqrt(np.maximum(mean2 - (mean * mean), 0.0))
            constant = local_std <= max(float(np.percentile(local_std, 10)), 0.002)
            artifact |= (extreme | constant) & (local_std <= 0.005)

    support &= ~(padding | artifact)
    return support, padding & finite_mask, artifact


def _deterministic_sample(values: np.ndarray, *, max_pixels: int, seed: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size <= max_pixels:
        return flat
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(flat.size, size=int(max_pixels), replace=False)
    return flat[indices]


def _fraction(count: int, total: int) -> float:
    return 0.0 if total <= 0 else float(count) / float(total)


def _dynamic_range_epsilon(dtype_name: str) -> float:
    try:
        dtype = np.dtype(dtype_name)
    except TypeError:
        return 1e-6
    if np.issubdtype(dtype, np.integer):
        return 1.0
    return 1e-6


def _appears_contrast_inverted(values: np.ndarray, *, low: float, high: float) -> bool:
    if high <= low:
        return False
    normalized = np.clip((np.asarray(values, dtype=np.float32) - low) / float(high - low), 0.0, 1.0)
    return bool(float(np.median(normalized)) >= 0.70 and float(np.mean(normalized)) >= 0.62)


def _join_warning(existing: str, next_value: str) -> str:
    if not existing:
        return next_value
    if not next_value or next_value in existing.split(";"):
        return existing
    return f"{existing};{next_value}"


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
