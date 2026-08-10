from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

NORMALIZATION_NONE = "none"
NORMALIZATION_SOURCE_PERCENTILE_UINT8 = "source_percentile_uint8"
NORMALIZATION_TILE_PERCENTILE_UINT8 = "tile_percentile_uint8"
NORMALIZATION_MODES = {
    NORMALIZATION_NONE,
    NORMALIZATION_SOURCE_PERCENTILE_UINT8,
    NORMALIZATION_TILE_PERCENTILE_UINT8,
}
NORMALIZATION_SCOPE_SOURCE = "source"
NORMALIZATION_SCOPE_PLANE = "plane"
NORMALIZATION_SCOPE_TILE = "tile"
NORMALIZATION_SCOPES = {
    NORMALIZATION_SCOPE_SOURCE,
    NORMALIZATION_SCOPE_PLANE,
    NORMALIZATION_SCOPE_TILE,
}
INVERT_POLICY_NONE = "none"
INVERT_POLICY_AUTO_REPORT_ONLY = "auto_report_only"
INVERT_POLICIES = {
    INVERT_POLICY_NONE,
    INVERT_POLICY_AUTO_REPORT_ONLY,
}
SCORING_VERSION = "simple_tissue_score_v1"
TILE_EXPORT_OUTPUT_DIR_ENV = "QUANTEM_TILE_OUTPUT_DIR"


@dataclass(frozen=True)
class TileExportConfig:
    tile_size: int = 2048
    overlap_fraction: float = 0.15
    min_tissue_fraction: float = 0.50
    borderline_tissue_fraction: float = 0.25
    max_tiles_per_source: int = 400
    # ``max_tiles_per_source`` is the cap the selection step applies, so no single source
    # (e.g. a gigapixel 2D mosaic) dominates the training set. The two 3D limits below state the
    # same 400-tile budget for volumes and are range-checked by ``validate``; the driver derives
    # its own plane count from ``max_tiles_per_source`` divided by the tiles one plane yields, and
    # for a volume with a known z-spacing the minimum-spacing rule settles the planes instead.
    max_tiles_per_3d_volume: int = 400
    max_planes_per_3d_volume: int = 80
    seed: int = 1337
    # Whole-plane downsample used only as the analysis surface for tile scoring and
    # normalization estimation (never the exported tile). Cost scales with its area, so it
    # stays modest: a 2048-px tile still maps to >100 px of stats per side here.
    thumbnail_max_size: int = 2048
    normalization: str = NORMALIZATION_SOURCE_PERCENTILE_UINT8
    low_percentile: float = 0.1
    high_percentile: float = 99.9
    normalization_scope: str = NORMALIZATION_SCOPE_SOURCE
    invert_policy: str = INVERT_POLICY_AUTO_REPORT_ONLY
    invert_source_ids: tuple[str, ...] = ()
    invert_all: bool = False
    allow_low_dynamic_range: bool = False
    scoring: str = SCORING_VERSION

    @property
    def stride(self) -> int:
        stride = round(float(self.tile_size) * (1.0 - float(self.overlap_fraction)))
        return max(1, int(stride))

    @property
    def overlap_px(self) -> int:
        return max(0, int(self.tile_size) - int(self.stride))

    @property
    def max_shift_px(self) -> int:
        return int(self.overlap_px)

    def validate(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be > 0.")
        if not (0.0 <= self.overlap_fraction < 1.0):
            raise ValueError("overlap_fraction must be in [0, 1).")
        if not (0.0 <= self.borderline_tissue_fraction <= self.min_tissue_fraction <= 1.0):
            raise ValueError(
                "Expected 0 <= borderline_tissue_fraction <= min_tissue_fraction <= 1."
            )
        if self.max_tiles_per_source <= 0:
            raise ValueError("max_tiles_per_source must be > 0.")
        if self.max_tiles_per_3d_volume <= 0:
            raise ValueError("max_tiles_per_3d_volume must be > 0.")
        if self.max_planes_per_3d_volume <= 0:
            raise ValueError("max_planes_per_3d_volume must be > 0.")
        if self.thumbnail_max_size <= 0:
            raise ValueError("thumbnail_max_size must be > 0.")
        if self.normalization not in NORMALIZATION_MODES:
            valid = ", ".join(sorted(NORMALIZATION_MODES))
            raise ValueError(f"normalization must be one of: {valid}.")
        if self.normalization_scope not in NORMALIZATION_SCOPES:
            valid = ", ".join(sorted(NORMALIZATION_SCOPES))
            raise ValueError(f"normalization_scope must be one of: {valid}.")
        if not (0.0 <= self.low_percentile < self.high_percentile <= 100.0):
            raise ValueError("Expected 0 <= low_percentile < high_percentile <= 100.")
        if self.invert_policy not in INVERT_POLICIES:
            valid = ", ".join(sorted(INVERT_POLICIES))
            raise ValueError(f"invert_policy must be one of: {valid}.")
        if self.invert_all and self.invert_source_ids:
            raise ValueError("invert_all cannot be combined with invert_source_ids.")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stride"] = self.stride
        payload["overlap_px"] = self.overlap_px
        payload["max_shift_px"] = self.max_shift_px
        return payload

    def identity_json_dict(self) -> dict[str, Any]:
        """The fields that define tile identity; the digest is taken over this."""
        return self.to_json_dict()

    def digest(self) -> str:
        return stable_digest(self.identity_json_dict())


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def stable_digest(payload: Any, *, length: int | None = 12) -> str:
    value = hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()
    return value if length is None else value[:length]


def default_tile_export_output_root() -> Path:
    configured = os.environ.get(TILE_EXPORT_OUTPUT_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    return Path("tiles").resolve(strict=False)


def resolve_output_root(path: Path | str | None = None) -> Path:
    return Path(path or default_tile_export_output_root()).expanduser().resolve(strict=False)


def ensure_repo_output_path(path: Path, *, root: Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    root_resolved = Path(root).expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Output path must stay under {root_resolved}: {resolved}") from exc
    return resolved
