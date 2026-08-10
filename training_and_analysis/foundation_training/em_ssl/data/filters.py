"""SSL tile-filtering rules with configurable thresholds and full exclusion accounting.

Design principle: the *library* defaults are permissive (only reject what is clearly
unusable); *configs* set the actual policy and are logged per run. Every excluded tile
is counted by reason so build_shards can report exactly what was
dropped and why — nothing is silently discarded.

Key domain facts driving the defaults:
  * 306,814 / 327,793 records are status=="accepted" (tools stream the manifest, so counts are
    recomputed per run).
  * Only 8 records are low_dynamic_range==true.
  * 75,392 records carry normalization_warning=="auto_reported_contrast_inverted" — a benign
    heuristic flag (entries still have inverted==false). It is not a default rejection reason;
    contrast inversion is harmless for SSL and partly absorbed by the brightness/contrast
    augmentation.
  * Short-side coverage: >=512: 327,118 · >=768: 325,191 · >=1024: 321,253. So a single
    shard set filtered at min_side>=512 serves all crop sizes; per-experiment 768/1024
    runs additionally require the larger short side at load time.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

# Warning *tokens* (warnings are ';'-joined) that should block a tile by default.
# Deliberately excludes "auto_reported_contrast_inverted".
DEFAULT_BLOCKING_WARNING_TOKENS: frozenset[str] = frozenset(
    {"insufficient_valid_support", "low_dynamic_range"}
)

@dataclass
class SSLFilterConfig:
    """Configurable SSL training-tile filter policy.

    A threshold of ``None`` disables that check (permissive). Set explicit values in the
    data config to enforce policy. ``min_side`` is normally driven by the experiment's
    largest global crop size.
    """

    allowed_status: frozenset[str] = frozenset({"accepted"})
    required_dtype: str | None = "uint8"
    exclude_low_dynamic_range: bool = True
    blocking_warning_tokens: frozenset[str] = DEFAULT_BLOCKING_WARNING_TOKENS
    min_side: int = 0  # require min(width, height) >= min_side; 0 disables.
    max_artifact_fraction: float | None = None  # e.g. 0.5; None disables.
    min_tissue_score: float | None = None  # permissive by default; e.g. 0.1 to enable.
    max_background_fraction: float | None = None  # e.g. 0.95; None disables.
    allowed_source_kinds: frozenset[str] | None = None  # None = all kinds.
    # License whitelist: keep only tiles whose ``license`` is in this set; None disables the check.
    # ``license`` is the dataset's verified license tag, attached during manifest enrichment.
    allowed_licenses: frozenset[str] | None = None
    allow_unlicensed: bool = True  # when allowed_licenses is set, also keep tiles with empty/missing license.

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "SSLFilterConfig":
        if not d:
            return cls()
        kw: dict[str, Any] = {}
        for f in (
            "required_dtype",
            "exclude_low_dynamic_range",
            "min_side",
            "max_artifact_fraction",
            "min_tissue_score",
            "max_background_fraction",
        ):
            if f in d and d[f] is not None:
                kw[f] = d[f]
            elif f in d:
                kw[f] = None
        if "allowed_status" in d and d["allowed_status"] is not None:
            kw["allowed_status"] = frozenset(d["allowed_status"])
        if "blocking_warning_tokens" in d and d["blocking_warning_tokens"] is not None:
            kw["blocking_warning_tokens"] = frozenset(d["blocking_warning_tokens"])
        if d.get("allowed_source_kinds"):
            kw["allowed_source_kinds"] = frozenset(d["allowed_source_kinds"])
        if d.get("allowed_licenses"):
            kw["allowed_licenses"] = frozenset(d["allowed_licenses"])
        if "allow_unlicensed" in d and d["allow_unlicensed"] is not None:
            kw["allow_unlicensed"] = bool(d["allow_unlicensed"])
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_status": sorted(self.allowed_status),
            "required_dtype": self.required_dtype,
            "exclude_low_dynamic_range": self.exclude_low_dynamic_range,
            "blocking_warning_tokens": sorted(self.blocking_warning_tokens),
            "min_side": self.min_side,
            "max_artifact_fraction": self.max_artifact_fraction,
            "min_tissue_score": self.min_tissue_score,
            "max_background_fraction": self.max_background_fraction,
            "allowed_source_kinds": (
                sorted(self.allowed_source_kinds) if self.allowed_source_kinds else None
            ),
            "allowed_licenses": (
                sorted(self.allowed_licenses) if self.allowed_licenses else None
            ),
            "allow_unlicensed": self.allow_unlicensed,
        }

class SSLTileFilter:
    """Callable predicate over manifest records with reason-coded exclusion accounting.

    Usage:
        filt = SSLTileFilter(SSLFilterConfig(min_side=512))
        kept = [r for r in records if filt(r)]
        print(filt.summary())   # {'total': N, 'kept': K, 'excluded_total': E,
                                #  'excluded_by_reason': {reason: count}, 'config': {...}}
    """

    def __init__(self, config: SSLFilterConfig | None = None):
        self.config = config or SSLFilterConfig()
        self.total = 0
        self.kept = 0
        self.excluded: collections.Counter[str] = collections.Counter()

    def _warning_tokens(self, record: dict[str, Any]) -> list[str]:
        w = record.get("normalization_warning") or ""
        return [t for t in str(w).split(";") if t]

    def reason(self, record: dict[str, Any]) -> str | None:
        """Return the first failing reason for a record, or None if it passes.

        Order matters only for which reason is *reported*; a tile is excluded if any
        check fails.
        """
        c = self.config
        status = record.get("status")
        if c.allowed_status and status not in c.allowed_status:
            return f"status!={'|'.join(sorted(c.allowed_status))}"
        if c.allowed_source_kinds and record.get("source_kind") not in c.allowed_source_kinds:
            return "source_kind_excluded"
        if c.allowed_licenses is not None:
            lic = str(record.get("license") or "").strip()
            if not lic:
                if not c.allow_unlicensed:
                    return "license:unlicensed"
            elif lic not in c.allowed_licenses:
                return f"license:{lic}"
        if c.required_dtype and record.get("tile_storage_dtype") != c.required_dtype:
            return f"dtype!={c.required_dtype}"
        if c.exclude_low_dynamic_range and bool(record.get("low_dynamic_range")):
            return "low_dynamic_range"
        if c.blocking_warning_tokens:
            toks = self._warning_tokens(record)
            blocked = [t for t in toks if t in c.blocking_warning_tokens]
            if blocked:
                return f"warning:{blocked[0]}"
        if c.min_side > 0:
            w = record.get("width")
            h = record.get("height")
            if not w or not h or min(int(w), int(h)) < c.min_side:
                return f"min_side<{c.min_side}"
        if c.max_artifact_fraction is not None:
            af = record.get("artifact_fraction")
            if af is not None and af > c.max_artifact_fraction:
                return f"artifact_fraction>{c.max_artifact_fraction}"
        if c.min_tissue_score is not None:
            ts = record.get("tissue_score")
            if ts is not None and ts < c.min_tissue_score:
                return f"tissue_score<{c.min_tissue_score}"
        if c.max_background_fraction is not None:
            bf = record.get("background_fraction")
            if bf is not None and bf > c.max_background_fraction:
                return f"background_fraction>{c.max_background_fraction}"
        return None

    def __call__(self, record: dict[str, Any]) -> bool:
        self.total += 1
        r = self.reason(record)
        if r is None:
            self.kept += 1
            return True
        self.excluded[r] += 1
        return False

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "excluded_total": self.total - self.kept,
            "excluded_by_reason": dict(self.excluded.most_common()),
            "config": self.config.to_dict(),
        }
