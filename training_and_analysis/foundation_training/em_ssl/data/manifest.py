"""Manifest parsing and robust tile-path resolution for the EM parent-tile corpus.

The master manifest is a JSONL file (one tile per line). It aggregates many successive
tiler runs, so each record carries both a run-relative path (`tile_path`,
relative to `run_dir`) and an exports-root-relative path (`output_tile_path`, which
already includes the `run_dir` prefix). The actual PNG files live under the exports
root — the grandparent of the manifest file:

    <exports_root>/manifests/parent_tiles.jsonl          # the manifest
    <exports_root>/<run_dir>/tiles/source_id=.../<tile>.png

Path resolution therefore tries, in order:
    tile_path, when it is already absolute
    exports_root / output_tile_path
    exports_root / run_dir / tile_path
    exports_root / <run dir named by the source_id index> / tile_path, for records that
        carry `tile_path` without `run_dir`
and, when a `tile_root` override is supplied, `tile_root / output_tile_path`,
`tile_root / run_dir / tile_path` and `tile_root / tile_path`. This keeps the loader
working whether the corpus is read in place or from a copied or renamed tree.

Nothing here imports torch — it is pure stdlib so validation/sharding tools are cheap.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

# There is no default manifest path: every entry point takes it on the command line,
# so a run cannot silently read an unintended corpus.
DEFAULT_MANIFEST_PATH = None

# Fields copied into each shard's per-tile JSON sidecar. Preserving source_id/asset_id keeps each
# tile's provenance inside the shard, which is what lets ``shard_writer`` balance shards by source
# and lets the FINO diagnostics group by source to detect provenance leakage. Deliberately compact
# rather than the whole record, to keep shards lean.
TILE_METADATA_FIELDS: tuple[str, ...] = (
    "tile_id",
    "source_id",
    "dataset_id",
    "asset_id",
    "source_kind",
    "run_id",
    "run_dir",
    "status",
    "width",
    "height",
    "tile_size",
    "effective_nm_per_px",
    # FINO metadata factors (discrete) — carried so the loader can build guide targets.
    "modality",
    "organ",
    # Logged diagnostics rather than guide objectives: fine tissue and species (organ is the objective).
    "tissue",
    "species",
    # Dataset license tag (verified) — carried for provenance and shard-time license filtering.
    "license",
    "normalization_method",
    "normalization_scope",
    "tile_storage_dtype",
    "low_dynamic_range",
    "normalization_warning",
    "artifact_fraction",
    "background_fraction",
    "tissue_score",
    "tile_mean_uint8",
    "tile_std_uint8",
    "tile_p01_uint8",
    "tile_p99_uint8",
    "inverted",
    "auto_reported_inverted",
)

def infer_exports_root(manifest_path: str | os.PathLike) -> Path:
    """Infer the exports root that `output_tile_path` is relative to.

    The manifest conventionally lives at ``<exports_root>/manifests/<name>.jsonl`` so
    the exports root is the grandparent. If the manifest is not under a ``manifests/``
    directory, the parent is used.
    """
    mp = Path(manifest_path).resolve()
    parent = mp.parent
    if parent.name.lower() == "manifests":
        return parent.parent
    return parent

def _join_posix(root: Path, rel: str) -> Path:
    """Join a POSIX-style relative path string onto an OS Path, cross-platform.

    Manifest paths use forward slashes; splitting on POSIX parts avoids backslash
    surprises on Windows and is a no-op on Linux.
    """
    return Path(root, *PurePosixPath(rel).parts)

def build_source_run_index(exports_root: str | os.PathLike) -> dict[str, list[str]]:
    """Map source_id -> [run_dir names] by scanning ``<exports_root>/<run>/tiles/source_id=*``.

    Needed because some manifest records carry ``tile_path`` (relative to a run dir) without
    ``run_dir`` / ``output_tile_path``. A source may appear in several runs (re-tiled),
    so all candidates are kept and path verification picks the one that holds the file.
    """
    exports_root = Path(exports_root)
    index: dict[str, list[str]] = {}
    for camp in sorted(exports_root.iterdir()):
        if not camp.is_dir() or camp.name.lower() in ("manifests",) or camp.name.startswith("_"):
            continue
        tiles_dir = camp / "tiles"
        if not tiles_dir.is_dir():
            continue
        try:
            with os.scandir(tiles_dir) as it:
                for entry in it:
                    if entry.is_dir() and entry.name.startswith("source_id="):
                        sid = entry.name[len("source_id=") :]
                        index.setdefault(sid, []).append(camp.name)
        except OSError:
            continue
    return index

def candidate_tile_paths(
    record: dict[str, Any],
    exports_root: Path,
    tile_root: Path | None = None,
    source_run_index: dict[str, list[str]] | None = None,
) -> list[Path]:
    """All plausible on-disk locations for a record's PNG, most-specific first."""
    cands: list[Path] = []
    tp = record.get("tile_path")
    otp = record.get("output_tile_path")
    run_dir = record.get("run_dir")

    # Absolute tile_path wins outright.
    if tp and PurePosixPath(tp).is_absolute():
        cands.append(Path(tp))
    if otp:
        cands.append(_join_posix(exports_root, otp))
    if run_dir and tp:
        cands.append(_join_posix(exports_root / run_dir, tp))
    # Records missing run_dir: resolve tile_path under each run dir
    # that the source_id index says contains this source.
    if tp and not run_dir and source_run_index is not None:
        for rd in source_run_index.get(str(record.get("source_id")), []):
            cands.append(_join_posix(exports_root / rd, tp))
    if tile_root is not None:
        if otp:
            cands.append(_join_posix(tile_root, otp))
        if run_dir and tp:
            cands.append(_join_posix(tile_root / run_dir, tp))
        if tp:
            cands.append(_join_posix(tile_root, tp))
    # De-duplicate while preserving order.
    seen: set[str] = set()
    uniq: list[Path] = []
    for c in cands:
        key = str(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def resolve_tile_path(
    record: dict[str, Any],
    exports_root: Path,
    tile_root: Path | None = None,
    verify: bool = False,
    source_run_index: dict[str, list[str]] | None = None,
) -> Path | None:
    """Resolve a record to a PNG path.

    If ``verify`` is False (default, fast) the first candidate is returned without a
    filesystem stat. If ``verify`` is True, the first *existing* candidate is returned,
    or ``None`` if none exist. For records missing ``run_dir``, pass ``source_run_index``
    (and prefer ``verify=True``, since several candidate run dirs may be tried).
    """
    cands = candidate_tile_paths(record, exports_root, tile_root, source_run_index)
    if not cands:
        return None
    if not verify:
        return cands[0]
    for c in cands:
        if c.exists():
            return c
    return None

def tile_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Compact, downstream-relevant metadata subset for a tile (shard sidecar)."""
    return {k: record.get(k) for k in TILE_METADATA_FIELDS if k in record}

def min_side(record: dict[str, Any]) -> int:
    """Shorter image side in pixels (0 if dimensions missing)."""
    w = record.get("width")
    h = record.get("height")
    if not w or not h:
        return 0
    return int(min(w, h))

def iter_manifest(manifest_path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL manifest (memory-flat, raises on malformed lines)."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Malformed JSON at line {lineno} of {manifest_path}: {exc}"
                ) from exc

@dataclass(frozen=True)
class ResolvedTile:
    """A filtered, path-resolved tile ready to be packed into a shard or loaded loose."""

    tile_id: str
    path: Path
    source_id: str
    metadata: dict[str, Any]

def iter_resolved_tiles(
    manifest_path: str | os.PathLike,
    predicate,
    exports_root: Path | None = None,
    tile_root: Path | None = None,
    verify_exists: bool = False,
    source_run_index: dict[str, list[str]] | None = None,
) -> Iterator[ResolvedTile]:
    """Yield ResolvedTile for every record passing ``predicate(record) -> bool``.

    ``predicate`` is typically an `em_ssl.data.filters.SSLTileFilter`. Path existence is
    only checked when ``verify_exists`` is True (slower; used when the resolved tiles are
    handed straight to ``shard_writer.build_shards``). ``source_run_index`` resolves records
    that lack ``run_dir``.
    """
    if exports_root is None:
        exports_root = infer_exports_root(manifest_path)
    for record in iter_manifest(manifest_path):
        if not predicate(record):
            continue
        path = resolve_tile_path(record, exports_root, tile_root, verify=verify_exists, source_run_index=source_run_index)
        if path is None:
            continue
        tile_id = record.get("tile_id")
        if not tile_id:
            # Fall back to a stable id derived from the output path.
            tile_id = PurePosixPath(record.get("output_tile_path", "")).stem
        yield ResolvedTile(
            tile_id=str(tile_id),
            path=path,
            source_id=str(record.get("source_id", "unknown")),
            metadata=tile_metadata(record),
        )
