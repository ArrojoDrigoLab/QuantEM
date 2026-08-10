"""Dataset / manifest / shard fingerprints for reproducibility.

The dataset_fingerprint is a content hash over the sorted kept tile_ids plus the filter policy
and the manifest file hash. ``em_ssl.tools.build_shards`` writes it to ``dataset_fingerprint.json``
in the data bundle, and a training run copies the tile-id and shard hashes onto the encoder
manifest in its ``checkpoint_index.json``. Two runs with the same fingerprint trained on provably
the same data selection, and because the hashes travel with the checkpoint, a later comparison of
their encoders can be checked against the corpus each was trained on.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def sha256_strings(items: Iterable[str]) -> str:
    """Order-independent content hash over a collection of strings (e.g. tile_ids)."""
    h = hashlib.sha256()
    for s in sorted(items):
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def manifest_fingerprint(manifest_path: str | os.PathLike, hash_content: bool = True) -> dict[str, Any]:
    p = Path(manifest_path)
    st = p.stat()
    fp: dict[str, Any] = {
        "path": str(p),
        "size_bytes": st.st_size,
        "mtime": int(st.st_mtime),
    }
    if hash_content:
        fp["sha256"] = sha256_file(p)
    return fp

def dataset_fingerprint(
    manifest_path: str | os.PathLike,
    kept_tile_ids: Iterable[str],
    filter_config: dict[str, Any],
    mean: float | None = None,
    std: float | None = None,
    extra: dict[str, Any] | None = None,
    hash_manifest_content: bool = True,
) -> dict[str, Any]:
    ids = list(kept_tile_ids)
    fp = {
        "manifest": manifest_fingerprint(manifest_path, hash_content=hash_manifest_content),
        "filter_config": filter_config,
        "num_tiles": len(ids),
        "tile_ids_sha256": sha256_strings(ids),
        "normalization": {"mean": mean, "std": std},
    }
    if extra:
        fp.update(extra)
    return fp

def shard_fingerprint(shard_index: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint a built shard set from its shard_index (list of per-shard sha256)."""
    shards = shard_index.get("shards", [])
    shas = [s.get("sha256", "") for s in shards]
    return {
        "num_shards": len(shards),
        "num_tiles": shard_index.get("num_tiles"),
        "seed": shard_index.get("seed"),
        "balance_by_source": shard_index.get("balance_by_source"),
        "shards_sha256": sha256_strings(shas),
    }

def write_json(obj: Any, path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return p
