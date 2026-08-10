"""Shared pytest fixtures: a synthetic single-channel EM corpus + built shards.

Everything is CPU-only and self-contained (no network, no real manifest), so the suite runs
anywhere. Tiles are genuine grayscale PNGs laid out like the real tiler output so manifest
parsing / path resolution / filtering / sharding are exercised against a realistic structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

def _make_png(path: Path, h: int, w: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = (rng.normal(150, 40, size=(h, w)).clip(0, 255)).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

@pytest.fixture(scope="session")
def mini_corpus(tmp_path_factory) -> dict:
    """Build a small synthetic exports tree + JSONL manifest. Returns key paths/records."""
    root = tmp_path_factory.mktemp("exports")
    run_dir = "run-test"
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    records = []
    # 12 tiles: 3 sources; most 600x600 (>=512), one is 400x400 (below 512), one rejected, one low_dynamic_range.
    specs = [
        # (idx, source, h, w, status, ldr, warning)
        *[(i, f"src{i%3}", 600, 600, "accepted", False, "") for i in range(8)],
        (8, "src0", 600, 600, "accepted", False, "auto_reported_contrast_inverted"),  # benign -> kept
        (9, "src1", 400, 400, "accepted", False, ""),  # too small for 512
        (10, "src2", 600, 600, "rejected", False, ""),  # not accepted
        (11, "src2", 600, 600, "accepted", True, "low_dynamic_range;insufficient_valid_support"),  # blocked
    ]
    for idx, src, h, w, status, ldr, warning in specs:
        tile_id = f"tid{idx:03d}"
        rel = f"tiles/source_id={src}/{tile_id}.png"
        out_rel = f"{run_dir}/{rel}"
        _make_png(root / run_dir / rel, h, w, seed=idx)
        rec = {
            "tile_id": tile_id,
            "source_id": src,
            "asset_id": f"asset-{src}",
            "source_kind": "image_file",
            "run_id": run_dir,
            "run_dir": run_dir,
            "status": status,
            "width": w,
            "height": h,
            "tile_size": 2048,
            "effective_nm_per_px": 5.0,
            "normalization_method": "source_percentile_uint8",
            "normalization_scope": "source",
            "tile_storage_dtype": "uint8",
            # FINO metadata factors (varied; some None/invalid to exercise masking + canonicalization).
            "modality": ["FIB-SEM", "fibsem", "SBF-SEM", None, "TEM"][idx % 5],
            "organ": ["Brain", "Kidney", None, "Liver"][idx % 4],
            "tissue": ["brain_neuropil", "kidney_tubule", None, "liver_hepatocyte"][idx % 4],
            "dataset_id": f"ds{idx % 2}",
            "low_dynamic_range": ldr,
            "normalization_warning": warning,
            "artifact_fraction": 0.02,
            "background_fraction": 0.1,
            "tissue_score": 0.6,
            "tile_mean_uint8": 150.0,
            "tile_std_uint8": 40.0,
            "tile_path": rel,
            "output_tile_path": out_rel,
            "sidecar_path": rel.replace(".png", ".json"),
            "output_sidecar_path": out_rel.replace(".png", ".json"),
            "inverted": False,
            "auto_reported_inverted": False,
        }
        records.append(rec)
    manifest_path = manifests / "parent_tiles.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return {
        "exports_root": root,
        "manifest_path": manifest_path,
        "records": records,
        "n_accepted_ge512": 9,  # 8 plain + 1 contrast-inverted (kept); excludes 400px, rejected, ldr
    }

@pytest.fixture(scope="session")
def built_shards(mini_corpus, tmp_path_factory) -> dict:
    """Build shards from the accepted >=512 tiles of the mini corpus."""
    from em_ssl.data.filters import SSLFilterConfig, SSLTileFilter
    from em_ssl.data.manifest import iter_resolved_tiles
    from em_ssl.data.shard_writer import build_shards

    filt = SSLTileFilter(SSLFilterConfig(min_side=512))
    tiles = list(
        iter_resolved_tiles(
            mini_corpus["manifest_path"],
            predicate=filt,
            exports_root=mini_corpus["exports_root"],
            verify_exists=True,
        )
    )
    shard_dir = tmp_path_factory.mktemp("shards")
    result = build_shards(
        tiles, shard_dir, shard_prefix="em_tiles_v0", samples_per_shard=4, seed=1337,
        balance_by_source=True, progress=False,
    )
    return {"shard_dir": shard_dir, "result": result, "tiles": tiles, "n": len(tiles)}
