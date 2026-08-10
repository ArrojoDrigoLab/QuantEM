"""Shard creation, reproducibility, single-channel readback, and the runtime min-side filter."""

from __future__ import annotations

from em_ssl.data.shard_dataset import EMShardDataset, list_shard_urls
from em_ssl.data.shard_writer import build_shards

def test_shards_created(built_shards):
    res = built_shards["result"]
    assert res.num_tiles == built_shards["n"] == 9
    assert res.num_shards >= 1
    urls = list_shard_urls(built_shards["shard_dir"], "em_tiles_v0")
    assert len(urls) == res.num_shards

def test_shard_readback_single_channel(built_shards):
    urls = list_shard_urls(built_shards["shard_dir"], "em_tiles_v0")
    ds = EMShardDataset(urls, transform=None, resampled=False, shuffle_buffer=0,
                        split_by_node=False, split_by_worker=False)
    seen = []
    for img, meta in ds:
        assert img.mode == "L"  # single channel, never RGB
        assert "source_id" in meta and "tile_id" in meta
        seen.append(meta["tile_id"])
    assert len(seen) == built_shards["n"]
    assert len(set(seen)) == built_shards["n"]

def test_shards_reproducible(built_shards, mini_corpus, tmp_path):
    # Rebuild from the same tiles with the same seed -> identical per-shard sha256.
    res2 = build_shards(
        built_shards["tiles"], tmp_path / "shards2", shard_prefix="em_tiles_v0",
        samples_per_shard=4, seed=1337, balance_by_source=True, progress=False,
    )
    a = [s.sha256 for s in built_shards["result"].shards]
    b = [s.sha256 for s in res2.shards]
    assert a == b and len(a) >= 1

def test_min_side_runtime_filter(built_shards):
    urls = list_shard_urls(built_shards["shard_dir"], "em_tiles_v0")
    ds = EMShardDataset(urls, transform=None, min_side=10_000, resampled=False,
                        shuffle_buffer=0, split_by_node=False, split_by_worker=False)
    assert list(ds) == []  # nothing passes an impossible min_side
