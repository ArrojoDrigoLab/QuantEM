"""Manifest parsing + tile-path resolution."""

from __future__ import annotations

from pathlib import Path

from em_ssl.data.manifest import (
    build_source_run_index,
    candidate_tile_paths,
    infer_exports_root,
    iter_manifest,
    min_side,
    resolve_tile_path,
    tile_metadata,
)

def test_iter_manifest_parses_all(mini_corpus):
    recs = list(iter_manifest(mini_corpus["manifest_path"]))
    assert len(recs) == len(mini_corpus["records"])
    assert all("tile_id" in r for r in recs)

def test_infer_exports_root(mini_corpus):
    root = infer_exports_root(mini_corpus["manifest_path"])
    assert Path(root) == Path(mini_corpus["exports_root"])

def test_path_resolution_resolves_real_files(mini_corpus):
    exports_root = mini_corpus["exports_root"]
    n_ok = 0
    for rec in iter_manifest(mini_corpus["manifest_path"]):
        p = resolve_tile_path(rec, exports_root, verify=True)
        assert p is not None and p.exists(), rec["tile_id"]
        n_ok += 1
    assert n_ok == len(mini_corpus["records"])

def test_candidate_paths(mini_corpus):
    rec = mini_corpus["records"][0]
    cands = candidate_tile_paths(rec, mini_corpus["exports_root"])
    # The two relative forms (output_tile_path vs run_dir/tile_path) resolve to the same
    # file here and are deduplicated. A tile_root override adds distinct candidates.
    assert len(cands) >= 1
    assert all(str(c).endswith(".png") for c in cands)
    assert cands[0].exists()
    more = candidate_tile_paths(rec, mini_corpus["exports_root"], tile_root=mini_corpus["exports_root"] / "alt")
    assert len(more) > len(cands)

def test_source_run_index_resolves_missing_run_dir(tmp_path):
    # A manifest record may carry tile_path but no run_dir; the source->run index recovers it.
    er = tmp_path
    sid = "abc123"
    rel = f"tiles/source_id={sid}/t0.png"
    (er / "campaignX" / f"tiles/source_id={sid}").mkdir(parents=True)
    (er / "campaignX" / rel).write_bytes(b"\x89PNG")
    idx = build_source_run_index(er)
    assert sid in idx and "campaignX" in idx[sid]
    rec = {"source_id": sid, "tile_path": rel, "run_dir": None, "output_tile_path": None}
    p = resolve_tile_path(rec, er, verify=True, source_run_index=idx)
    assert p is not None and p.exists()
    # Without the index the record is unresolvable (no run_dir / output_tile_path).
    assert resolve_tile_path(rec, er, verify=True) is None

def test_min_side_and_metadata(mini_corpus):
    rec = mini_corpus["records"][9]  # the 400x400 tile
    assert min_side(rec) == 400
    meta = tile_metadata(rec)
    assert meta["source_id"] == rec["source_id"]
    assert "width" in meta and "tile_id" in meta
