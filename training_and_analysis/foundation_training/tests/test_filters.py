"""SSL filtering rules + reason-coded exclusion accounting."""

from __future__ import annotations

from em_ssl.data.filters import SSLFilterConfig, SSLTileFilter
from em_ssl.data.manifest import iter_manifest

def _summary(mini_corpus, cfg):
    filt = SSLTileFilter(cfg)
    kept = [r for r in iter_manifest(mini_corpus["manifest_path"]) if filt(r)]
    return filt, kept

def test_default_min512_keeps_accepted_and_benign_warning(mini_corpus):
    filt, kept = _summary(mini_corpus, SSLFilterConfig(min_side=512))
    # 8 plain accepted + 1 contrast-inverted (benign) = 9; excludes 400px, rejected, ldr-blocked
    assert filt.kept == mini_corpus["n_accepted_ge512"] == 9
    ids = {r["tile_id"] for r in kept}
    assert "tid008" in ids  # auto_reported_contrast_inverted kept
    assert "tid009" not in ids  # 400px excluded by min_side
    assert "tid010" not in ids  # rejected
    assert "tid011" not in ids  # low_dynamic_range + insufficient_valid_support blocked

def test_contrast_inverted_is_not_blocked_without_min_side(mini_corpus):
    filt, kept = _summary(mini_corpus, SSLFilterConfig(min_side=0))
    ids = {r["tile_id"] for r in kept}
    assert "tid008" in ids  # benign warning never blocks

def test_exclusion_reasons_accounted(mini_corpus):
    filt, _ = _summary(mini_corpus, SSLFilterConfig(min_side=512))
    reasons = filt.summary()["excluded_by_reason"]
    assert reasons.get("status!=accepted", 0) == 1
    assert reasons.get("min_side<512", 0) == 1
    # ldr tile is blocked by the low_dynamic_range check (which runs before the warning check)
    assert filt.summary()["excluded_total"] == 3

def test_low_dynamic_range_block_can_be_disabled(mini_corpus):
    cfg = SSLFilterConfig(min_side=0, exclude_low_dynamic_range=False, blocking_warning_tokens=frozenset())
    filt, kept = _summary(mini_corpus, cfg)
    ids = {r["tile_id"] for r in kept}
    assert "tid011" in ids  # now kept
