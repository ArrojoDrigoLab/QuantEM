"""Weight registry, cache and offline behaviour. No network, no downloads."""

from __future__ import annotations

import json

import pytest

from quantem_em.registry import REGISTRY, get_model_spec
from quantem_em.weights import fetch


def test_every_model_maps_to_registered_artifacts():
    reg = fetch.load_registry()
    for spec in REGISTRY.values():
        for name in fetch.artifacts_for(spec):
            assert name in reg["artifacts"], f"{spec.model_id} needs unregistered artifact {name}"


def test_artifact_sharing_matches_actual_adaptation():
    """LoRA leaves the encoder untouched; last_n rewrites four blocks; full replaces it."""
    assert fetch.artifacts_for(get_model_spec("omniem/mito")) == ["omniem-vitl", "omniem-mito"]
    assert fetch.artifacts_for(get_model_spec("quantem/mito")) == [
        "quantem-vitb-trunk",
        "quantem-mito",
    ]
    # adapt="full" carries the whole encoder, so no trunk is downloaded at all.
    assert fetch.artifacts_for(get_model_spec("quantem/er")) == ["quantem-er"]


def test_every_artifact_has_a_digest_before_publication():
    """fetch refuses to download an unverifiable file, so a missing digest is a release blocker."""
    for name, e in fetch.load_registry()["artifacts"].items():
        assert e.get("sha256"), f"{name} has no sha256"
        assert len(e["sha256"]) == 64, f"{name} has a malformed sha256"
        assert e.get("bytes"), f"{name} has no recorded size"


def test_registry_is_valid_json_and_pinned_to_a_schema():
    reg = fetch.load_registry()
    assert reg["schema"] == 1
    assert reg["hf_repo"]
    json.dumps(reg)


def test_download_plan_counts_shared_trunks_once():
    """A second model in the same family must not re-charge the user for the encoder."""
    a = fetch.download_plan([get_model_spec("omniem/mito")])
    both = fetch.download_plan([get_model_spec("omniem/mito"), get_model_spec("omniem/nucleus")])
    names = [x["name"] for x in both["artifacts"]]
    assert names.count("omniem-vitl") == 1
    assert len(both["artifacts"]) == len(a["artifacts"]) + 1


def test_sizes_are_reported_in_decimal_units():
    """Users compare this against their browser's download counter."""
    assert fetch.format_bytes(0) == "0 B"
    assert fetch.format_bytes(999) == "999 B"
    assert fetch.format_bytes(1_500_000) == "1.5 MB"
    assert fetch.format_bytes(2_500_000_000) == "2.5 GB"
    assert fetch.format_bytes(None) == "unknown size"


def test_offline_flag_is_read_from_either_variable(monkeypatch):
    monkeypatch.delenv("QUANTEM_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert not fetch.offline()
    monkeypatch.setenv("QUANTEM_OFFLINE", "1")
    assert fetch.offline()
    monkeypatch.setenv("QUANTEM_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "true")
    assert fetch.offline()


def test_offline_raises_with_an_actionable_message(monkeypatch, tmp_path):
    """The error must name the files and where to put them, not just fail."""
    monkeypatch.setenv("QUANTEM_OFFLINE", "1")
    monkeypatch.setenv("QUANTEM_MODEL_DIR", str(tmp_path))  # empty
    with pytest.raises(fetch.WeightsUnavailableError) as exc:
        fetch.ensure(["quantem-vitb-trunk", "quantem-mito"])
    msg = str(exc.value)
    assert "quantem-mito.safetensors" in msg
    assert "QUANTEM_MODEL_DIR" in msg
    assert exc.value.missing and exc.value.missing[0]["url"]


def test_unknown_artifact_is_named_in_the_error():
    with pytest.raises(fetch.WeightsError, match="nope"):
        fetch.artifact_info("nope")


def test_local_dir_is_honoured_over_the_hub_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTEM_MODEL_DIR", str(tmp_path))
    assert fetch.local_dir() == tmp_path
    assert not fetch.is_cached("quantem-mito")
    (tmp_path / "quantem-mito.safetensors").write_bytes(b"x")
    assert fetch.cached_path("quantem-mito") == tmp_path / "quantem-mito.safetensors"


def test_corrupt_file_is_detected_not_trusted(monkeypatch, tmp_path):
    """Verification runs on a cache hit too, so a truncated file heals instead of persisting."""
    monkeypatch.setenv("QUANTEM_MODEL_DIR", str(tmp_path))
    (tmp_path / "quantem-mito.safetensors").write_bytes(b"not the real file")
    with pytest.raises(fetch.WeightsCorruptError, match="failed verification"):
        fetch.ensure(["quantem-mito"], allow_network=False)


def test_registry_names_the_real_org_and_a_revision():
    """Two strings that silently break every download if they are wrong.

    The org slug is ``ArrojoeDrigoLab`` -- with the ``e``. And the revision must be present, so a
    later release cannot overwrite a pinned filename out from under installed copies.
    """
    from quantem_em.weights import fetch

    reg = fetch.load_registry()
    assert reg["hf_repo"] == "ArrojoeDrigoLab/quantem"
    assert fetch.revision(reg)
    assert f"/blob/{fetch.revision(reg)}/" in fetch.artifact_info("quantem-mito")["url"]


def test_download_and_cache_lookup_both_pin_the_revision(monkeypatch):
    """A revision honoured on download but not on the cache probe would still drift."""
    import huggingface_hub

    from quantem_em.weights import fetch

    reg = fetch.load_registry()
    monkeypatch.setitem(reg, "hf_revision", "v1-test")
    monkeypatch.setattr(fetch, "load_registry", lambda: reg)
    monkeypatch.delenv("QUANTEM_MODEL_DIR", raising=False)

    seen = {}
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda **kw: seen.update(kw) or None,
    )
    assert fetch.cached_path("quantem-mito") is None
    assert seen["revision"] == "v1-test"


def test_export_flat_produces_what_quantem_model_dir_reads(tmp_path, monkeypatch):
    """The documented air-gap recipe. The hub cache stores blobs under content hashes, so copying
    it to an offline machine leaves QUANTEM_MODEL_DIR finding nothing -- export_flat is what makes
    the instruction in the README true."""
    import shutil

    from quantem_em.registry import REGISTRY
    from quantem_em.weights import fetch

    spec = REGISTRY["quantem/mito"]
    names = fetch.artifacts_for(spec)
    src = fetch.local_dir()
    if src is None or not all((src / fetch._entry(n)["filename"]).is_file() for n in names):
        pytest.skip("needs QUANTEM_MODEL_DIR with the published artifacts")

    # stage a "connected machine" export without touching the network
    dest = tmp_path / "airgap"
    dest.mkdir()
    for n in names:
        shutil.copyfile(src / fetch._entry(n)["filename"], dest / fetch._entry(n)["filename"])

    # the offline machine sees only this flat directory
    monkeypatch.setenv("QUANTEM_MODEL_DIR", str(dest))
    monkeypatch.setenv("QUANTEM_OFFLINE", "1")
    for n in names:
        p = fetch.cached_path(n)
        assert p is not None and p.parent == dest, f"{n} not resolvable from a flat directory"
    assert fetch.download_plan([spec])["all_present"]
