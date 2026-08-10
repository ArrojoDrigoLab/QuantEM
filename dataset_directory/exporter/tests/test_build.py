"""End-to-end: extract in, published artifacts out, gate applied."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantem_directory import build as build_module
from quantem_directory import extract as extract_module
from quantem_directory import verify as verify_module


@pytest.fixture
def built(tmp_path: Path, extract_dir: Path, urls_csv: Path):
    corpus = extract_module.load(extract_dir, urls_csv=urls_csv)
    out = tmp_path / "data"
    report = build_module.build(corpus, out, thumb_ids=["a1", "a3"], source_snapshot="2026-01-01")
    return out, report


def _read(out: Path, name: str) -> dict:
    return json.loads((out / name).read_text(encoding="utf-8"))


class TestCounts:
    def test_corpus_totals(self, built):
        _out, report = built
        counts = report["counts"]
        assert counts["datasets"] == 3
        assert counts["assets"] == 5
        assert counts["images_2d"] + counts["volumes_3d"] == 5

    def test_dimensionality_uses_the_reconciled_rule(self, built):
        # a1 has a z-step and depth; a2 has neither despite living in a volume
        # dataset. Only the reconciled rule separates them.
        _out, report = built
        assert report["counts"]["volumes_3d"] == 1
        assert report["counts"]["images_2d"] == 4


class TestPrivacy:
    def test_withheld_groups_do_not_reach_the_output(self, built):
        out, _report = built
        facets = _read(out, "facets.json")
        assert "lab" not in facets["dictionaries"]
        assert "license" not in facets["dictionaries"]
        assert "Example Lab" not in (out / "facets.json").read_text(encoding="utf-8")

    def test_licence_is_absent_from_every_artifact(self, built):
        out, _report = built
        for name in ("facets.json", "datasets.json", "assets.json", "datasets.csv"):
            assert "CC BY 4.0" not in (out / name).read_text(encoding="utf-8")


class TestColumns:
    def test_every_column_is_the_same_length(self, built):
        out, _report = built
        assets = _read(out, "assets.json")
        n = assets["n"]
        assert all(len(c) == n for c in assets["columns"].values())
        assert all(len(c) == n for c in assets["single"].values())

    def test_multi_valued_facets_round_trip(self, built):
        out, _report = built
        assets = _read(out, "assets.json")
        facets = _read(out, "facets.json")
        kingdoms = assets["multi"]["kingdom"]
        assert len(kingdoms["offsets"]) == assets["n"] + 1
        assert kingdoms["offsets"][-1] == len(kingdoms["values"])

        # The host-and-symbiont asset must keep both kingdoms.
        vocabulary = facets["dictionaries"]["kingdom"]
        per_asset = [
            {vocabulary[v] for v in kingdoms["values"][kingdoms["offsets"][i]: kingdoms["offsets"][i + 1]]}
            for i in range(assets["n"])
        ]
        assert {"Animalia", "Bacteria"} in per_asset

    def test_a_missing_modality_is_null_not_a_fabricated_value(self, built):
        out, _report = built
        assets = _read(out, "assets.json")
        assert None in assets["single"]["modality"]

    def test_unknown_resolution_gets_a_real_band(self, built):
        out, _report = built
        assets = _read(out, "assets.json")
        facets = _read(out, "facets.json")
        bands = facets["dictionaries"]["resolution"]
        used = {bands[i] for i in assets["single"]["resolution"]}
        assert "Unknown" in used

    def test_asset_ids_are_undashed_hex_for_thumbnail_paths(self, built):
        out, _report = built
        assets = _read(out, "assets.json")
        assert all("-" not in i for i in assets["columns"]["id"])


class TestDatasets:
    def test_link_falls_back_to_the_experiment(self, built):
        out, _report = built
        rows = {r["name"]: r for r in _read(out, "datasets.json")["rows"]}
        assert rows["Human Islet TEM Series"]["url"] == "https://doi.org/10.6019/EMPIAR-12585"

    def test_an_undeposited_dataset_has_a_null_link(self, built):
        out, _report = built
        rows = {r["name"]: r for r in _read(out, "datasets.json")["rows"]}
        assert rows["Unpublished Adipose SEM"]["url"] is None

    def test_repository_is_derived_from_the_link(self, built):
        out, _report = built
        facets = _read(out, "facets.json")
        repositories = facets["dictionaries"]["repository"]
        rows = {r["name"]: r for r in _read(out, "datasets.json")["rows"]}
        assert repositories[rows["Mouse Liver FIB-SEM Volume"]["repository"]] == "Zenodo"
        assert repositories[rows["Human Islet TEM Series"]["repository"]] == "EMPIAR"
        assert repositories[rows["Unpublished Adipose SEM"]["repository"]] == "Not yet deposited"

    def test_counts_are_internally_consistent(self, built):
        out, _report = built
        rows = _read(out, "datasets.json")["rows"]
        assert all(r["n2d"] + r["n3d"] == r["n"] for r in rows)
        assert sum(r["n"] for r in rows) == _read(out, "assets.json")["n"]

    def test_heroes_come_only_from_assets_that_have_a_thumbnail(self, built):
        out, _report = built
        rows = {r["name"]: r for r in _read(out, "datasets.json")["rows"]}
        assert rows["Mouse Liver FIB-SEM Volume"]["hero"] == ["a1"]
        assert rows["Unpublished Adipose SEM"]["hero"] == []


class TestTrees:
    def test_a_child_may_appear_under_several_parents(self, built):
        # Neither taxonomy nor anatomy is a strict hierarchy in this corpus, so
        # the tree is built from co-occurrence and a value can repeat.
        out, _report = built
        facets = {f["key"]: f for f in _read(out, "facets.json")["facets"]}
        assert facets["taxonomy"]["kind"] == "tree"
        assert facets["anatomy"]["ranks"] == ["organ", "tissue context"]

    def test_tree_counts_are_per_parent(self, built):
        out, _report = built
        facets = {f["key"]: f for f in _read(out, "facets.json")["facets"]}
        animalia = next(r for r in facets["taxonomy"]["roots"] if r["label"] == "Animalia")
        assert animalia["n"] == 5
        assert {c["label"] for c in animalia["children"]} >= {"Mus musculus", "Homo sapiens"}


class TestVerify:
    def test_a_matching_expectation_passes(self, tmp_path, built):
        out, report = built
        expected = tmp_path / "expected.json"
        expected.write_text(json.dumps(report["counts"]), encoding="utf-8")
        verify_module.verify(out, expected_counts=expected)

    def test_a_drifted_count_fails(self, tmp_path, built):
        out, report = built
        counts = dict(report["counts"])
        counts["assets"] += 1
        expected = tmp_path / "expected.json"
        expected.write_text(json.dumps(counts), encoding="utf-8")
        with pytest.raises(verify_module.VerificationFailed) as raised:
            verify_module.verify(out, expected_counts=expected)
        assert "assets" in str(raised.value)

    def test_an_excluded_link_fails(self, tmp_path, built):
        out, report = built
        expected = tmp_path / "expected.json"
        expected.write_text(json.dumps(report["counts"]), encoding="utf-8")
        with pytest.raises(verify_module.VerificationFailed) as raised:
            verify_module.verify(
                out,
                expected_counts=expected,
                excluded={"url_substrings": ["zenodo.1"], "dataset_names": []},
            )
        assert "excluded resource" in str(raised.value)


class TestVocabularyOverrides:
    def test_a_value_can_be_dropped_without_losing_the_asset(self, tmp_path, extract_dir, urls_csv):
        corpus = extract_module.load(
            extract_dir,
            urls_csv=urls_csv,
            vocabulary_overrides={"Tissue Region": {"Hepatocyte": None}},
        )
        out = tmp_path / "data"
        report = build_module.build(corpus, out)
        assert report["counts"]["assets"] == 5
        assert "Hepatocyte" not in _read(out, "facets.json")["dictionaries"]["Tissue Region"]

    def test_an_undeposited_dataset_can_be_given_its_link_later(self, tmp_path, extract_dir, urls_csv):
        # In-house datasets get their public link as each deposition completes.
        # That has to work without a fresh corpus export, or the directory shows
        # "deposition pending" long after the deposition happened.
        corpus = extract_module.load(
            extract_dir,
            urls_csv=urls_csv,
            link_overrides={"Unpublished Adipose SEM": "https://doi.org/10.6019/S-BIAD9999"},
        )
        out = tmp_path / "data"
        build_module.build(corpus, out)
        rows = {r["name"]: r for r in _read(out, "datasets.json")["rows"]}
        assert rows["Unpublished Adipose SEM"]["url"] == "https://doi.org/10.6019/S-BIAD9999"
        repositories = _read(out, "facets.json")["dictionaries"]["repository"]
        assert repositories[rows["Unpublished Adipose SEM"]["repository"]] == "BioImage Archive"

    def test_a_value_can_be_renamed(self, tmp_path, extract_dir, urls_csv):
        corpus = extract_module.load(
            extract_dir,
            urls_csv=urls_csv,
            vocabulary_overrides={"organ": {"Adipose": "Adipose tissue"}},
        )
        out = tmp_path / "data"
        build_module.build(corpus, out)
        organs = _read(out, "facets.json")["dictionaries"]["organ"]
        assert "Adipose tissue" in organs and "Adipose" not in organs
