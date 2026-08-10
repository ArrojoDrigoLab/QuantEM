"""A miniature corpus extract, so the pipeline is testable without the real data.

Shaped to exercise the cases that actually caused trouble: an asset with two
kingdoms and two species, an asset with no modality tag, an asset with no
parsable resolution, a dataset whose link comes from its experiment rather than
itself, and a dataset with no link at all.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

DATASETS = [
    # id, name, doi, experiment_name
    ("d1", "Mouse Liver FIB-SEM Volume", "10.5281/zenodo.1", "Mouse Liver FIB-SEM"),
    ("d2", "Human Islet TEM Series", "", "Human Islet TEM"),
    ("d3", "Unpublished Adipose SEM", "", "Mouse Adipose SEM"),
]

DATASET_NAMES = {i: n for i, n, _d, _e in DATASETS}
EXPERIMENT_NAMES = {i: e for i, _n, _d, e in DATASETS}

# dataset_id -> (source_url, experiment_doi) for the URL side table
URLS = {
    "d1": ("", ""),
    "d2": ("", "10.6019/EMPIAR-12585"),
    "d3": ("", ""),
}

ASSETS = [
    # id, name, dataset, w, h, depth, resolution_field, inplane, tiles
    ("a1", "liver volume 01", "d1", 2048, 2048, 310, "4x4x8nm", 4.0, 400),
    ("a2", "liver volume 02", "d1", 2048, 2048, 1, "4nm x 4nm", 4.0, 120),
    ("a3", "islet montage A", "d2", 4096, 4096, None, "1.59nm x 1.59nm", 1.59, 40),
    ("a4", "islet montage B", "d2", 4096, 4096, None, "", None, 12),
    ("a5", "adipose survey", "d3", 1024, 1024, None, "50 nm/pixel", 50.0, 3),
]

TAGS = [
    ("a1", "kingdom", "Animalia"),
    ("a1", "species", "Mus musculus"),
    ("a1", "organ", "Liver"),
    ("a1", "Tissue Region", "Hepatocyte"),
    ("a1", "modality", "FIB-SEM"),
    ("a1", "dimensionality", "3D"),
    ("a2", "kingdom", "Animalia"),
    ("a2", "species", "Mus musculus"),
    ("a2", "organ", "Liver"),
    ("a2", "modality", "FIB-SEM"),
    # Two kingdoms and two species on one asset: a host and its symbiont.
    ("a3", "kingdom", "Animalia"),
    ("a3", "kingdom", "Bacteria"),
    ("a3", "species", "Homo sapiens"),
    ("a3", "species", "Nesciobacter abundans"),
    ("a3", "organ", "Pancreas"),
    ("a3", "Tissue Region", "Pancreatic islet"),
    ("a3", "modality", "TEM"),
    ("a4", "kingdom", "Animalia"),
    ("a4", "species", "Homo sapiens"),
    ("a4", "organ", "Pancreas"),
    ("a4", "modality", "TEM"),
    # No modality tag at all, and a withheld group that must not surface.
    ("a5", "kingdom", "Animalia"),
    ("a5", "species", "Mus musculus"),
    ("a5", "organ", "Adipose"),
    ("a5", "lab", "Example Lab"),
    ("a5", "license", "CC BY 4.0"),
]


def _write(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def extract_dir(tmp_path: Path) -> Path:
    root = tmp_path / "extract"
    _write(
        root / "datasets.csv",
        ["dataset_id", "name", "origin", "catalog_source_key", "doi", "experiment_name", "experiment_origin"],
        [(i, n, "CATALOG", "zenodo", doi, exp, "CATALOG") for i, n, doi, exp in DATASETS],
    )
    _write(
        root / "asset_meta.csv",
        [
            "asset_id", "display_name", "width", "height", "depth", "resolution_field",
            "dataset_id", "dataset_name", "experiment_name",
        ],
        [
            # dataset_name and experiment_name are carried on the asset row in
            # the extract, so the fixture reproduces that shape.
            (a, name, w, h, d if d is not None else "", res, ds,
             DATASET_NAMES[ds], EXPERIMENT_NAMES[ds])
            for a, name, ds, w, h, d, res, _nm, _t in ASSETS
        ],
    )
    _write(root / "asset_tag_long.csv", ["asset_id", "group", "name"], TAGS)
    _write(
        root / "asset_tiles.csv",
        ["asset_id", "accepted_tiles", "has_tiles", "ctm_canonical_tiles", "tiles_summary_accepted"],
        [(a, t, 1, t, t) for a, _n, _ds, _w, _h, _d, _r, _nm, t in ASSETS],
    )
    _write(
        root / "derived" / "asset_tidy.csv",
        ["asset_id", "dim"],
        [(a, "3D" if d and d > 1 else "2D") for a, _n, _ds, _w, _h, d, _r, _nm, _t in ASSETS],
    )
    _write(
        root / "derived" / "asset_inplane_resolution_nm.csv",
        ["", "inplane_nm"],
        [(a, nm) for a, _n, _ds, _w, _h, _d, _r, nm, _t in ASSETS if nm is not None],
    )
    return root


@pytest.fixture
def urls_csv(tmp_path: Path) -> Path:
    path = tmp_path / "dataset_urls.csv"
    _write(
        path,
        ["dataset_id", "source_url", "doi", "dataset_doi", "experiment_doi"],
        [
            (i, URLS[i][0], "", next(d for x, _n, d, _e in DATASETS if x == i), URLS[i][1])
            for i, _n, _d, _e in DATASETS
        ],
    )
    return path
