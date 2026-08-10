"""Packaging invariants.

These catch a failure mode that no other test can see: code that works perfectly from a source
checkout but is broken for everyone who installs it, because a data file was never declared as
package data. ``registry.json`` was genuinely missing from the first wheel built here.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_registry_json_is_declared_as_package_data(cfg):
    data = cfg["tool"]["setuptools"]["package-data"]
    assert "registry.json" in data.get("quantem_em.weights", []), (
        "registry.json must be declared, or pip-installed copies cannot resolve any model"
    )


def test_registry_json_is_findable_at_runtime():
    """The path fetch.py actually uses -- not a guess at where it should be."""
    from quantem_em.weights import fetch

    assert fetch._REGISTRY_PATH.is_file()
    assert fetch.load_registry()["artifacts"]


def test_declared_licence_has_a_file(cfg):
    """A licence classifier with no LICENSE file is a release blocker, not a detail."""
    declared = cfg["project"]["license"]
    assert declared, "no licence declared"
    candidates = [p.name for p in ROOT.iterdir() if p.name.upper().startswith("LICEN")]
    assert candidates, f"pyproject declares {declared} but there is no LICENSE file"


def test_version_is_not_a_dev_placeholder_when_tagged(cfg):
    """Guards against publishing 0.1.0.dev0 by accident.

    Only enforced when RELEASE=1, so day-to-day development is unaffected.
    """
    import os

    if not os.environ.get("RELEASE"):
        pytest.skip("set RELEASE=1 to enforce release-version rules")
    v = cfg["project"]["version"]
    assert "dev" not in v and not v.endswith("0.0"), f"refusing to release version {v!r}"


def test_no_direct_url_dependencies(cfg):
    """PyPI rejects any distribution declaring a URL dependency, and the whole reason the encoder
    goes through timm is to avoid needing one. Make that impossible to regress."""
    deps = cfg["project"]["dependencies"] + [
        d for group in cfg["project"].get("optional-dependencies", {}).values() for d in group
    ]
    for d in deps:
        assert "@" not in d and "://" not in d, (
            f"direct-reference dependency would be rejected: {d}"
        )


def test_core_declares_no_gui_dependencies(cfg):
    """quantem-core must stay installable without napari or Qt."""
    deps = " ".join(cfg["project"]["dependencies"]).lower()
    for forbidden in ("napari", "qtpy", "pyqt", "pyside", "magicgui"):
        assert forbidden not in deps, f"{forbidden} must not be a quantem-core dependency"
