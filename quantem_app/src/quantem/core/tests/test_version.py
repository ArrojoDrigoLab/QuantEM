"""``quantem.__version__`` must be a real release number in every build shape.

UAT round 13, finding 2: release bundles from the frozen build stamped
``quantem_version: "0+unknown"`` into scientific provenance manifests --
PyInstaller strips the dist-info ``importlib.metadata`` reads and ships no
pyproject.toml, so both sources of the version were gone. The fix bakes the
number into :mod:`quantem._version`; these tests pin the two facts that keep
that honest:

* the baked copy equals ``[project].version`` in pyproject.toml (the single
  authoritative source), so it cannot silently drift; and
* a frozen interpreter with no metadata resolves to the baked release, never
  to ``0+unknown`` and never by reading whatever pyproject.toml happens to sit
  near the install directory.
"""

from __future__ import annotations

import sys
import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import quantem
from quantem._version import FALLBACK_VERSION

PYPROJECT = Path(quantem.__file__).resolve().parents[2] / "pyproject.toml"


def test_the_baked_fallback_matches_pyproject():
    """The gate that makes the baked copy safe to keep."""
    with PYPROJECT.open("rb") as fh:
        authoritative = str(tomllib.load(fh)["project"]["version"])
    assert FALLBACK_VERSION == authoritative, (
        f"quantem/_version.py says {FALLBACK_VERSION!r} but pyproject.toml says "
        f"{authoritative!r}. Bump them together -- the baked copy is what a "
        "frozen build stamps into provenance manifests."
    )


def test_a_frozen_build_with_no_metadata_resolves_the_baked_release(
    monkeypatch, tmp_path
):
    """The exact frozen-build shape: no dist-info, no pyproject, sys.frozen."""

    def refuse(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", refuse)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    # Relocate the package to a directory with no pyproject.toml above it, as
    # in an install dir -- any pyproject found there would be somebody else's.
    fake_init = tmp_path / "app" / "_internal" / "quantem" / "__init__.py"
    fake_init.parent.mkdir(parents=True)
    monkeypatch.setattr(quantem, "__file__", str(fake_init))

    resolved = quantem._resolve_version()

    assert resolved == FALLBACK_VERSION
    assert resolved != "0+unknown"


def test_an_unpacked_wheel_on_a_bare_path_also_gets_the_baked_release(
    monkeypatch, tmp_path
):
    """No metadata, no pyproject, NOT frozen: the last-ditch shape that used
    to be the only way to reach '0+unknown'."""

    def refuse(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", refuse)
    fake_init = tmp_path / "site-packages" / "quantem" / "__init__.py"
    fake_init.parent.mkdir(parents=True)
    monkeypatch.setattr(quantem, "__file__", str(fake_init))

    assert quantem._resolve_version() == FALLBACK_VERSION
