"""Frozen-build provenance: no borrowed git identity, no falsely-absent packages.

UAT round 13, findings 2 and 3, both observed in a real release bundle written
by the frozen build:

* the sidecar's git discovery walked UP from the install directory into the
  repository the app happened to be unzipped inside, and stamped THAT repo's
  ``git_worktree_clean: false, git_uncommitted_files: 11`` into a scientific
  manifest -- somebody else's dirty checkout presented as the app's identity;
* ``environment.packages`` reported torch/scipy/scikit-image/pandas/shapely as
  ``null`` "not installed", in the same manifest whose ``skimage_note`` says
  scikit-image sits under every measurement -- PyInstaller strips the dist-info
  that ``importlib.metadata`` reads, while the libraries themselves are right
  there.

These tests simulate the frozen interpreter (``sys.frozen``) and the stripped
metadata (``PackageNotFoundError``) in-process; the assertions are exactly what
the UAT read out of the real sidecar.
"""

from __future__ import annotations

import sys
from importlib import metadata as importlib_metadata

import pytest

from quantem.analysis import provenance


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)


def test_a_frozen_build_records_git_as_not_applicable(frozen):
    """This test tree IS inside a git repository -- exactly the trap: an
    unfrozen run would find it. Frozen must refuse to look at all."""
    out = provenance.release()

    assert out["git_commit"] is None
    assert out["git_worktree_clean"] is None
    assert "git_uncommitted_files" not in out
    reason = out["unavailable"]["git_commit"]
    assert "frozen" in reason or "packaged" in reason
    assert "installed inside" in reason


def test_a_frozen_build_still_names_a_real_version(frozen):
    assert provenance.release()["quantem_version"] != "0+unknown"


def test_repo_root_refuses_to_answer_when_frozen(frozen):
    assert provenance._repo_root() is None


def test_an_unfrozen_checkout_still_records_its_git_identity():
    """The development-tree behaviour is unchanged: this checkout has a .git
    above the package, and release() must keep reporting it."""
    repo = provenance._repo_root()
    if repo is None:  # an installed copy running the suite; nothing to pin
        pytest.skip("not running from a checkout")
    out = provenance.release()
    assert ("git_commit" in out and out["git_commit"]) or (
        "git_commit" in out.get("unavailable", {})
    )


def test_an_installed_copy_does_not_borrow_the_enclosing_repository(monkeypatch, tmp_path):
    """The pip channel's version of the frozen trap, found in round-14 validation.

    A venv created inside somebody's checkout is completely ordinary. The walk
    up from ``quantem.__file__`` used to sail past ``site-packages`` and stamp
    that stranger's commit and dirty state into the user's manifests. Here the
    package sits in a site-packages under a repository that is not it.
    """
    (tmp_path / ".git").mkdir()
    installed = tmp_path / ".venv" / "Lib" / "site-packages" / "quantem" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")

    import quantem

    monkeypatch.setattr(quantem, "__file__", str(installed))

    assert provenance._repo_root() is None


def _stripped_metadata(monkeypatch):
    def refuse(name: str) -> str:
        raise importlib_metadata.PackageNotFoundError(name)

    # ``provenance`` did ``from importlib import metadata``, so this patches
    # the exact object it calls through.
    monkeypatch.setattr(provenance.metadata, "version", refuse)


def test_packages_fall_back_to_the_modules_own_version(monkeypatch):
    """With every dist-info stripped, the versions still come from the
    imported libraries instead of being declared not installed."""
    import numpy

    _stripped_metadata(monkeypatch)
    env = provenance.environment()

    assert env["packages"]["numpy"] == numpy.__version__
    assert "packages.numpy" not in env.get("unavailable", {})
    # scikit-image is the one the manifest itself contradicted.
    import skimage

    assert env["packages"]["scikit-image"] == skimage.__version__
    assert "packages.scikit-image" not in env.get("unavailable", {})


def test_a_truly_absent_package_is_still_null_with_the_reason(monkeypatch):
    """The fallback must not fabricate: no metadata AND no importable module
    keeps the honest null + sentence."""
    _stripped_metadata(monkeypatch)
    monkeypatch.setitem(
        provenance._DISTRIBUTION_MODULES, "torch", "definitely_not_a_module_qx"
    )

    env = provenance.environment()

    assert env["packages"]["torch"] is None
    assert "not installed" in env["unavailable"]["packages.torch"]
