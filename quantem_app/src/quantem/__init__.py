"""QuantEM — organelle segmentation and quantitative analysis for electron microscopy."""

from __future__ import annotations

from quantem._version import FALLBACK_VERSION


def _resolve_version() -> str:
    """The application version, from its single source: ``[project]`` in pyproject.toml.

    Installed (wheel, sdist, or editable), the build backend has already copied
    that value into the distribution metadata, so read it back from there.
    Running from a checkout that was never installed — ``PYTHONPATH=src`` — there
    is no metadata, so read pyproject.toml itself rather than keeping a second
    copy of the number here to forget to bump.

    Frozen (PyInstaller), neither exists: the bundler strips the dist-info and
    ships no pyproject.toml, and walking the filesystem for one from inside an
    install directory can only find somebody *else's* pyproject.toml. So a
    frozen build goes straight to :data:`quantem._version.FALLBACK_VERSION`,
    the baked copy that a test pins to ``[project].version``. This is what
    keeps ``quantem_version`` in provenance manifests a real release number
    instead of ``0+unknown``.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("quantem-app")
    except PackageNotFoundError:
        pass

    import sys

    if getattr(sys, "frozen", False):
        # No metadata and no checkout in a frozen build; any pyproject.toml
        # near the executable belongs to whatever directory the app was
        # installed into, not to QuantEM.
        return FALLBACK_VERSION

    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject.open("rb") as fh:
            return str(tomllib.load(fh)["project"]["version"])
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        # No metadata and no pyproject: an unpacked wheel on a bare sys.path.
        # The baked copy is the release the wheel was built from.
        return FALLBACK_VERSION


__version__ = _resolve_version()
