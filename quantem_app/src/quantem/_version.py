"""The baked application version, for builds where no metadata survives.

``[project].version`` in ``pyproject.toml`` is the single *authoritative*
source of the version number. Installed copies read it back through
``importlib.metadata``; a checkout reads ``pyproject.toml`` itself. Neither
works in a frozen (PyInstaller) build: the bundler strips the dist-info and
does not ship ``pyproject.toml``, which is how release bundles came to stamp
``quantem_version: "0+unknown"`` into scientific provenance manifests.

So the number is baked here as a plain module constant that freezes with the
package. It is a *copy* and copies drift, which is why
``quantem.core.tests.test_version.test_the_baked_fallback_matches_pyproject``
fails the suite the moment this constant and ``pyproject.toml`` disagree --
the gate, not good intentions, is what keeps them equal.
"""

from __future__ import annotations

#: Must equal ``[project].version`` in ``quantem_app/pyproject.toml``.
FALLBACK_VERSION = "0.1.3"
