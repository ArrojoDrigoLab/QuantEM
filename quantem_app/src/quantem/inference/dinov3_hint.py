"""Where Meta's DINOv3 package might be, and whether the hint really points at it.

``QUANTEM_DINOV3_PATH`` is a **development-only** escape hatch (see
:mod:`quantem.inference.encoders`): QuantEM neither vendors nor depends on
Meta's package, and no shipped install needs it.

It gets a module of its own because two callers must answer the same question
and had drifted apart:

* :func:`quantem.inference.encoders._import_dinov3` puts the hint on
  ``sys.path`` and imports.
* :func:`quantem.registry.catalogue.probe_runnable` has to say *up front*
  whether that import would work, without importing
  :mod:`quantem.inference.encoders` -- that module imports torch at module
  scope and a ``GET /api/models/`` must stay cheap.

The importer accepted **any** directory and let the import decide; the probe
demanded a ``dinov3`` *subdirectory*. A hint naming a directory that holds a
``dinov3`` module rather than a ``dinov3/`` package therefore satisfied the
importer and failed the probe, and a hint naming a ``dinov3/`` directory with no
``__init__`` did the reverse -- the Models screen and the run path disagreeing
about the same environment variable.

The definition kept here is the importer's, because it is the one that has to be
true: **the hint is the directory to put on ``sys.path``**, and it is good
exactly when ``import dinov3`` would resolve from it. That is asked with the
same finder the import statement uses, so the answer cannot be a different
answer -- and nothing is imported to get it.

Standard library only, on purpose. Nothing here may grow a torch import.
"""

from __future__ import annotations

import importlib.machinery
import os
from pathlib import Path

#: Points at a checkout of https://github.com/facebookresearch/dinov3 -- the
#: directory that *contains* the ``dinov3`` package, i.e. what you would add to
#: ``sys.path``, not the package directory itself.
DINOV3_PATH_ENV_VAR = "QUANTEM_DINOV3_PATH"

#: The package the hint is expected to make importable.
DINOV3_MODULE = "dinov3"


def dinov3_hint() -> str:
    """The configured hint, stripped. Empty string when unset."""
    return os.environ.get(DINOV3_PATH_ENV_VAR, "").strip()


def hint_provides_dinov3(hint: str | None = None) -> bool:
    """True when putting ``hint`` on ``sys.path`` would make ``import dinov3`` work.

    Never imports anything: :class:`importlib.machinery.PathFinder` resolves the
    name against that one directory and reports whether a spec exists. A hint
    that is empty, is not a directory, or holds no importable ``dinov3`` is
    False.
    """
    hint = dinov3_hint() if hint is None else hint.strip()
    if not hint or not Path(hint).is_dir():
        return False
    try:
        spec = importlib.machinery.PathFinder.find_spec(DINOV3_MODULE, [hint])
    except (ImportError, ValueError, OSError):
        # A malformed entry on the hint, or a directory that vanished between
        # the check above and the search. Either way: no dinov3 here.
        return False
    return spec is not None
