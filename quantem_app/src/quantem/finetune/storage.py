"""Where an adapter's weights live.

A separate module from :mod:`quantem.finetune.models` on purpose: the job must
be able to save a head on an install where ``quantem.finetune`` is not (yet) in
``INSTALLED_APPS``, and importing a Django model there raises.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path

from quantem.core.config import MODELS_DIR

#: Adapters live beside the downloaded packs, under the user data directory.
#: Never inside the installation -- a pip install may be read-only.
ADAPTERS_DIR = MODELS_DIR / "adapters"

#: What a staged head is called while it is being written. Promoted over
#: ``head.pt`` only once the run has succeeded; see :func:`promote_head`.
STAGED_HEAD_NAME = "head.pt.incoming"


def adapter_dir(adapter_id: str) -> Path:
    return ADAPTERS_DIR / str(adapter_id)


def adapter_head_path(adapter_id: str) -> Path:
    """The trained neck + decoder for one adapter.

    Small by design: the frozen encoder is not copied here. It is already in the
    registry cache addressed by digest, and a 525 MB copy per adapter would be
    the largest thing this app ever wrote for no information gained.
    """
    return adapter_dir(adapter_id) / "head.pt"


def unsaved_head_path() -> Path:
    """A scratch head file for a run that has no adapter row behind it.

    Unique per call, deliberately. This used to be a fixed ``adapters/unsaved/
    head.pt``: two unrecorded runs at once wrote the same file, and the second
    one's weights were verified against the first one's. A run with no row to
    point at it has nothing to collide with, so the id costs nothing.
    """
    return adapter_dir(f"unsaved-{uuid.uuid4().hex}") / "head.pt"


def staged_head_path(adapter_id: str) -> Path:
    """Where a head is written before it is allowed to replace the live one.

    Overwriting a fine-tune must not damage it. The new weights are written
    here, verified, and only then moved over ``head.pt`` -- so a run that fails
    at any point, including inside ``torch.save``, leaves the previous version
    exactly as it was and still loadable.
    """
    return adapter_dir(adapter_id) / STAGED_HEAD_NAME


def promote_head(staged: Path, final: Path) -> Path:
    """Move a verified staged head over the live one, atomically where possible.

    ``os.replace`` is atomic on both POSIX and Windows for paths on one volume,
    which these always are: they are siblings in the same adapter folder.
    """
    staged = Path(staged)
    final = Path(final)
    if staged == final:
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(staged), str(final))
    return final


def discard_staged_head(staged: Path) -> None:
    """Remove a staged head that will never be promoted. Never raises.

    Called on the failure path, where the exception being handled is the one
    worth reporting: a scratch file that could not be deleted must not replace
    it with a less useful one.
    """
    with contextlib.suppress(OSError):
        Path(staged).unlink(missing_ok=True)


def relative_head_path(path: Path) -> str:
    """``MODELS_DIR``-relative form, for storing on the adapter row.

    Relative because the user data directory moves: a backup restored under a
    different home would otherwise carry absolute paths that no longer exist.
    Forward slashes, matching ``quantem.core.local_storage``.
    """
    try:
        return str(Path(path).relative_to(MODELS_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)
