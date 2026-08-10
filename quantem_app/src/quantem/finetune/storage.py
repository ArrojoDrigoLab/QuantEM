"""Where an adapter's weights live.

A separate module from :mod:`quantem.finetune.models` on purpose: the job must
be able to save a head on an install where ``quantem.finetune`` is not (yet) in
``INSTALLED_APPS``, and importing a Django model there raises.
"""

from __future__ import annotations

from pathlib import Path

from quantem.core.config import MODELS_DIR

#: Adapters live beside the downloaded packs, under the user data directory.
#: Never inside the installation -- a pip install may be read-only.
ADAPTERS_DIR = MODELS_DIR / "adapters"


def adapter_dir(adapter_id: str) -> Path:
    return ADAPTERS_DIR / str(adapter_id)


def adapter_head_path(adapter_id: str) -> Path:
    """The trained neck + decoder for one adapter.

    Small by design: the frozen encoder is not copied here. It is already in the
    registry cache addressed by digest, and a 525 MB copy per adapter would be
    the largest thing this app ever wrote for no information gained.
    """
    return adapter_dir(adapter_id) / "head.pt"


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
