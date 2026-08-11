"""SAM runtimes, one module each, one built at a time.

The abstraction exists so the runtime can be swapped -- stock Meta ``vit_b``,
SAM-HQ2, the full ``micro_sam`` package -- without the view, the cache or the
geometry knowing. It is deliberately *not* a plugin system: there is no
discovery, no entry points and no registration. :func:`get_backend` is an
``if``.

THE LOADED MODEL IS CACHED FOR THE LIFE OF THE PROCESS
------------------------------------------------------
This is the first of the two caches this feature needs, and the more important
one. The weights are 375 MB and take seconds to load onto a device; a user in
the labeling view draws boxes in quick succession, so a cold load per request
would make the feature unusable. The model is built once and every later
request reuses it.

The second cache -- :mod:`quantem.sam.embedding_cache` -- is a different thing
and does not replace this one. It saves the *encoder pass* over a crop the user
has already visited. A box in a fresh region misses that cache and pays a new
encode, but it must still not pay a model load.

**The cache key identifies the weights, not the model's name.** Keying on an id
alone is a real defect in the implementation this was ported from: two entries
differing only in checkpoint path collide, and the second silently gets the
first's weights. Here the key is the backend identity together with the
checkpoint file's path, size and modification time, so a re-downloaded or
swapped file is a different key and cannot be served a stale model.
"""

from __future__ import annotations

import os
import threading

from quantem.sam.backends.base import Embedding, MaskCandidate, SamBackend
from quantem.sam.config import STUB_MODE_ENV_VAR

__all__ = [
    "Embedding",
    "MaskCandidate",
    "SamBackend",
    "get_backend",
    "loaded_backend_keys",
    "reset_backend",
    "stub_mode",
]

#: Weights fingerprint -> built backend. At most one entry in practice; a dict
#: rather than a slot so a changed checkpoint cannot be served a stale model.
_MODELS: dict[tuple, SamBackend] = {}
_MODEL_LOCK = threading.Lock()


def stub_mode() -> bool:
    """True when the deterministic no-weights backend is in force."""
    return os.environ.get(STUB_MODE_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def _weights_key() -> tuple:
    """Something that actually identifies the weights on disk right now."""
    if stub_mode():
        return ("stub",)

    from quantem.sam.checkpoint import checkpoint_path
    from quantem.sam.config import CHECKPOINT

    path = checkpoint_path()
    try:
        stat = path.stat()
        fingerprint = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        fingerprint = (str(path), -1, -1)
    return (CHECKPOINT.identity, *fingerprint)


def get_backend() -> SamBackend:
    """The process-wide backend for the current weights, built on first use.

    Double-checked: the lock is taken to look, released for the multi-second
    load, and retaken to publish. Holding it across the load would turn every
    concurrent first prompt into a stall; the cost of not holding it is that two
    simultaneous cold starts may both build and one copy is dropped. That is the
    same trade ``inference.engine.load_model`` already makes, and it is the
    right one.
    """
    key = _weights_key()
    with _MODEL_LOCK:
        cached = _MODELS.get(key)
        if cached is not None:
            return cached

    if stub_mode():
        from quantem.sam.backends.stub import StubBackend

        built: SamBackend = StubBackend()
    else:
        from quantem.sam.backends.sam1 import Sam1Backend

        built = Sam1Backend()

    with _MODEL_LOCK:
        # Another thread may have won the race; prefer its copy so every caller
        # shares one model rather than two.
        existing = _MODELS.get(key)
        if existing is not None:
            return existing
        # One set of weights at a time. Replacing rather than accumulating keeps
        # this from becoming the unbounded cache the embedding one refuses to be.
        _MODELS.clear()
        _MODELS[key] = built
        return built


def loaded_backend_keys() -> list[tuple]:
    """Which weights are loaded. For tests and for the status endpoint."""
    with _MODEL_LOCK:
        return list(_MODELS)


def reset_backend() -> None:
    """Drop the loaded model. For tests, and after the weights are downloaded."""
    with _MODEL_LOCK:
        _MODELS.clear()
