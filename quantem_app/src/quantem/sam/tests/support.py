"""Shared test scaffolding.

The suite runs **offline, on CPU, with no weights**: everything here routes
through the stub backend. A test that needed 375 MB of downloaded weights and a
GPU would not run in CI, and the plumbing this feature is mostly made of --
crop planning, the caches, the coordinate round trip, object creation -- is
exactly what would then go untested.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from unittest import mock

from quantem.sam.backends import reset_backend
from quantem.sam.config import STUB_MODE_ENV_VAR
from quantem.sam.embedding_cache import EMBEDDINGS

#: A urlconf mounting only this package's routes.
TEST_URLCONF = "quantem.sam.tests.urls"


@contextlib.contextmanager
def stub_environment() -> Iterator[None]:
    """Run the deterministic backend, with both caches empty on entry and exit."""
    with mock.patch.dict("os.environ", {STUB_MODE_ENV_VAR: "1"}):
        reset_backend()
        EMBEDDINGS.clear()
        try:
            yield
        finally:
            reset_backend()
            EMBEDDINGS.clear()
