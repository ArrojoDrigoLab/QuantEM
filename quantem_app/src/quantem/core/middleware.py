"""Local-only request guard.

QuantEM has **no authentication**. It is a single-user desktop application whose
server exists only to back its own window: it binds loopback, holds one person's
own images, and has no accounts, no sessions and no users. A login would be
ceremony protecting nothing.

What remains is not auth but a scope check: the server must only ever answer the
machine it runs on. Binding to ``127.0.0.1`` already does that at the socket, and
``ALLOWED_HOSTS`` rejects DNS-rebinding attempts at the Host header. This
middleware closes the third door — a web page you happen to have open making
cross-origin requests to your loopback port — by refusing requests that carry a
foreign ``Origin``.

Same-origin requests from the app itself carry either no ``Origin`` or the
server's own, so the app is unaffected; so are command-line and in-process
clients, which never set the header at all.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.http import HttpResponseForbidden

logger = logging.getLogger(__name__)


def _allowed_origins() -> set[str]:
    origins = set(getattr(settings, "CORS_ALLOWED_ORIGINS", []) or [])
    extra = getattr(settings, "QUANTEM_UI_ORIGINS", []) or []
    return {o.rstrip("/") for o in (*origins, *extra)}


def _is_loopback_origin(origin: str) -> bool:
    """True for ``http(s)://127.0.0.1[:port]``, ``localhost``, ``[::1]``.

    The port is deliberately not checked: the server picks a free one at every
    launch, and the desktop shell serves the UI from a different one again.
    Anything on this machine's loopback interface is the user themselves.
    """
    from urllib.parse import urlparse

    try:
        host = (urlparse(origin).hostname or "").lower()
    except ValueError:
        return False
    return host in {"127.0.0.1", "localhost", "::1"}


class LocalOnlyMiddleware:
    """Reject requests whose ``Origin`` is not this machine."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        origin = request.headers.get("Origin")
        # No Origin: a same-origin navigation, the CLI, or an in-process test
        # client. None of those can be a hostile third-party page.
        if not origin:
            return self.get_response(request)

        if _is_loopback_origin(origin) or origin.rstrip("/") in _allowed_origins():
            return self.get_response(request)

        logger.warning("Rejected a request from a non-local origin: %s", origin)
        return HttpResponseForbidden(
            "QuantEM only answers requests from this machine."
        )
