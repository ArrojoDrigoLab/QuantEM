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
from django.core.exceptions import (
    RequestDataTooBig,
    SuspiciousOperation,
    TooManyFieldsSent,
    TooManyFilesSent,
)
from django.http import HttpResponseForbidden, JsonResponse, UnreadablePostError
from django.http.multipartparser import MultiPartParserError

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
        return HttpResponseForbidden("QuantEM only answers requests from this machine.")


#: What to say for each way a request can be refused before the view that would
#: have explained it ever runs. Each one is a sentence about the file the person
#: chose, not about the setting that stopped it: ``settings.
#: DATA_UPLOAD_MAX_NUMBER_FILES`` is the true cause and is useless to the reader,
#: which is invariant I-12's whole point.
_UNREADABLE_REQUEST_COPY: tuple[tuple[type[Exception], str], ...] = (
    (
        TooManyFilesSent,
        "That import had too many files in it. Import your images in smaller "
        "batches and try again.",
    ),
    (
        TooManyFieldsSent,
        "That request had more fields in it than QuantEM accepts. Nothing was changed.",
    ),
    (
        RequestDataTooBig,
        "That request was too large for QuantEM to read. If you were importing "
        "an image, import it on its own rather than with others.",
    ),
    (
        MultiPartParserError,
        "That upload did not arrive in one piece, so QuantEM could not read the "
        "file. Nothing was imported; try again.",
    ),
    (
        UnreadablePostError,
        "The upload stopped before it finished, so QuantEM could not read the "
        "file. Nothing was imported; try again.",
    ),
    # Last: everything above is a SuspiciousOperation, so the catch-all has to
    # come after the specific ones.
    (
        SuspiciousOperation,
        "QuantEM could not read that request. Nothing was changed.",
    ),
)


class ApiErrorShapeMiddleware:
    """Answer a refused API request with the app's own error shape.

    Some rejections never reach the view that would have explained them. A
    multipart body is parsed lazily, on the first touch of ``request.FILES``,
    and the exceptions that parse raises -- ``TooManyFilesSent``,
    ``RequestDataTooBig``, ``MultiPartParserError`` -- are
    ``SuspiciousOperation``s, which Django answers with its own built-in page::

        <!doctype html><html lang="en"><head><title>Bad Request (400)</title>
        </head><body><h1>Bad Request (400)</h1><p></p></body></html>

    An empty ``<p>``. Measured on 2026-08-10: 101 files in one request produced
    exactly that, with ``Content-Type: text/html`` -- so the client's JSON parse
    failed too and the import panel had nothing but the status code to show.

    This turns those into the shape every other refusal on ``/api/`` uses.
    ``error`` and ``detail`` both carry the sentence because both names are
    live: the file-upload client reads ``error`` (``shared/api/core/http.ts``)
    and every DRF view answers with ``detail``, and a request that died in the
    parser does not know which endpoint it was going to reach.

    Only ``/api/`` is touched. A browser asking for a page still gets Django's
    HTML, which is the right answer for a browser.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_exception(self, request, exception):
        if not request.path.startswith("/api/"):
            return None
        for kind, sentence in _UNREADABLE_REQUEST_COPY:
            if isinstance(exception, kind):
                logger.warning(
                    "Refused %s before the view ran: %s",
                    request.path,
                    type(exception).__name__,
                )
                return JsonResponse({"error": sentence, "detail": sentence}, status=400)
        return None
