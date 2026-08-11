"""Serve the built frontend from the local server.

The pip/conda channel launches ``quantem-app run``, which starts this server on a
loopback port and opens a window at it — so the server has to serve the UI, not
just the API. The desktop installer wraps the same package and can either do the
same or serve ``dist/`` from the shell; either way this is the fallback that
makes ``pip install quantem-app && quantem-app`` a complete application.

Two behaviours that matter:

* **Hashed assets are immutable.** Vite fingerprints every chunk, so
  ``/assets/index-BCh8Obmf.js`` can be cached forever. ``index.html`` must never
  be, or an updated app keeps booting the previous bundle.
* **Unknown paths fall through to ``index.html``.** The frontend uses
  ``HashRouter``, so this mostly matters for a bare reload, but it also means a
  mistyped URL shows the app rather than a Django 404 page.

API routes are matched first: this is registered last in the URLconf.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt

#: Where the built frontend lives in a source checkout:
#: ``quantem_app/frontend/dist``, three levels up from this file.
_CHECKOUT_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


def _packaged_dist() -> Path | None:
    """The frontend build shipped inside an installed wheel, or ``None``.

    The wheel carries ``frontend/dist`` as package data under
    ``quantem/_frontend`` (see ``[tool.hatch.build.targets.wheel.force-include]``
    in pyproject.toml), so an installed distribution serves its UI with no
    checkout anywhere. Resolved through :mod:`importlib.resources` so it follows
    the package wherever the installer put it; wheels install unzipped, so the
    traversable is a real directory and the ``Path`` round-trip is exact.
    """
    try:
        from importlib.resources import files

        root = Path(str(files("quantem") / "_frontend"))
    except Exception:
        return None
    return root if (root / "index.html").is_file() else None


def dist_root() -> Path:
    """Resolution order: setting, checkout build, packaged build.

    The checkout wins over the packaged copy so that a developer who edits the
    frontend and rebuilds ``dist/`` sees that build, even in a venv that also
    has a released wheel installed. When neither exists the checkout path is
    still returned: ``_index_response`` turns it into the actionable 404.
    """
    configured = getattr(settings, "QUANTEM_FRONTEND_DIST", None)
    if configured:
        return Path(configured)
    if (_CHECKOUT_DIST / "index.html").is_file():
        return _CHECKOUT_DIST
    packaged = _packaged_dist()
    return packaged if packaged is not None else _CHECKOUT_DIST


def frontend_available() -> bool:
    return (dist_root() / "index.html").is_file()


def _runtime_config_script(request: HttpRequest) -> str:
    """The ``window.__APP_CONFIG__`` the frontend reads at boot.

    ``frontend/src/config.ts::getRuntimeConfig`` has always read this object, and
    nothing ever set it — the seam existed but was dead. The desktop
    shell will inject it before the bundle loads; for the pip channel the server
    does it here, so ``quantem-app run`` works with no shell at all.

    There is no auth token: QuantEM is single-user and loopback-only.
    """
    # The absolute origin, not "" -- the viewer builds zarr store locations with
    # `new URL(path, apiBaseUrl)`, and an empty base throws "Invalid URL". The
    # port is chosen per launch, so it has to come from the live request rather
    # than a setting.
    origin = request.build_absolute_uri("/").rstrip("/")
    config = {
        "apiBaseUrl": origin,
        "dev": bool(settings.DEBUG),
    }
    return "<script>window.__APP_CONFIG__=" + json.dumps(config) + ";</script>"


def _index_response(request: HttpRequest) -> HttpResponse:
    index = dist_root() / "index.html"
    if not index.is_file():
        raise Http404(
            "The QuantEM frontend has not been built. Run `npm run build` in "
            "frontend/, or install a released distribution which ships it."
        )
    html = index.read_text(encoding="utf-8")
    # Inject before the first script so the config exists when the bundle boots.
    html = html.replace("</head>", _runtime_config_script(request) + "</head>", 1)
    response = HttpResponse(html, content_type="text/html")
    # Never cache the entry point: it names the hashed bundles.
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


#: Prefixes owned by the backend. An unmatched path under one of these is a
#: genuine 404, not a client route. Without this the catch-all below answers a
#: mistyped endpoint with index.html, and the caller gets HTML where it expected
#: JSON — which surfaces as an unrelated parse error, several layers away.
_API_PREFIXES = (
    "api/",
    "ngff/",
    "segmentation-overlays/",
    "static/",
)


@csrf_exempt
def serve_frontend(request: HttpRequest, path: str = "") -> HttpResponse:
    """Serve a built asset, or ``index.html`` for anything unrecognised.

    ``csrf_exempt`` because this view changes no state and every request method
    must reach the 404 below. Without it, CsrfViewMiddleware rejects any POST
    that falls through to this catch-all *before* the view runs -- so a mistyped
    API endpoint (``.../roi/<id>/complete/`` with a trailing slash, where the
    route has none) answered POSTs with a 403 CSRF HTML page instead of the 404
    every other wrong path gets. Nothing in QuantEM sets a CSRF cookie (the API
    is DRF views, exempt by design; single user, loopback only), so that 403
    was unresolvable from the client side.
    """
    if path.startswith(_API_PREFIXES):
        raise Http404(f"No such endpoint: /{path}")
    root = dist_root()
    if path:
        candidate = (root / path).resolve()
        try:
            # Reject traversal out of dist/, even though Django's path converter
            # already normalises: this is served on a real socket.
            candidate.relative_to(root.resolve())
        except ValueError:
            raise Http404("Not found") from None
        if candidate.is_file():
            content_type, _ = mimetypes.guess_type(candidate.name)
            response = FileResponse(
                candidate.open("rb"),
                content_type=content_type or "application/octet-stream",
            )
            if "/assets/" in f"/{path}":
                # Vite content-hashes these; they can never change under a name.
                response["Cache-Control"] = "public, max-age=31536000, immutable"
            return response
    return _index_response(request)
