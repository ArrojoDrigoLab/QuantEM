"""URLconf for the analysis API tests.

``core/urls.py`` does not yet include ``quantem.analysis.urls`` -- the project
lead wires that. Overriding ``ROOT_URLCONF`` with this module lets the endpoint
tests exercise the real routes, at the real paths from ``API_CONTRACT.md``,
without touching a file this package does not own.
"""

from django.urls import include, path

urlpatterns = [
    path("api/", include("quantem.analysis.urls")),
]
