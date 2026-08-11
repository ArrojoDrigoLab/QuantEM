"""Test urlconf: exactly what ``core/urls.py`` needs to add.

Mounting only this package's routes keeps the API tests from depending on that
shared file having been wired up yet.
"""

from django.urls import include, path

urlpatterns = [
    path("api/", include("quantem.sam.urls")),
]
