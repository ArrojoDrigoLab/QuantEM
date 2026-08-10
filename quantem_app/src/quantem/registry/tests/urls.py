"""Test urlconf: exactly what ``core/urls.py`` needs to add.

Mirrors the finetune and analysis apps, so the API tests here do not wait on
``core/urls.py`` -- which is owned elsewhere -- being wired up.
"""

from django.urls import include, path

urlpatterns = [
    path("api/", include("quantem.registry.urls")),
]
