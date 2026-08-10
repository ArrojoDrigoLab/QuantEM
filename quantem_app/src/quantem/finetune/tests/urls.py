"""Test urlconf: exactly what ``core/urls.py`` needs to add."""

from django.urls import include, path

urlpatterns = [
    path("api/", include("quantem.finetune.urls")),
]
