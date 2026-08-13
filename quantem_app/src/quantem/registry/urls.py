"""Model registry routes.

Mounted the same way as the jobs and finetune apps -- ``path("api/",
include("quantem.registry.urls"))`` from ``quantem.core.urls`` -- so the paths
here are relative to ``api/`` and carry no prefix of their own.

``<str:pack_id>`` rather than a slug converter: pack ids contain a colon
(``quantem:mito``), which is a legal path character, and the ids are validated
against :data:`quantem.inference.specs.MODEL_SPECS` in the view where an unknown
one can be named in the error.
"""

from django.urls import path

from quantem.registry.views import ModelDetailView, ModelInstallView, ModelListView

urlpatterns = [
    path("models/", ModelListView.as_view(), name="model-list"),
    path("models/<str:pack_id>/", ModelDetailView.as_view(), name="model-detail"),
    path(
        "models/<str:pack_id>/install/",
        ModelInstallView.as_view(),
        name="model-install",
    ),
]
