"""Guided fine-tuning routes.

Mounted the same way as the jobs app -- ``path("api/", include(...))`` from
``quantem.core.urls`` -- so the paths here are relative to ``api/``.
"""

from django.urls import path

from quantem.finetune.views import (
    AdaptCropsView,
    AdapterApplyView,
    AdapterDetailView,
    AdaptStartView,
)

urlpatterns = [
    path(
        "segmentations/<uuid:seg_id>/adapt/crops/",
        AdaptCropsView.as_view(),
        name="adapt-crops",
    ),
    path(
        "segmentations/<uuid:seg_id>/adapt/",
        AdaptStartView.as_view(),
        name="adapt-start",
    ),
    path(
        "adapters/<uuid:adapter_id>/",
        AdapterDetailView.as_view(),
        name="adapter-detail",
    ),
    path(
        "adapters/<uuid:adapter_id>/apply/",
        AdapterApplyView.as_view(),
        name="adapter-apply",
    ),
]
