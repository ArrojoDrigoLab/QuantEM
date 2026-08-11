"""Guided fine-tuning routes.

Mounted the same way as the jobs app -- ``path("api/", include(...))`` from
``quantem.core.urls`` -- so the paths here are relative to ``api/``.

Two families, deliberately kept apart. The ``segmentations/.../adapt/`` and
``adapters/<id>/`` routes are the labeling view's Improve panel and are
unchanged. The ``finetune/`` routes are the named, experiment-scoped flow; they
are additive, so nothing that already works stops working while both are on
screen.
"""

from django.urls import path

from quantem.finetune.run_views import (
    FineTuneAdaptersView,
    FineTunePreviewView,
    FineTuneRunApplyView,
    FineTuneRunDetailView,
    FineTuneRunProgressView,
    FineTuneRunsView,
    FineTuneScopeView,
)
from quantem.finetune.views import (
    AdaptCropsView,
    AdapterApplyView,
    AdapterDetailView,
    AdaptLatestView,
    AdaptStartView,
)

urlpatterns = [
    # --- the named fine-tune flow ---
    path("finetune/scope/", FineTuneScopeView.as_view(), name="finetune-scope"),
    path("finetune/preview/", FineTunePreviewView.as_view(), name="finetune-preview"),
    path("finetune/runs/", FineTuneRunsView.as_view(), name="finetune-runs"),
    # Declared before the detail route for readability only; the paths differ,
    # so they cannot collide.
    path(
        "finetune/runs/<uuid:adapter_id>/progress/",
        FineTuneRunProgressView.as_view(),
        name="finetune-run-progress",
    ),
    path(
        "finetune/runs/<uuid:adapter_id>/apply/",
        FineTuneRunApplyView.as_view(),
        name="finetune-run-apply",
    ),
    path(
        "finetune/runs/<uuid:adapter_id>/",
        FineTuneRunDetailView.as_view(),
        name="finetune-run-detail",
    ),
    path(
        "finetune/adapters/",
        FineTuneAdaptersView.as_view(),
        name="finetune-adapters",
    ),
    # --- the Improve panel, unchanged ---
    path(
        "segmentations/<uuid:seg_id>/adapt/crops/",
        AdaptCropsView.as_view(),
        name="adapt-crops",
    ),
    # Declared before the bare ``adapt/`` POST route for readability only; the
    # two cannot collide, the paths differ.
    path(
        "segmentations/<uuid:seg_id>/adapt/latest/",
        AdaptLatestView.as_view(),
        name="adapt-latest",
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
