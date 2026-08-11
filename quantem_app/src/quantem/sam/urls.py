"""Box-prompted object adding routes.

Mounted the same way as the jobs, registry and finetune apps --
``path("api/", include("quantem.sam.urls"))`` from ``quantem.core.urls`` -- so
the paths here are relative to ``api/``.
"""

from django.urls import path

from quantem.sam.views import SamBoxPromptView, SamModelDownloadView, SamModelView

urlpatterns = [
    path("sam/model/", SamModelView.as_view(), name="sam-model"),
    path(
        "sam/model/download/",
        SamModelDownloadView.as_view(),
        name="sam-model-download",
    ),
    path(
        "sam/segmentations/<uuid:seg_id>/box/",
        SamBoxPromptView.as_view(),
        name="sam-box-prompt",
    ),
]
