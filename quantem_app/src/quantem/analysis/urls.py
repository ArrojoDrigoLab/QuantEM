"""Analysis routes.

Mounted by the project URLconf with ``path("api/", include("quantem.analysis.urls"))``.
The prefix lives there, not here, so the whole app can be moved under a
different mount point without editing this file.
"""

from django.urls import path

from quantem.segmentation.api_views.analysis import (
    AnalysisRunDetailView,
    AnalysisRunExportView,
    GlobalAreaAnalysisView,
    SegmentationAnalysisView,
)

from .views import AnalysisGroupRollupView

urlpatterns = [
    path(
        "segmentations/<uuid:seg_id>/analysis/global-area/",
        GlobalAreaAnalysisView.as_view(),
        name="segmentation-global-area-analysis",
    ),
    path(
        "segmentations/<uuid:seg_id>/analysis/",
        SegmentationAnalysisView.as_view(),
        name="segmentation-analysis",
    ),
    # Before the ``<uuid:run_id>`` route: "groups" is not a UUID so the two
    # cannot collide today, but the ordering makes that independent of how the
    # converter happens to be spelled.
    path(
        "analysis/groups/",
        AnalysisGroupRollupView.as_view(),
        name="analysis-group-rollup",
    ),
    path(
        "analysis/<uuid:run_id>/",
        AnalysisRunDetailView.as_view(),
        name="analysis-run-detail",
    ),
    # <str:...> cannot contain "/", and the view still resolves the name inside
    # the run's own directory before opening anything.
    path(
        "analysis/<uuid:run_id>/export/<str:name>",
        AnalysisRunExportView.as_view(),
        name="analysis-run-export",
    ),
]
