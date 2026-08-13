"""Experiment, dataset and image-grouping routes.

Relative to ``api/``, like every other app's URL module, so the mount point is
decided in one place and not repeated here.

``assets/grouping/`` is in this file rather than in ``assets/urls.py`` because
grouping is this app's rule, not the asset app's: the resource being changed is
an image, but what may and may not be written is
:func:`quantem.library.models.validate_asset_grouping`. Keeping the route next
to the rule is what stops a second, unvalidated assignment path appearing later.
"""

from django.urls import path

from quantem.library.views import (
    AssetGroupingView,
    AssetLibraryEditView,
    DatasetDetailView,
    DatasetListCreateView,
    ExperimentDetailView,
    ExperimentListCreateView,
)

urlpatterns = [
    path(
        "assets/<uuid:asset_id>/library-edit/",
        AssetLibraryEditView.as_view(),
        name="asset-library-edit",
    ),
    path(
        "experiments/",
        ExperimentListCreateView.as_view(),
        name="experiment-list",
    ),
    path(
        "experiments/<uuid:experiment_id>/",
        ExperimentDetailView.as_view(),
        name="experiment-detail",
    ),
    path("datasets/", DatasetListCreateView.as_view(), name="dataset-list"),
    path(
        "datasets/<uuid:dataset_id>/",
        DatasetDetailView.as_view(),
        name="dataset-detail",
    ),
    # Not ``assets/<id>/grouping/``: the library assigns a *selection*, and one
    # image is a selection of one. A per-image route would be the same code
    # with a different arity and a second place for the rule to be forgotten.
    path("assets/grouping/", AssetGroupingView.as_view(), name="asset-grouping"),
]
