"""Asset and system routes.

Mounted by the project URLconf with
``path("api/", include("quantem.assets.urls"))`` -- so the paths here are
relative to ``api/``, and the app can be moved under a different mount point
without editing this file. Moved out of ``core/urls.py`` unchanged, in the
order they were resolved in.

The NGFF *serving* routes are not here: they sit under ``ngff/``, not ``api/``,
and so need their own mount. See :mod:`quantem.assets.ngff_urls`.
"""

from django.urls import path

from quantem.assets.views import (
    AssetDetailView,
    AssetListView,
    AssetNgffThumbnailView,
    AssetProcessedPngView,
    AssetUploadView,
    SystemHandshakeView,
    SystemStatusView,
)
from quantem.library.urls import urlpatterns as library_urls
from quantem.segmentation.views import AssetSegmentationListCreateView

urlpatterns = [
    # Specific API endpoints that need to be matched before router
    path("assets/upload/", AssetUploadView.as_view(), name="asset-upload"),
    path("assets/", AssetListView.as_view(), name="asset-list"),
    path(
        "assets/<uuid:asset_id>/",
        AssetDetailView.as_view(),
        name="asset-detail",
    ),
    path(
        "assets/<uuid:asset_id>/preview-png",
        AssetProcessedPngView.as_view(),
        name="asset-preview-png",
    ),
    path(
        "assets/<uuid:asset_id>/ngff-thumbnail/",
        AssetNgffThumbnailView.as_view(),
        name="asset-ngff-thumbnail",
    ),
    path(
        "assets/<uuid:asset_id>/segmentations/",
        AssetSegmentationListCreateView.as_view(),
        name="asset-segmentations",
    ),
    path("system/status/", SystemStatusView.as_view(), name="system-status"),
    path("system/handshake/", SystemHandshakeView.as_view(), name="system-handshake"),
    # The grouping layer over this library: experiments, datasets, and the one
    # route that puts images into them. Spliced in here, from
    # :mod:`quantem.library.urls`, rather than mounted from ``core/urls.py``:
    # both modules resolve under ``api/``, the two apps are one feature seen
    # from two sides, and the project URLconf is a file several packages edit
    # at once. Exactly the idiom ``segmentation/urls.py`` already uses for the
    # same reason. Owning app keeps its own file; the mount is one line.
    *library_urls,
]
