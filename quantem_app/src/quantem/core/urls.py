"""
URL configuration for the QuantEM local server.

The surface here is exactly what a segmentation + proofreading + analysis
desktop app needs: import an image, run one of the released organelle models
over it (whole image or an ROI), proofread the resulting instances, and read the
overlay back as OME-Zarr. Nothing broader is exposed.
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from quantem.assets.views import (
    AssetDetailView,
    AssetListView,
    AssetNgffThumbnailView,
    AssetProcessedPngView,
    AssetUploadView,
    SystemHandshakeView,
    SystemStatusView,
    asset_ngff_file,
    asset_ngff_root,
)
from quantem.core.spa import serve_frontend
from quantem.segmentation.feedback.views import SegmentationUserFeedbackView
from quantem.segmentation.views import (
    AssetSegmentationListCreateView,
    CompletedRoiListCreateView,
    CompletedRoiSubtractView,
    InferredSegmentsView,
    OrganelleApplyFullImageView,
    OrganelleRerunRoiView,
    ProbabilityMapListCreateView,
    SegmentationClearManualLabelsView,
    SegmentationCompleteView,
    SegmentationConfigView,
    SegmentationConfirmBatchView,
    SegmentationDetailView,
    SegmentationOverlayLutView,
    SegmentationOverlayManifestView,
    SegmentationOverlayRebuildView,
    SegmentationRemoveAreaView,
    SegmentationRoiActivateView,
    SegmentationRoiCompleteView,
    SegmentationRoiDetailView,
    SegmentationRoiListCreateView,
    SegmentationRoiSegmentationCompleteView,
    SegmentationRoiSegmentsView,
    SegmentationTypeViewSet,
    SegmentationUncertainSegmentsView,
    SegmentBatchDeleteView,
    SegmentBatchLabelUpdateView,
    SegmentCreateView,
    SegmentLabelUpdateView,
    SegmentsAtPointView,
    SegmentsQueryRegionView,
    segmentation_overlay_ngff_file,
    segmentation_overlay_ngff_root,
)

# Create router and register viewsets
router = DefaultRouter()
router.register(
    r"segmentation-types", SegmentationTypeViewSet, basename="segmentation-type"
)

urlpatterns = [
    # Specific API endpoints that need to be matched before router
    path("api/assets/upload/", AssetUploadView.as_view(), name="asset-upload"),
    path("api/assets/", AssetListView.as_view(), name="asset-list"),
    path(
        "api/assets/<uuid:asset_id>/",
        AssetDetailView.as_view(),
        name="asset-detail",
    ),
    path(
        "api/assets/<uuid:asset_id>/preview-png",
        AssetProcessedPngView.as_view(),
        name="asset-preview-png",
    ),
    path(
        "api/assets/<uuid:asset_id>/ngff-thumbnail/",
        AssetNgffThumbnailView.as_view(),
        name="asset-ngff-thumbnail",
    ),
    path(
        "api/assets/<uuid:asset_id>/segmentations/",
        AssetSegmentationListCreateView.as_view(),
        name="asset-segmentations",
    ),
    path("api/system/status/", SystemStatusView.as_view(), name="system-status"),
    path(
        "api/system/handshake/", SystemHandshakeView.as_view(), name="system-handshake"
    ),
    path("api/", include("quantem.jobs.urls")),
    # Router includes (more general routes)
    path("api/", include(router.urls)),
    # One segmentation: GET (with the deletion preview) and DELETE. The
    # sub-resource routes below all carry a literal suffix, so this bare path
    # collides with none of them.
    path(
        "api/segmentations/<uuid:seg_id>/",
        SegmentationDetailView.as_view(),
        name="segmentation-detail",
    ),
    # Probability map endpoints
    path(
        "api/segmentations/<uuid:segmentation_id>/probability-maps/",
        ProbabilityMapListCreateView.as_view(),
        name="probability-maps",
    ),
    # Segment endpoints
    path(
        "api/segments/<uuid:segment_id>/label/",
        SegmentLabelUpdateView.as_view(),
        name="segment-label-update",
    ),
    path(
        "api/segments/labels/batch/",
        SegmentBatchLabelUpdateView.as_view(),
        name="segment-label-batch-update",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/segments/",
        SegmentCreateView.as_view(),
        name="segment-create",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/segments/confirm-batch/",
        SegmentationConfirmBatchView.as_view(),
        name="segment-confirm-batch",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/segments/remove-area/",
        SegmentationRemoveAreaView.as_view(),
        name="segment-remove-area",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/segments/delete-batch/",
        SegmentBatchDeleteView.as_view(),
        name="segment-delete-batch",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/labels/clear",
        SegmentationClearManualLabelsView.as_view(),
        name="segmentation-labels-clear",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/segments/at-point",
        SegmentsAtPointView.as_view(),
        name="segments-at-point",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/segments/query-region",
        SegmentsQueryRegionView.as_view(),
        name="segments-query-region",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/overlay-manifest/",
        SegmentationOverlayManifestView.as_view(),
        name="segmentation-overlay-manifest",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/overlay-rebuild/",
        SegmentationOverlayRebuildView.as_view(),
        name="segmentation-overlay-rebuild",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/overlay-lut/",
        SegmentationOverlayLutView.as_view(),
        name="segmentation-overlay-lut",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/complete",
        SegmentationCompleteView.as_view(),
        name="segmentation-complete",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/segments/uncertain",
        SegmentationUncertainSegmentsView.as_view(),
        name="segmentation-uncertain-segments",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/roi/",
        SegmentationRoiListCreateView.as_view(),
        name="segmentation-roi",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/completed-rois/",
        CompletedRoiListCreateView.as_view(),
        name="segmentation-completed-rois",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/completed-rois/subtract/",
        CompletedRoiSubtractView.as_view(),
        name="segmentation-completed-rois-subtract",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/roi/segments",
        SegmentationRoiSegmentsView.as_view(),
        name="segmentation-roi-segments",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/roi/activate/",
        SegmentationRoiActivateView.as_view(),
        name="segmentation-roi-activate",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/roi/complete",
        SegmentationRoiCompleteView.as_view(),
        name="segmentation-roi-complete",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/roi/<uuid:roi_id>/complete",
        SegmentationRoiSegmentationCompleteView.as_view(),
        name="segmentation-roi-segmentation-complete",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/roi/<uuid:roi_id>/",
        SegmentationRoiDetailView.as_view(),
        name="segmentation-roi-detail",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/inferred",
        InferredSegmentsView.as_view(),
        name="inferred-segments",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/user-feedback/",
        SegmentationUserFeedbackView.as_view(),
        name="segmentation-user-feedback",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/config/",
        SegmentationConfigView.as_view(),
        name="segmentation-config",
    ),
    path(
        "api/segmentations/<uuid:seg_id>/rerun-roi/",
        OrganelleRerunRoiView.as_view(),
        name="organelle-rerun-roi",
    ),
    # Full-image segmentation endpoint
    path(
        "api/segmentations/<uuid:seg_id>/apply-full-image/",
        OrganelleApplyFullImageView.as_view(),
        name="organelle-apply-full-image",
    ),
    # NGFF (OME-Zarr) serving endpoints
    path(
        "ngff/assets/<uuid:asset_id>.zarr",
        asset_ngff_root,
        name="asset-ngff-root",
    ),
    path(
        "ngff/assets/<uuid:asset_id>.zarr/<path:ngff_path>",
        asset_ngff_file,
        name="asset-ngff-file",
    ),
    path(
        "segmentation-overlays/<uuid:seg_id>.zarr",
        segmentation_overlay_ngff_root,
        name="segmentation-overlay-ngff-root",
    ),
    path(
        "segmentation-overlays/<uuid:seg_id>.zarr/<path:ngff_path>",
        segmentation_overlay_ngff_file,
        name="segmentation-overlay-ngff-file",
    ),
    # Feature apps own their own routes; included before the SPA catch-all.
    path("api/", include("quantem.registry.urls")),
    path("api/", include("quantem.analysis.urls")),
    path("api/", include("quantem.finetune.urls")),
    # The built frontend, served last so every API route above wins. This is what
    # makes `pip install quantem && quantem` a complete application rather than
    # just an API server.
    path("", serve_frontend, name="frontend-index"),
    path("<path:path>", serve_frontend, name="frontend-asset"),
]
