"""Segmentation, segment and proofreading routes.

Mounted by the project URLconf with
``path("api/", include("quantem.segmentation.urls"))`` -- so the paths here are
relative to ``api/``. Moved out of ``core/urls.py`` unchanged and in the order
they were resolved in, including the DRF router, which sat immediately before
them there.

**Adding a route: do not add it here.** The five modules spliced in at the
bottom exist so that packages adding routes at the same time each own one file
and never collide in this one. Each is empty until its owner fills it, and an
empty ``urlpatterns`` adds nothing to the resolver. Add to the one that matches
your feature, or add a sixth module beside them.

The overlay NGFF *serving* routes are not here: they sit under
``segmentation-overlays/``, not ``api/``, and so need their own mount. See
:mod:`quantem.segmentation.overlay_urls`.
"""

from django.urls import path
from rest_framework.routers import DefaultRouter

from quantem.segmentation.api_views.analysis_masks import (
    urlpatterns as analysis_mask_urls,
)
from quantem.segmentation.api_views.preview import urlpatterns as preview_urls
from quantem.segmentation.api_views.propose import urlpatterns as propose_urls
from quantem.segmentation.api_views.quality import urlpatterns as quality_urls
from quantem.segmentation.api_views.rethreshold import urlpatterns as rethreshold_urls
from quantem.segmentation.api_views.runs import urlpatterns as runs_urls
from quantem.segmentation.feedback.views import SegmentationUserFeedbackView
from quantem.segmentation.views import (
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
)

# Create router and register viewsets
router = DefaultRouter()
router.register(r"segmentation-types", SegmentationTypeViewSet, basename="segmentation-type")

urlpatterns = [
    # Router includes (more general routes)
    *router.urls,
    # One segmentation: GET (with the deletion preview) and DELETE. The
    # sub-resource routes below all carry a literal suffix, so this bare path
    # collides with none of them.
    path(
        "segmentations/<uuid:seg_id>/",
        SegmentationDetailView.as_view(),
        name="segmentation-detail",
    ),
    # Probability map endpoints
    path(
        "segmentations/<uuid:segmentation_id>/probability-maps/",
        ProbabilityMapListCreateView.as_view(),
        name="probability-maps",
    ),
    # Segment endpoints
    path(
        "segments/<uuid:segment_id>/label/",
        SegmentLabelUpdateView.as_view(),
        name="segment-label-update",
    ),
    path(
        "segments/labels/batch/",
        SegmentBatchLabelUpdateView.as_view(),
        name="segment-label-batch-update",
    ),
    path(
        "segmentations/<uuid:seg_id>/segments/",
        SegmentCreateView.as_view(),
        name="segment-create",
    ),
    path(
        "segmentations/<uuid:seg_id>/segments/confirm-batch/",
        SegmentationConfirmBatchView.as_view(),
        name="segment-confirm-batch",
    ),
    path(
        "segmentations/<uuid:seg_id>/segments/remove-area/",
        SegmentationRemoveAreaView.as_view(),
        name="segment-remove-area",
    ),
    path(
        "segmentations/<uuid:seg_id>/segments/delete-batch/",
        SegmentBatchDeleteView.as_view(),
        name="segment-delete-batch",
    ),
    path(
        "segmentations/<uuid:seg_id>/labels/clear",
        SegmentationClearManualLabelsView.as_view(),
        name="segmentation-labels-clear",
    ),
    path(
        "segmentations/<uuid:seg_id>/segments/at-point",
        SegmentsAtPointView.as_view(),
        name="segments-at-point",
    ),
    path(
        "segmentations/<uuid:seg_id>/segments/query-region",
        SegmentsQueryRegionView.as_view(),
        name="segments-query-region",
    ),
    path(
        "segmentations/<uuid:seg_id>/overlay-manifest/",
        SegmentationOverlayManifestView.as_view(),
        name="segmentation-overlay-manifest",
    ),
    path(
        "segmentations/<uuid:seg_id>/overlay-rebuild/",
        SegmentationOverlayRebuildView.as_view(),
        name="segmentation-overlay-rebuild",
    ),
    path(
        "segmentations/<uuid:seg_id>/overlay-lut/",
        SegmentationOverlayLutView.as_view(),
        name="segmentation-overlay-lut",
    ),
    path(
        "segmentations/<uuid:seg_id>/complete",
        SegmentationCompleteView.as_view(),
        name="segmentation-complete",
    ),
    path(
        "segmentations/<uuid:seg_id>/segments/uncertain",
        SegmentationUncertainSegmentsView.as_view(),
        name="segmentation-uncertain-segments",
    ),
    path(
        "segmentations/<uuid:seg_id>/roi/",
        SegmentationRoiListCreateView.as_view(),
        name="segmentation-roi",
    ),
    path(
        "segmentations/<uuid:seg_id>/completed-rois/",
        CompletedRoiListCreateView.as_view(),
        name="segmentation-completed-rois",
    ),
    path(
        "segmentations/<uuid:seg_id>/completed-rois/subtract/",
        CompletedRoiSubtractView.as_view(),
        name="segmentation-completed-rois-subtract",
    ),
    path(
        "segmentations/<uuid:seg_id>/roi/segments",
        SegmentationRoiSegmentsView.as_view(),
        name="segmentation-roi-segments",
    ),
    path(
        "segmentations/<uuid:seg_id>/roi/activate/",
        SegmentationRoiActivateView.as_view(),
        name="segmentation-roi-activate",
    ),
    path(
        "segmentations/<uuid:seg_id>/roi/complete",
        SegmentationRoiCompleteView.as_view(),
        name="segmentation-roi-complete",
    ),
    path(
        "segmentations/<uuid:seg_id>/roi/<uuid:roi_id>/complete",
        SegmentationRoiSegmentationCompleteView.as_view(),
        name="segmentation-roi-segmentation-complete",
    ),
    path(
        "segmentations/<uuid:seg_id>/roi/<uuid:roi_id>/",
        SegmentationRoiDetailView.as_view(),
        name="segmentation-roi-detail",
    ),
    path(
        "segmentations/<uuid:seg_id>/inferred",
        InferredSegmentsView.as_view(),
        name="inferred-segments",
    ),
    path(
        "segmentations/<uuid:seg_id>/user-feedback/",
        SegmentationUserFeedbackView.as_view(),
        name="segmentation-user-feedback",
    ),
    path(
        "segmentations/<uuid:seg_id>/config/",
        SegmentationConfigView.as_view(),
        name="segmentation-config",
    ),
    path(
        "segmentations/<uuid:seg_id>/rerun-roi/",
        OrganelleRerunRoiView.as_view(),
        name="organelle-rerun-roi",
    ),
    # Full-image segmentation endpoint
    path(
        "segmentations/<uuid:seg_id>/apply-full-image/",
        OrganelleApplyFullImageView.as_view(),
        name="organelle-apply-full-image",
    ),
    # One module per feature that is adding routes; see the module docstring.
    *analysis_mask_urls,
    *quality_urls,
    *rethreshold_urls,
    *propose_urls,
    *preview_urls,
    *runs_urls,
]
