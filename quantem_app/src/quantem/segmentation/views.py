"""Compatibility export module for segmentation API views.

These are the endpoints the QuantEM proofreading screen uses. Corpus catalog,
SAM prompting, comparator ("other model") runs, membrane classifier,
cell-proposal and manual-refinement surfaces are not part of QuantEM and are not
exported here.
"""

from quantem.segmentation.api_views.actions import (
    OrganelleApplyFullImageView,
    OrganelleRerunRoiView,
    SegmentationConfigView,
)
from quantem.segmentation.api_views.overlay import (
    SegmentationOverlayLutView,
    SegmentationOverlayManifestView,
    SegmentationOverlayRebuildView,
    segmentation_overlay_ngff_file,
    segmentation_overlay_ngff_root,
)
from quantem.segmentation.api_views.roi import (
    CompletedRoiListCreateView,
    CompletedRoiSubtractView,
    SegmentationRoiActivateView,
    SegmentationRoiCompleteView,
    SegmentationRoiDetailView,
    SegmentationRoiListCreateView,
    SegmentationRoiSegmentationCompleteView,
    SegmentationRoiSegmentsView,
)
from quantem.segmentation.api_views.segmentation import (
    AssetSegmentationListCreateView,
    ProbabilityMapListCreateView,
    SegmentationCompleteView,
    SegmentationDetailView,
)
from quantem.segmentation.api_views.segmentation_types import (
    SegmentationTypeViewSet,
)
from quantem.segmentation.api_views.segments import (
    InferredSegmentsView,
    SegmentationClearManualLabelsView,
    SegmentationConfirmBatchView,
    SegmentationRemoveAreaView,
    SegmentationUncertainSegmentsView,
    SegmentBatchDeleteView,
    SegmentBatchLabelUpdateView,
    SegmentCreateView,
    SegmentLabelUpdateView,
    SegmentsAtPointView,
    SegmentsQueryRegionView,
)

__all__ = [
    "SegmentationTypeViewSet",
    "AssetSegmentationListCreateView",
    "CompletedRoiListCreateView",
    "CompletedRoiSubtractView",
    "InferredSegmentsView",
    "OrganelleApplyFullImageView",
    "OrganelleRerunRoiView",
    "ProbabilityMapListCreateView",
    "SegmentBatchDeleteView",
    "SegmentBatchLabelUpdateView",
    "SegmentCreateView",
    "SegmentLabelUpdateView",
    "SegmentationClearManualLabelsView",
    "SegmentationCompleteView",
    "SegmentationConfigView",
    "SegmentationConfirmBatchView",
    "SegmentationDetailView",
    "SegmentationOverlayLutView",
    "SegmentationOverlayManifestView",
    "SegmentationOverlayRebuildView",
    "SegmentationRemoveAreaView",
    "SegmentationRoiActivateView",
    "SegmentationRoiCompleteView",
    "SegmentationRoiDetailView",
    "SegmentationRoiListCreateView",
    "SegmentationRoiSegmentationCompleteView",
    "SegmentationRoiSegmentsView",
    "SegmentationUncertainSegmentsView",
    "SegmentsAtPointView",
    "SegmentsQueryRegionView",
    "segmentation_overlay_ngff_file",
    "segmentation_overlay_ngff_root",
]
