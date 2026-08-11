import type { AssetDetail, UserFeedback } from "@/shared/types";
import type { SegmentationLeftPanelProps } from "@/features/segmentation/components/leftPanel/types";

function noop() {}

export function makeLeftPanelImage(): AssetDetail {
  return {
    id: "img-1",
    file_path: "",
    original_filename: "source.tif",
    display_name: "Image 1",
    is_eval_set: false,
    width: 1000,
    height: 1000,
    channels: 1,
    bit_depth: 8,
    preprocess_stage: "DONE",
    preprocess_progress: 100,
    tags: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ngff_ready: true,
    ngff_url: "/ngff/img-1.zarr",
  };
}

export function makeLeftPanelFeedback(
  overrides: Partial<UserFeedback> = {}
): UserFeedback {
  return {
    id: "feedback-1",
    segmentation: "seg-1",
    input_type: "point",
    point: { x: 25, y: 30 },
    polygon_coords: null,
    feedback_type: "CONFIRMED",
    utilized_status: "SUCCESS",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

export function makeLeftPanelProps(
  overrides: Partial<SegmentationLeftPanelProps> = {}
): SegmentationLeftPanelProps {
  return {
    viewer: {
      image: makeLeftPanelImage(),
      segmentationTypeInternalName: "quantem_internal_mito",
      useSmoothedGeometry: false,
      layerStyles: {
        candidateStrokeWidth: 2,
        candidateFillOpacity: 0.18,
        confirmedStrokeWidth: 2,
        confirmedFillOpacity: 0.15,
      },
      viewport: null,
      onViewportChange: noop,
      overlayNgffLayers: [],
      transientFitBounds: null,
      transientFitBoundsKey: null,
    },
    workflow: {
      mode: "review",
      leftMode: "hover",
      reviewPhase: "model",
      correctionTool: "draw",
      navigateMode: false,
      groupConfirmActive: false,
      targetCursorActive: true,
      roiPlacementActive: false,
      samBoxActive: false,
    },
    segments: {
      items: [],
      highlightedSegmentId: null,
      hoverPoint: null,
      hoverCount: 0,
      tooMany: false,
      onClick: noop,
      onMouseMove: noop,
      onMouseLeave: noop,
      groupSelectionBBox: null,
      groupHighlightedSegmentIds: [],
    },
    roi: {
      activeRoi: null,
      completedRois: [],
      roiPoints: [],
      roiPointsSubmitted: 0,
      roiComplete: false,
      roiLabelMode: "positive",
      brushSize: 24,
      brushColor: "#33cc66",
      roiStrokes: [],
      onRoiLabelModeChange: noop,
      onBrushSizeChange: noop,
      onBrushStroke: noop,
      onSubmitRoiLabels: noop,
      onClearRoiLabels: noop,
      onReselectRoi: noop,
      onMarkRoiComplete: noop,
    },
    drawing: {
      pendingPolygon: null,
      brushStrokes: [],
      brushSize: 24,
      onDrawComplete: noop,
      onBrushStroke: noop,
      onEraseStroke: noop,
      onAddStroke: noop,
      onAccept: noop,
      onCancel: noop,
    },
    uncertain: {
      limit: 50,
      onLimitChange: noop,
      onRefresh: noop,
    },
    completedRoi: {
      active: false,
      loading: false,
      mode: "include",
      items: [],
      polygons: [],
      liveSectionPoints: [],
      hasDraft: false,
      canClosePolygon: false,
      canSave: false,
      isSaving: false,
      onModeChange: noop,
      onClosePolygon: noop,
      onRequestSave: noop,
      onClear: noop,
    },
    feedback: {
      items: [],
    },
    overlays: {},
    ...overrides,
  };
}
