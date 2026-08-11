import type {
  WorkflowMode,
  LeftMode,
} from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import type { Point } from "@/utils/geometry";
import type {
  CompletedRoi,
  CompletedRoiMode,
  CorrectionTool,
  SegmentationRoi,
  SegmentObject,
  UserFeedback,
} from "@/shared/types/segmentation";
import type { AssetDetail } from "@/shared/types/images";
import type { BBox } from "@/shared/types/common";
import type {
  SegmentOverlay,
  ViewerFitBounds,
  ViewerIdMapOverlaySpec,
  ViewerNgffOverlayLayerSpec,
  ViewportState,
} from "@/viewer/types";
import type { LeftPanelLayerStyles } from "@/features/segmentation/overlays/segments";
import type { RoiStroke } from "@/features/segmentation/overlays/roi";
import type { DraftPolygon } from "@/shared/geometry/draftGeometry";

export interface LeftPanelViewerState {
  image: AssetDetail;
  segmentationTypeInternalName?: string | null;
  useSmoothedGeometry: boolean;
  layerStyles: LeftPanelLayerStyles;
  viewport: ViewportState | null;
  onViewportChange: (viewport: ViewportState) => void;
  overlayNgffLayers?: ViewerNgffOverlayLayerSpec[];
  /** The ID-map segmentation review overlay (labels + border + render-time LUT). */
  idMapOverlay?: ViewerIdMapOverlaySpec | null;
  onOverlayRevisionDisplayed?: (revision: number | null) => void;
  transientFitBounds?: ViewerFitBounds | null;
  transientFitBoundsKey?: string | null;
}

export interface LeftPanelWorkflowState {
  mode: WorkflowMode;
  leftMode: LeftMode;
  reviewPhase: "model" | "correction";
  correctionTool: CorrectionTool;
  navigateMode: boolean;
  groupConfirmActive: boolean;
  targetCursorActive: boolean;
  /** ROI placement is active -- the viewer must route clicks to placement. */
  roiPlacementActive: boolean;
  /** Box-to-object owns the next drag instead of the brush. */
  samBoxActive?: boolean;
}

export interface LeftPanelSegmentState {
  items: SegmentObject[];
  highlightedSegmentId: string | null;
  hoverPoint: Point | null;
  hoverCount: number;
  tooMany: boolean;
  onClick: (point: Point) => void;
  onPress?: (point: Point, screenPoint: Point) => void;
  onDrag?: (point: Point, screenPoint: Point) => void;
  onRelease?: (point: Point, screenPoint: Point) => void;
  onMouseMove: (point: Point) => void;
  onMouseLeave: () => void;
  groupSelectionBBox: BBox | null;
  groupHighlightedSegmentIds: string[];
}

export interface LeftPanelRoiState {
  activeRoi: SegmentationRoi | null;
  /** Every rectangular ROI marked done for this segmentation. */
  completedRois: SegmentationRoi[];
  roiPoints: Array<{ x: number; y: number; label: number; size: number }>;
  roiPointsSubmitted: number;
  roiComplete: boolean;
  roiLabelMode: "positive" | "negative";
  brushSize: number;
  brushColor: string;
  roiStrokes: RoiStroke[];
  onRoiLabelModeChange: (mode: "positive" | "negative") => void;
  onBrushSizeChange: (size: number) => void;
  onBrushStroke: (points: Point[]) => void;
  onSubmitRoiLabels: () => void;
  onClearRoiLabels: () => void;
  onReselectRoi: () => void;
  onMarkRoiComplete: () => void;
}

export interface LeftPanelDrawingState {
  pendingPolygon: Point[] | null;
  brushStrokes: RoiStroke[];
  brushSize: number;
  onDrawComplete: (points: Point[]) => void;
  onBrushStroke: (points: Point[]) => void;
  onEraseStroke: (points: Point[]) => void;
  /** The "add" correction tool: paint a stroke to add an object. */
  onAddStroke: (points: Point[]) => void;
  onAccept: () => void;
  onCancel: () => void;
}

export interface LeftPanelUncertainState {
  limit: number;
  onLimitChange: (limit: number) => void;
  onRefresh: () => void;
}

export interface LeftPanelCompletedRoiState {
  active: boolean;
  loading: boolean;
  mode: CompletedRoiMode;
  items: CompletedRoi[];
  polygons: DraftPolygon[];
  liveSectionPoints: Point[];
  hasDraft: boolean;
  canClosePolygon: boolean;
  canSave: boolean;
  isSaving: boolean;
  onModeChange: (mode: CompletedRoiMode) => void;
  onClosePolygon: () => void;
  onRequestSave: () => void;
  onClear: () => void;
}

export interface LeftPanelFeedbackState {
  items: UserFeedback[];
}

export interface LeftPanelOverlayState {
  disableCorrectionBrush?: boolean;
  hideActiveRoiOverlay?: boolean;
  extraTransientOverlays?: SegmentOverlay[];
}

export interface SegmentationLeftPanelProps {
  viewer: LeftPanelViewerState;
  workflow: LeftPanelWorkflowState;
  segments: LeftPanelSegmentState;
  roi: LeftPanelRoiState;
  drawing: LeftPanelDrawingState;
  uncertain: LeftPanelUncertainState;
  completedRoi: LeftPanelCompletedRoiState;
  feedback: LeftPanelFeedbackState;
  overlays: LeftPanelOverlayState;
}
