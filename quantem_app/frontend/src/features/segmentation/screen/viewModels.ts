import type { ComponentProps } from "react";
import { SegmentationHeader } from "@/features/segmentation/components/SegmentationHeader";
import { SegmentationLeftPanel } from "@/features/segmentation/components/SegmentationLeftPanel";
import { SegmentationRightPanel } from "@/features/segmentation/components/SegmentationRightPanel";
import { SegmentationSidebar } from "@/features/segmentation/screen/components/SegmentationSidebar";
import { LABELING_LEFT_PANEL_STATES } from "@/features/segmentation/screen/utils/constants";
import type {
  LeftMode,
  WorkflowMode,
} from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import type { CorrectionTool, SegmentObject } from "@/shared/types";

type SegmentationHeaderProps = ComponentProps<typeof SegmentationHeader>;
type SegmentationSidebarProps = ComponentProps<typeof SegmentationSidebar>;
type SegmentationLeftPanelProps = ComponentProps<typeof SegmentationLeftPanel>;
type SegmentationRightPanelProps = ComponentProps<typeof SegmentationRightPanel>;

export function buildSegmentationHeaderProps(
  props: SegmentationHeaderProps
): SegmentationHeaderProps {
  return props;
}

export function buildSegmentationSidebarProps(
  props: SegmentationSidebarProps
): SegmentationSidebarProps {
  return props;
}

export function buildLeftPanelWorkflowState({
  isTissueSegmentation,
  tissueTool,
  leftNavigateMode,
  workflowMode,
  leftMode,
  correctionMode,
  isGroupActionModeActive,
  roiPlacementActive,
}: {
  isTissueSegmentation: boolean;
  tissueTool: "brush" | "polygon" | "exclude";
  leftNavigateMode: boolean;
  workflowMode: WorkflowMode;
  leftMode: LeftMode;
  correctionMode: {
    reviewPhase: "model" | "correction";
    correctionTool: CorrectionTool;
  };
  isGroupActionModeActive: boolean;
  roiPlacementActive: boolean;
}) {
  if (isTissueSegmentation) {
    // The tissue brush reuses the existing correction-draw brush plumbing; the
    // polygon/exclude tools route clicks through the interaction router, so the
    // panel state only needs to keep the brush off and pan on.
    if (tissueTool === "brush") {
      return {
        mode: "review" as const,
        leftMode: "hover" as const,
        reviewPhase: "correction" as const,
        correctionTool: "draw" as const,
        navigateMode: leftNavigateMode,
        groupConfirmActive: false,
        targetCursorActive: false,
        roiPlacementActive: false,
      };
    }
    return {
      mode: "review" as const,
      leftMode: "hover" as const,
      reviewPhase: "model" as const,
      correctionTool: "polygon" as const,
      navigateMode: leftNavigateMode,
      groupConfirmActive: false,
      targetCursorActive: false,
      roiPlacementActive: false,
    };
  }
  return {
    mode: workflowMode,
    leftMode,
    reviewPhase: correctionMode.reviewPhase,
    correctionTool: correctionMode.correctionTool,
    navigateMode: leftNavigateMode,
    groupConfirmActive: isGroupActionModeActive,
    targetCursorActive: false,
    roiPlacementActive,
  };
}

export function buildSegmentationRoiViewModel(
  activeRoi: SegmentationLeftPanelProps["roi"]["activeRoi"]
): SegmentationLeftPanelProps["roi"] {
  return {
    activeRoi,
    roiPoints: [],
    roiPointsSubmitted: 0,
    roiComplete: Boolean(activeRoi?.is_complete),
    roiLabelMode: "positive",
    brushSize: 24,
    brushColor: "#33cc66",
    roiStrokes: [],
    onRoiLabelModeChange: () => {},
    onBrushSizeChange: () => {},
    onBrushStroke: () => {},
    onSubmitRoiLabels: () => {},
    onClearRoiLabels: () => {},
    onReselectRoi: () => {},
    onMarkRoiComplete: () => {},
  };
}

export function buildSegmentationLeftPanelProps({
  base,
  leftSegments,
  applyLabelOverrides,
  workflowMode,
  reviewInteractionSegments,
  hoverPoint,
  hoverSegments,
  highlightedSegmentId,
  groupSelectionBBox,
  groupBboxHighlightedSegmentIds,
}: {
  base: SegmentationLeftPanelProps;
  leftSegments: SegmentObject[];
  applyLabelOverrides: (items: SegmentObject[]) => SegmentObject[];
  workflowMode: WorkflowMode;
  reviewInteractionSegments: SegmentObject[];
  hoverPoint: { x: number; y: number } | null;
  hoverSegments: SegmentObject[];
  highlightedSegmentId: string | null;
  groupSelectionBBox: SegmentationLeftPanelProps["segments"]["groupSelectionBBox"];
  groupBboxHighlightedSegmentIds: string[];
}): SegmentationLeftPanelProps {
  const appliedLeftSegments =
    workflowMode === "review" ? reviewInteractionSegments : applyLabelOverrides(leftSegments);
  const leftPanelSegments = appliedLeftSegments.filter((segment) =>
    LABELING_LEFT_PANEL_STATES.has(segment.label_state)
  );

  return {
    ...base,
    segments: {
      ...base.segments,
      items: leftPanelSegments,
      highlightedSegmentId: base.completedRoi.active ? null : highlightedSegmentId,
      hoverPoint: base.completedRoi.active ? null : hoverPoint,
      hoverCount: base.completedRoi.active ? 0 : hoverSegments.length,
      groupSelectionBBox,
      groupHighlightedSegmentIds: groupBboxHighlightedSegmentIds,
    },
  };
}

export function buildSegmentationRightPanelProps(
  props: SegmentationRightPanelProps
): SegmentationRightPanelProps {
  return props;
}
