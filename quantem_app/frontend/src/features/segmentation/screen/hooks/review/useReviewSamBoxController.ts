/** Binds the box-to-object correction tool to the labeling screen. */

import { createElement, type ReactNode, useCallback, useEffect, useMemo } from "react";

import { SamBoxToolControls } from "@/features/sam/SamBoxToolControls";
import { useSamBoxTool, type SamBoxTool } from "@/features/sam/useSamBoxTool";
import type { SamBoxResponse } from "@/features/sam/types";
import type { WorkflowMode } from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import type {
  CorrectionModeState,
  CorrectionTool,
} from "@/shared/types/segmentation";
import type { SegmentOverlay } from "@/viewer/types";

interface UseReviewSamBoxControllerArgs {
  currentSegmentationId: string | null;
  workflowMode: WorkflowMode;
  correctionMode: CorrectionModeState;
  leftNavigateMode: boolean;
  /** Tissue segmentations run a different toolbar entirely. */
  isTissueSegmentation: boolean;
  onCorrectionToolChange: (tool: CorrectionTool) => void;
  /** Refetch the objects and overlay raster after an object is stored. */
  onOverlayMutation: (overlay: unknown) => void;
  showErrorToast: (message: string) => void;
  showNoticeToast?: (message: string) => void;
  registerAnnotationActivity?: () => void;
}

export interface ReviewSamBoxController {
  tool: SamBoxTool;
  /** The user selected box-to-object, so the viewer reserves the drag for it. */
  isSelected: boolean;
  /** True while the tool has claimed the pointer. */
  isActive: boolean;
  /** Rendered as a correction sub-tool, beneath Review / Correct. */
  controls: ReactNode;
  /** Appended to the viewer's transient overlay list. */
  overlays: SegmentOverlay[];
  handleImagePress: (
    imagePoint: { x: number; y: number },
    screenPoint: { x: number; y: number }
  ) => void;
  handleImageDrag: (
    imagePoint: { x: number; y: number },
    screenPoint: { x: number; y: number }
  ) => void;
  handleImageRelease: (
    imagePoint: { x: number; y: number },
    screenPoint: { x: number; y: number }
  ) => void;
}

export function useReviewSamBoxController({
  currentSegmentationId,
  workflowMode,
  correctionMode,
  leftNavigateMode,
  isTissueSegmentation,
  onCorrectionToolChange,
  onOverlayMutation,
  showErrorToast,
  showNoticeToast,
  registerAnnotationActivity,
}: UseReviewSamBoxControllerArgs): ReviewSamBoxController {
  const visible =
    !isTissueSegmentation &&
    !leftNavigateMode &&
    workflowMode === "review" &&
    correctionMode.reviewPhase === "correction" &&
    Boolean(currentSegmentationId);
  const isSelected = visible && correctionMode.correctionTool === "sam";

  const handleCreated = useCallback(
    (response: SamBoxResponse) => {
      onOverlayMutation(response.overlay);
      if (response.created === 0 && showNoticeToast) {
        showNoticeToast(
          "Nothing was stored: the object found in that box was too thin to keep."
        );
      }
    },
    [onOverlayMutation, showNoticeToast]
  );

  const tool = useSamBoxTool({
    segmentationId: currentSegmentationId,
    available: isSelected,
    onObjectCreated: handleCreated,
    onError: showErrorToast,
    registerActivity: registerAnnotationActivity,
  });
  const setToolActive = tool.setActive;

  // Selecting this correction tool must claim the pointer immediately. Leaving
  // it restores the standard draw tool, which also clears any partial box.
  useEffect(() => {
    setToolActive(isSelected);
  }, [isSelected, setToolActive]);

  const toggleTool = useCallback(() => {
    onCorrectionToolChange(isSelected ? "draw" : "sam");
  }, [isSelected, onCorrectionToolChange]);

  const controls = useMemo(
    () =>
      visible
        ? createElement(SamBoxToolControls, {
            tool,
            selected: isSelected,
            onToggle: toggleTool,
          })
        : null,
    [isSelected, toggleTool, tool, visible]
  );

  return {
    tool,
    isSelected,
    isActive: tool.isActive,
    controls,
    overlays: tool.overlays,
    handleImagePress: tool.handleImagePress,
    handleImageDrag: tool.handleImageDrag,
    handleImageRelease: tool.handleImageRelease,
  };
}
