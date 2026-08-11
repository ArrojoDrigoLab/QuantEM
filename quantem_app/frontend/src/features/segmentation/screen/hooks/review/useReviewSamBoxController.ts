/**
 * Binds the box-to-object tool to the labeling screen.
 *
 * The tool itself lives in `features/sam` and knows nothing about this screen.
 * This adapter supplies the three things it needs from here -- what counts as
 * "available", what to refresh once an object lands, and where an error goes --
 * and hands back the slot content and the transient overlays the screen already
 * knows how to render.
 */

import { useCallback, useMemo } from "react";
import { createElement, type ReactNode } from "react";
import { SamBoxToolControls } from "@/features/sam/SamBoxToolControls";
import { useSamBoxTool, type SamBoxTool } from "@/features/sam/useSamBoxTool";
import type { SamBoxResponse } from "@/features/sam/types";
import type { WorkflowMode } from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import type { CorrectionModeState } from "@/shared/types/segmentation";
import type { SegmentOverlay } from "@/viewer/types";

interface UseReviewSamBoxControllerArgs {
  currentSegmentationId: string | null;
  workflowMode: WorkflowMode;
  correctionMode: CorrectionModeState;
  leftNavigateMode: boolean;
  /** Tissue segmentations run a different toolbar entirely. */
  isTissueSegmentation: boolean;
  /** Refetch the objects and the overlay raster once the object is stored. */
  onOverlayMutation: (overlay: unknown) => void;
  showErrorToast: (message: string) => void;
  showNoticeToast?: (message: string) => void;
  registerAnnotationActivity?: () => void;
}

export interface ReviewSamBoxController {
  tool: SamBoxTool;
  /** True while the tool owns the pointer -- the router checks this. */
  isActive: boolean;
  /** Rendered into the toolbar's `extraModes` slot. */
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
  onOverlayMutation,
  showErrorToast,
  showNoticeToast,
  registerAnnotationActivity,
}: UseReviewSamBoxControllerArgs): ReviewSamBoxController {
  // Offered in the same place as the other correction tools, and never while
  // Navigate owns the pointer.
  const available =
    !isTissueSegmentation &&
    !leftNavigateMode &&
    workflowMode === "review" &&
    correctionMode.reviewPhase === "correction" &&
    Boolean(currentSegmentationId);

  const handleCreated = useCallback(
    (response: SamBoxResponse) => {
      onOverlayMutation(response.overlay);
      if (response.created === 0 && showNoticeToast) {
        // The request succeeded and stored nothing -- a mask that came back
        // thinner than a pixel. Saying so beats a box that silently vanishes.
        showNoticeToast(
          "Nothing was stored: the object found in that box was too thin to keep."
        );
      }
    },
    [onOverlayMutation, showNoticeToast]
  );

  const tool = useSamBoxTool({
    segmentationId: currentSegmentationId,
    available,
    onObjectCreated: handleCreated,
    onError: showErrorToast,
    registerActivity: registerAnnotationActivity,
  });

  const controls = useMemo(
    () => (available ? createElement(SamBoxToolControls, { tool }) : null),
    [available, tool]
  );

  return {
    tool,
    isActive: tool.isActive,
    controls,
    overlays: tool.overlays,
    handleImagePress: tool.handleImagePress,
    handleImageDrag: tool.handleImageDrag,
    handleImageRelease: tool.handleImageRelease,
  };
}
