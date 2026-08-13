import { useCallback, useMemo } from "react";
import { useDrawing } from "@/hooks/useDrawing";
import { useReviewDrawController } from "@/features/segmentation/screen/hooks/review/useReviewDrawController";
import { useReviewModeState } from "@/features/segmentation/screen/hooks/review/useReviewModeState";
import { useReviewGroupSelection } from "@/features/segmentation/screen/hooks/useReviewGroupSelection";
import { useReviewPointActions } from "@/features/segmentation/screen/hooks/useReviewPointActions";
import type {
  ConfirmBatchResponse,
  CorrectionTool,
  ImageSegmentation,
  LabelState,
  SegmentationOverlayMutationState,
  SegmentObject,
} from "@/shared/types";
import type { Point } from "@/utils/geometry";

interface UseSegmentationReviewWorkflowArgs {
  currentSegmentation: ImageSegmentation | null;
  activeSourceModel: string | null;
  isErSegmentation: boolean;
  supportsPointFeedback: boolean;
  hoverSegments: SegmentObject[];
  highlightedSegmentId: string | null;
  hoverPoint: Point | null;
  hoverActionMode: "confirm" | "reject" | "group-confirm" | "group-reject" | "test";
  setHoverActionMode: (mode: "confirm" | "reject" | "group-confirm" | "group-reject" | "test") => void;
  clearHoverInteraction: () => void;
  applyLabelOverrides: (items: SegmentObject[]) => SegmentObject[];
  applyOptimisticLabel: (
    segmentId: string,
    labelState: LabelState,
    sourceSegment?: SegmentObject | null,
    options?: { stageOverlay?: boolean }
  ) => void;
  rollbackOptimisticLabel: (segmentId: string) => void;
  stageOptimisticRevisionTargets: (segmentIds: string[], targetRevision?: number | null) => void;
  getOptimisticTargetRevision: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => number | null;
  handleOverlayMutationRefresh: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => void;
  registerAnnotationActivity: () => void;
  showErrorToast: (message: string) => void;
  /** Something that worked but did not do what the gesture looked like. */
  showNoticeToast: (message: string) => void;
  /**
   * Leave Navigate mode.
   *
   * Navigate is on by default and suppresses every labeling click and drag, so
   * picking a drawing tool used to leave the user painting into a dead canvas
   * with only a passive "Navigate mode is active" note to explain it. Choosing
   * a tool is an unambiguous statement of intent to label, so it turns Navigate
   * off; the A shortcut and the checkbox still put it back.
   */
  exitNavigateMode: () => void;
  drawing: ReturnType<typeof useDrawing>;
  submitConfirmedGeometriesOptimistically: (options: {
    geometries?: Array<Array<[number, number]>>;
    geometryRings?: Array<Array<Array<[number, number]>>>;
    operations?: Array<"include" | "exclude">;
    samScores?: Array<number | null | undefined>;
    mergeOverlaps?: boolean;
    manualCreation?: boolean;
  }) => Promise<ConfirmBatchResponse | null>;
}

export function useSegmentationReviewWorkflow({
  currentSegmentation,
  activeSourceModel,
  isErSegmentation,
  supportsPointFeedback,
  hoverSegments,
  highlightedSegmentId,
  hoverPoint,
  hoverActionMode,
  setHoverActionMode,
  clearHoverInteraction,
  applyLabelOverrides,
  applyOptimisticLabel,
  rollbackOptimisticLabel,
  stageOptimisticRevisionTargets,
  getOptimisticTargetRevision,
  handleOverlayMutationRefresh,
  registerAnnotationActivity,
  showErrorToast,
  showNoticeToast,
  exitNavigateMode,
  drawing,
  submitConfirmedGeometriesOptimistically,
}: UseSegmentationReviewWorkflowArgs) {
  const modeState = useReviewModeState({
    currentSegmentationId: currentSegmentation?.id,
    isErSegmentation,
    supportsPointFeedback,
    hoverActionMode,
    setHoverActionMode,
  });
  const group = useReviewGroupSelection({
    currentSegmentation,
    activeSourceModel,
    isErSegmentation,
    workflowMode: modeState.workflowMode,
    correctionMode: modeState.correctionMode,
    leftMode: modeState.leftMode,
    hoverActionMode,
    registerAnnotationActivity,
    applyOptimisticLabel,
    rollbackOptimisticLabel,
    clearHoverInteraction,
    stageOptimisticRevisionTargets,
    getOptimisticTargetRevision,
    handleOverlayMutationRefresh,
    showErrorToast,
  });
  const pointActions = useReviewPointActions({
    currentSegmentation,
    activeSourceModel,
    hoverPoint,
    hoverSegments,
    highlightedSegmentId,
    applyLabelOverrides,
    applyOptimisticLabel,
    rollbackOptimisticLabel,
    clearHoverInteraction,
    registerAnnotationActivity,
    stageOptimisticRevisionTargets,
    getOptimisticTargetRevision,
    handleOverlayMutationRefresh,
    showErrorToast,
  });
  const draw = useReviewDrawController({
    currentSegmentation,
    activeSourceModel,
    isErSegmentation,
    drawing,
    registerAnnotationActivity,
    handleOverlayMutationRefresh,
    showErrorToast,
    showNoticeToast,
    submitConfirmedGeometriesOptimistically,
  });

  const reviewInteractionSegments = useMemo(() => {
    const base = applyLabelOverrides(hoverSegments);
    const extra = applyLabelOverrides(group.groupSelectionPreviewSegments);
    if (extra.length === 0) return base;
    const seen = new Set(base.map((segment) => segment.id));
    const merged = [...base];
    for (const segment of extra) {
      if (!seen.has(segment.id)) {
        seen.add(segment.id);
        merged.push(segment);
      }
    }
    return merged;
  }, [applyLabelOverrides, group.groupSelectionPreviewSegments, hoverSegments]);

  const handleReviewPhaseChange = useCallback(
    (phase: "model" | "correction") => {
      modeState.setCorrectionMode((prev) => ({ ...prev, reviewPhase: phase }));
      if (phase === "model") {
        drawing.clearDrawing();
      } else {
        // Switching to Correct is a request to edit, not to pan.
        exitNavigateMode();
      }
    },
    [drawing, exitNavigateMode, modeState]
  );

  /**
   * The user picked a labeling action (Confirm Object, Confirm Group, ...).
   *
   * Like choosing a drawing tool, this is an unambiguous statement of intent to
   * label, so it leaves Navigate. Without it, switching to Confirm Group left
   * Navigate on and the first box-drag panned the image while the side panel
   * said "Drag a box in labeling view" -- the interaction router drops every
   * press/drag/release while Navigate is on.
   *
   * Deliberately separate from the raw `setHoverActionMode` prop, which
   * `useReviewModeState` also calls to *coerce* the mode (ER has no point
   * tools; a segmentation without point feedback cannot stay on a group mode).
   * Those are the app correcting itself, not the user asking to label, and they
   * must not silently disable panning.
   */
  const handleHoverActionModeChange = useCallback(
    (mode: "confirm" | "reject" | "group-confirm" | "group-reject" | "test") => {
      setHoverActionMode(mode);
      exitNavigateMode();
    },
    [exitNavigateMode, setHoverActionMode]
  );

  const handleCorrectionToolChange = useCallback(
    (tool: CorrectionTool) => {
      modeState.setCorrectionMode((prev) => ({ ...prev, correctionTool: tool }));
      group.clearGroupBboxSelection();
      clearHoverInteraction();
      exitNavigateMode();
      // "erase" keeps the drawn strokes so the eraser can remove them; every
      // other tool starts from a clean canvas.
      if (tool !== "draw" && tool !== "erase") {
        drawing.clearDrawing();
      }
    },
    [clearHoverInteraction, drawing, exitNavigateMode, group, modeState]
  );

  return {
    mode: {
      workflowMode: modeState.workflowMode,
      leftMode: modeState.leftMode,
      correctionMode: modeState.correctionMode,
      isCorrectionReview: modeState.isCorrectionReview,
      hideModelParameterControls: modeState.hideModelParameterControls,
      isGroupActionModeActive: modeState.isGroupActionModeActive,
      activeGroupActionLabelState: modeState.activeGroupActionLabelState,
      handleReviewPhaseChange,
      handleCorrectionToolChange,
      handleHoverActionModeChange,
    },
    draw,
    group,
    pointActions: {
      handleApplyPointAction: pointActions.handleApplyPointAction,
      handleResetConfirmedToCandidate: pointActions.handleResetConfirmedToCandidate,
    },
    derived: {
      reviewInteractionSegments,
    },
  };
}
