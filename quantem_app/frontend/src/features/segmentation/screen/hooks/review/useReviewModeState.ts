import { useEffect, useState } from "react";
import {
  useSegmentationWorkflowMode,
  type LeftMode,
  type WorkflowMode,
} from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import type { CorrectionModeState, LabelState } from "@/shared/types";

interface UseReviewModeStateArgs {
  currentSegmentationId: string | null | undefined;
  isErSegmentation: boolean;
  supportsPointFeedback: boolean;
  hoverActionMode: "confirm" | "reject" | "group-confirm" | "group-reject" | "test";
  setHoverActionMode: (mode: "confirm" | "reject" | "group-confirm" | "group-reject" | "test") => void;
}

function isGroupActionMode(
  workflowMode: WorkflowMode,
  correctionMode: CorrectionModeState,
  leftMode: LeftMode,
  hoverActionMode: "confirm" | "reject" | "group-confirm" | "group-reject" | "test"
) {
  return (
    workflowMode === "review" &&
    correctionMode.reviewPhase === "model" &&
    leftMode === "hover" &&
    (hoverActionMode === "group-confirm" || hoverActionMode === "group-reject")
  );
}

const DEFAULT_CORRECTION_MODE: CorrectionModeState = {
  reviewPhase: "model",
  correctionTool: "draw",
};

export function useReviewModeState({
  currentSegmentationId,
  isErSegmentation,
  supportsPointFeedback,
  hoverActionMode,
  setHoverActionMode,
}: UseReviewModeStateArgs) {
  const { workflowMode, leftMode, setLeftMode } = useSegmentationWorkflowMode();
  const [correctionMode, setCorrectionMode] =
    useState<CorrectionModeState>(DEFAULT_CORRECTION_MODE);

  useEffect(() => {
    setCorrectionMode(DEFAULT_CORRECTION_MODE);
  }, [currentSegmentationId]);

  useEffect(() => {
    if (workflowMode !== "review") {
      if (leftMode !== "hover") {
        setLeftMode("hover");
      }
      return;
    }

    const nextMode: LeftMode =
      correctionMode.reviewPhase === "model"
        ? "hover"
        : correctionMode.correctionTool === "draw" ||
            correctionMode.correctionTool === "erase"
          ? "draw"
          : correctionMode.correctionTool === "completed_roi"
            ? "completed_roi"
            : correctionMode.correctionTool === "add"
              ? "add"
              : // The polygon tool drives clicks through the router's erPolygon
                // slot, so it stays in an inert "hover" leftMode (brush/draw off).
                "hover";
    if (leftMode !== nextMode) {
      setLeftMode(nextMode);
    }
  }, [workflowMode, correctionMode, leftMode, setLeftMode]);

  useEffect(() => {
    if (supportsPointFeedback) return;
    if (
      hoverActionMode !== "group-confirm" &&
      hoverActionMode !== "group-reject" &&
      hoverActionMode !== "test"
    ) {
      return;
    }
    setHoverActionMode("confirm");
  }, [hoverActionMode, setHoverActionMode, supportsPointFeedback]);

  // ER Review uses group selection only (point tools are hidden), so coerce any
  // point mode to a group mode -- this also engages the drag-box selection.
  useEffect(() => {
    if (!isErSegmentation) return;
    if (workflowMode !== "review" || correctionMode.reviewPhase !== "model") return;
    if (
      hoverActionMode === "confirm" ||
      hoverActionMode === "reject" ||
      hoverActionMode === "test"
    ) {
      setHoverActionMode("group-confirm");
    }
  }, [
    isErSegmentation,
    workflowMode,
    correctionMode.reviewPhase,
    hoverActionMode,
    setHoverActionMode,
  ]);

  const isCorrectionReview =
    workflowMode === "review" && correctionMode.reviewPhase === "correction";
  const isGroupActionModeActive = isGroupActionMode(
    workflowMode,
    correctionMode,
    leftMode,
    hoverActionMode
  );
  const activeGroupActionLabelState: LabelState | null =
    hoverActionMode === "group-confirm"
      ? "CONFIRMED"
      : hoverActionMode === "group-reject"
        ? "EXCLUDED"
        : null;

  return {
    workflowMode,
    leftMode,
    correctionMode,
    setCorrectionMode,
    isCorrectionReview,
    hideModelParameterControls: isCorrectionReview,
    isGroupActionModeActive,
    activeGroupActionLabelState,
  };
}
