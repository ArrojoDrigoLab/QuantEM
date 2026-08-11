import { useCallback, useState } from "react";

/**
 * The labeling screen's two mode axes.
 *
 * `annotate` is gone from both. It was a whole third workflow — an ROI you
 * painted with a brush and submitted as a block of labels — whose handlers had
 * already been reduced to `() => {}`, so choosing it put the canvas into a mode
 * that could not do anything and offered no way to tell. Nothing sets it now,
 * so the mode cannot be entered, and `LeftMode`'s matching `"annotate"` value
 * is unreachable too. The panel it used to render is deleted separately.
 */
export type WorkflowMode = "review" | "uncertain";
export type LeftMode = "hover" | "draw" | "completed_roi" | "add";

export function useSegmentationWorkflowMode() {
  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>("review");
  const [leftMode, setLeftMode] = useState<LeftMode>("hover");

  const handleWorkflowModeChange = useCallback((mode: WorkflowMode) => {
    setWorkflowMode(mode);
    setLeftMode("hover");
  }, []);

  return {
    workflowMode,
    leftMode,
    setWorkflowMode: handleWorkflowModeChange,
    setLeftMode,
  };
}
