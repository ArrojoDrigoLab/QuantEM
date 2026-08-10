import { useCallback, useState } from "react";

export type WorkflowMode = "annotate" | "review" | "uncertain";
export type LeftMode = "hover" | "draw" | "completed_roi" | "annotate" | "add";

export function useSegmentationWorkflowMode() {
  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>("review");
  const [leftMode, setLeftMode] = useState<LeftMode>("hover");

  const handleWorkflowModeChange = useCallback((mode: WorkflowMode) => {
    setWorkflowMode(mode);
    setLeftMode(mode === "annotate" ? "annotate" : "hover");
  }, []);

  return {
    workflowMode,
    leftMode,
    setWorkflowMode: handleWorkflowModeChange,
    setLeftMode,
  };
}
