import type {
  LeftPanelUncertainState,
  LeftPanelWorkflowState,
} from "@/features/segmentation/components/leftPanel/types";

interface LeftPanelStatusMessageProps {
  workflow: LeftPanelWorkflowState;
  uncertain: LeftPanelUncertainState;
  tooMany: boolean;
}

export function LeftPanelStatusMessage({
  workflow,
  uncertain,
  tooMany,
}: LeftPanelStatusMessageProps) {
  return (
    <>
      {tooMany && (
        <div className="overlay-warning">
          Too many shapes to display at this zoom level. Please zoom in.
        </div>
      )}
      {workflow.mode === "uncertain" && (
        <div className="uncertain-controls">
          <label htmlFor="uncertain-limit">Uncertain count</label>
          <input
            id="uncertain-limit"
            type="number"
            min={10}
            max={200}
            value={uncertain.limit}
            onChange={(e) => uncertain.onLimitChange(Number(e.target.value))}
          />
          <button onClick={uncertain.onRefresh}>Refresh</button>
        </div>
      )}
    </>
  );
}
