import { CONFIRMED_AREA_TOOLTIP } from "@/shared/constants/confirmedArea";
import type {
  LeftPanelCompletedRoiState,
  LeftPanelDrawingState,
} from "@/features/segmentation/components/leftPanel/types";

interface LeftPanelDrawingActionsProps {
  drawing: LeftPanelDrawingState;
  completedRoi: LeftPanelCompletedRoiState;
}

export function LeftPanelDrawingActions({
  drawing,
  completedRoi,
}: LeftPanelDrawingActionsProps) {
  if (completedRoi.active) {
    const isExclude = completedRoi.mode === "exclude";
    return (
      <div className="polygon-actions confirmed-area-actions">
        <div className="confirmed-area-mode-toggle">
          <button
            type="button"
            className={`confirmed-area-mode-button include${
              isExclude ? "" : " active"
            }`}
            aria-pressed={!isExclude}
            disabled={completedRoi.isSaving}
            onClick={() => completedRoi.onModeChange("include")}
          >
            Include
          </button>
          <button
            type="button"
            className={`confirmed-area-mode-button exclude${
              isExclude ? " active" : ""
            }`}
            aria-pressed={isExclude}
            disabled={completedRoi.isSaving}
            onClick={() => completedRoi.onModeChange("exclude")}
          >
            Exclude
          </button>
        </div>
        <button
          type="button"
          disabled={!completedRoi.canClosePolygon}
          onClick={completedRoi.onClosePolygon}
        >
          Close Polygon (R)
        </button>
        <button
          type="button"
          disabled={!completedRoi.canSave || completedRoi.isSaving}
          onClick={completedRoi.onRequestSave}
          title={CONFIRMED_AREA_TOOLTIP}
        >
          {completedRoi.isSaving
            ? "Saving..."
            : isExclude
              ? "Remove from confirmed area"
              : "Add to confirmed area"}
        </button>
        <button
          type="button"
          disabled={!completedRoi.hasDraft}
          onClick={completedRoi.onClear}
        >
          Clear
        </button>
      </div>
    );
  }

  if (drawing.pendingPolygon) {
    return (
      <div className="polygon-actions">
        <button onClick={drawing.onAccept}>Accept Shape</button>
        <button onClick={drawing.onCancel}>Cancel</button>
      </div>
    );
  }

  return null;
}
