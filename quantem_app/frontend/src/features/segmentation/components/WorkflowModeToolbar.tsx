/**
 * Toolbar for selecting workflow mode and interaction tools.
 *
 * Three packages want to add a control here, and three packages editing one
 * JSX tree is how a toolbar ends up with two Confirm buttons and a merge
 * conflict. So the file has one owner and two named slots instead:
 * `extraModes` for anything that changes what a click means, `extraTools` for
 * anything that acts on the current selection. A package renders into a slot
 * from its own file and never opens this one.
 */

import type { ReactNode } from "react";
import {
  CONFIRMED_AREA_API_ALIAS_NOTE,
  CONFIRMED_AREA_EXPLANATION,
  CONFIRMED_AREA_LABEL,
  CONFIRMED_AREA_TOOLTIP,
} from "@/shared/constants/confirmedArea";
import type { WorkflowMode } from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import type {
  GroupHoverActionMode,
  HoverActionMode,
} from "@/hooks/useHoverSelection";
import type { CorrectionTool } from "@/shared/types/segmentation";
import "./WorkflowModeToolbar.css";

const ICON_PROPS = {
  width: 18,
  height: 18,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

/** Filled-polygon (pentagon) tool icon. */
function PolygonIcon() {
  return (
    <svg {...ICON_PROPS}>
      <polygon points="12 3 21 9.5 17.5 20 6.5 20 3 9.5" />
    </svg>
  );
}

/** Brush/pencil draw tool icon. */
function DrawIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M12 19l7-7a2.1 2.1 0 0 0-3-3l-7 7-1 4z" />
      <path d="M8 16l-4 4" />
    </svg>
  );
}

/** Eraser tool icon. */
function EraseIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M16.5 4.5l3 3a2 2 0 0 1 0 2.8L11 19H6.5L4 16.5a2 2 0 0 1 0-2.8l9.7-9.2a2 2 0 0 1 2.8 0z" />
      <path d="M9 11l4 4" />
    </svg>
  );
}

interface WorkflowModeToolbarProps {
  workflowMode: WorkflowMode;
  reviewPhase: "model" | "correction";
  correctionTool: CorrectionTool;
  hoverActionMode: HoverActionMode;
  drawBrushSize: number;
  hasDrawStrokes: boolean;
  onReviewPhaseChange: (phase: "model" | "correction") => void;
  onCorrectionToolChange: (tool: CorrectionTool) => void;
  onHoverActionModeChange: (mode: HoverActionMode) => void;
  onDrawBrushSizeChange: (size: number) => void;
  onConfirmShape: () => void;
  onClearDrawing: () => void;
  showGroupConfirm?: boolean;
  showTestPoint?: boolean;
  /** ER Review uses group selection only -- hides the point-action buttons. */
  isErSegmentation?: boolean;
  canApplyGroupAction?: boolean;
  onApplyGroupAction?: (mode: GroupHoverActionMode) => void;
  /** ER polygon tool: a draft polygon is in progress. */
  polygonHasDraft?: boolean;
  /** ER polygon tool: the draft can be closed into a filled object. */
  polygonCanClose?: boolean;
  /** ER polygon tool: close the draft and commit it as a filled ER object. */
  onClosePolygon?: () => void;
  /**
   * Extra mode buttons, rendered beside Review / Correct.
   *
   * For controls that change what a click on the image does.
   */
  extraModes?: ReactNode;
  /**
   * Extra tool controls, rendered at the end of the toolbar.
   *
   * For controls that act on the current object or selection rather than
   * changing the mode.
   */
  extraTools?: ReactNode;
}

export function WorkflowModeToolbar({
  workflowMode,
  reviewPhase,
  correctionTool,
  hoverActionMode,
  drawBrushSize,
  hasDrawStrokes,
  onReviewPhaseChange,
  onCorrectionToolChange,
  onHoverActionModeChange,
  onDrawBrushSizeChange,
  onConfirmShape,
  onClearDrawing,
  showGroupConfirm = true,
  showTestPoint = false,
  isErSegmentation = false,
  canApplyGroupAction = false,
  onApplyGroupAction,
  polygonHasDraft = false,
  polygonCanClose = false,
  onClosePolygon,
  extraModes,
  extraTools,
}: WorkflowModeToolbarProps) {
  const canConfirm = correctionTool === "draw" ? hasDrawStrokes : false;
  const canClear = correctionTool === "draw" ? hasDrawStrokes : false;
  const handleGroupActionClick = (mode: GroupHoverActionMode) => {
    if (hoverActionMode === mode && canApplyGroupAction && onApplyGroupAction) {
      onApplyGroupAction(mode);
      return;
    }
    onHoverActionModeChange(mode);
  };

  return (
    <div className="mode-toolbar">
      {workflowMode === "review" && (
        <div className="mode-toolbar-group">
          <button
            className={reviewPhase === "model" ? "active" : ""}
            onClick={() => onReviewPhaseChange("model")}
            aria-pressed={reviewPhase === "model"}
          >
            Review
          </button>
          <button
            className={reviewPhase === "correction" ? "active" : ""}
            onClick={() => onReviewPhaseChange("correction")}
            aria-pressed={reviewPhase === "correction"}
          >
            Correct
          </button>
          {extraModes}
        </div>
      )}

      {workflowMode === "review" && reviewPhase === "model" && (
        <>
          {!isErSegmentation && (
          <div className="hover-action-buttons">
            <button
              className={`mode-action-button mode-action-button--confirm ${
                hoverActionMode === "confirm" ? "active" : ""
              }`}
              onClick={() => onHoverActionModeChange("confirm")}
              aria-pressed={hoverActionMode === "confirm"}
            >
              Confirm Object
            </button>
            <button
              className={`mode-action-button mode-action-button--reject ${
                hoverActionMode === "reject" ? "active" : ""
              }`}
              onClick={() => onHoverActionModeChange("reject")}
              aria-pressed={hoverActionMode === "reject"}
            >
              Reject Object
            </button>
            {showTestPoint && (
              <button
                className={`mode-action-button mode-action-button--test ${
                  hoverActionMode === "test" ? "active" : ""
                }`}
                onClick={() => onHoverActionModeChange("test")}
                aria-pressed={hoverActionMode === "test"}
              >
                Test Point
              </button>
            )}
          </div>
          )}
          {showGroupConfirm && (
            <div className="group-action-buttons">
              <button
                className={`mode-action-button mode-action-button--group-confirm ${
                  hoverActionMode === "group-confirm" ? "active" : ""
                }`}
                onClick={() => handleGroupActionClick("group-confirm")}
                aria-pressed={hoverActionMode === "group-confirm"}
              >
                Confirm Group
              </button>
              <button
                className={`mode-action-button mode-action-button--group-reject ${
                  hoverActionMode === "group-reject" ? "active" : ""
                }`}
                onClick={() => handleGroupActionClick("group-reject")}
                aria-pressed={hoverActionMode === "group-reject"}
              >
                Reject Group
              </button>
            </div>
          )}
        </>
      )}

      {workflowMode === "review" && reviewPhase === "correction" && isErSegmentation && (
        <>
          {/* Three drawing tools as icon buttons on one line. */}
          <div className="mode-toolbar-group mode-tool-icons">
            <button
              type="button"
              className={`icon-tool-button ${correctionTool === "polygon" ? "active" : ""}`}
              onClick={() => onCorrectionToolChange("polygon")}
              aria-pressed={correctionTool === "polygon"}
              aria-label="Polygon"
              title="Polygon: click to place vertices, press R (or Close polygon) to fill"
            >
              <PolygonIcon />
            </button>
            <button
              type="button"
              className={`icon-tool-button ${correctionTool === "draw" ? "active" : ""}`}
              onClick={() => onCorrectionToolChange("draw")}
              aria-pressed={correctionTool === "draw"}
              aria-label="Draw"
              title="Draw: paint with the brush (adjust diameter below)"
            >
              <DrawIcon />
            </button>
            <button
              type="button"
              className={`icon-tool-button ${correctionTool === "erase" ? "active" : ""}`}
              onClick={() => onCorrectionToolChange("erase")}
              aria-pressed={correctionTool === "erase"}
              aria-label="Erase"
              title="Erase drawn strokes and delete model candidates under the brush"
            >
              <EraseIcon />
            </button>
          </div>

          {/* Confirmed area is a separate, distinct control (training mask).
              The explanation sits next to the button because it is the only
              thing that makes the button's purpose guessable. */}
          <div className="mode-toolbar-group confirmed-area-group">
            <button
              className={correctionTool === "completed_roi" ? "active" : ""}
              onClick={() => onCorrectionToolChange("completed_roi")}
              aria-pressed={correctionTool === "completed_roi"}
              aria-describedby="confirmed-area-explainer"
              title={CONFIRMED_AREA_TOOLTIP}
            >
              {CONFIRMED_AREA_LABEL}
            </button>
            <p className="mode-toolbar-explainer" id="confirmed-area-explainer">
              {CONFIRMED_AREA_EXPLANATION}{" "}
              <span className="mode-toolbar-alias">
                {CONFIRMED_AREA_API_ALIAS_NOTE}
              </span>
            </p>
          </div>

          {(correctionTool === "draw" || correctionTool === "erase") && (
            <div className="mode-toolbar-group">
              <label className="prompt-edge-toggle" htmlFor="draw-brush-size">
                Brush diameter
                <input
                  id="draw-brush-size"
                  type="range"
                  min={4}
                  max={128}
                  step={1}
                  value={drawBrushSize}
                  onChange={(event) =>
                    onDrawBrushSizeChange(Number(event.target.value))
                  }
                />
                <span>{drawBrushSize}px</span>
              </label>
            </div>
          )}

          {correctionTool === "draw" && (
            <div className="mode-toolbar-group">
              <button disabled={!canConfirm} onClick={onConfirmShape}>
                Confirm Drawn Area
              </button>
              <button disabled={!canClear} onClick={onClearDrawing}>
                Clear
              </button>
            </div>
          )}

          {correctionTool === "polygon" && (
            <div className="mode-toolbar-group">
              {polygonHasDraft ? (
                <button disabled={!polygonCanClose} onClick={onClosePolygon}>
                  Close polygon (R)
                </button>
              ) : (
                <span className="mode-toolbar-hint">
                  Click to place vertices; press R to fill.
                </span>
              )}
            </div>
          )}
        </>
      )}

      {workflowMode === "review" && reviewPhase === "correction" && !isErSegmentation && (
        <>
          <div className="mode-toolbar-group">
            <button
              className={correctionTool === "draw" ? "active" : ""}
              onClick={() => onCorrectionToolChange("draw")}
              aria-pressed={correctionTool === "draw"}
            >
              Draw
            </button>
            <button
              className={correctionTool === "completed_roi" ? "active" : ""}
              onClick={() => onCorrectionToolChange("completed_roi")}
              aria-pressed={correctionTool === "completed_roi"}
              aria-describedby="confirmed-area-explainer-simple"
              title={CONFIRMED_AREA_TOOLTIP}
            >
              {CONFIRMED_AREA_LABEL}
            </button>
          </div>

          <p className="mode-toolbar-explainer" id="confirmed-area-explainer-simple">
            {CONFIRMED_AREA_EXPLANATION}{" "}
            <span className="mode-toolbar-alias">
              {CONFIRMED_AREA_API_ALIAS_NOTE}
            </span>
          </p>

          {correctionTool === "draw" && (
            <div className="mode-toolbar-group">
              <label className="prompt-edge-toggle" htmlFor="draw-brush-size">
                Brush diameter
                <input
                  id="draw-brush-size"
                  type="range"
                  min={4}
                  max={128}
                  step={1}
                  value={drawBrushSize}
                  onChange={(event) =>
                    onDrawBrushSizeChange(Number(event.target.value))
                  }
                />
                <span>{drawBrushSize}px</span>
              </label>
            </div>
          )}

          {correctionTool === "draw" && (
            <div className="mode-toolbar-group">
              <button disabled={!canConfirm} onClick={onConfirmShape}>
                Confirm Drawn Area
              </button>
              <button disabled={!canClear} onClick={onClearDrawing}>
                Clear
              </button>
            </div>
          )}
        </>
      )}

      {extraTools && <div className="mode-toolbar-group">{extraTools}</div>}
    </div>
  );
}
