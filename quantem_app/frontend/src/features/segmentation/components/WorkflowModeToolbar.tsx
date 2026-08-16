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

/** Paint-brush tool icon. */
function BrushIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="m9.1 14.9 8-8a2.85 2.85 0 1 1 4 4l-8 8" />
      <path d="M7.1 14.9a3 3 0 0 0-3 3c0 1.4-1 2.2-2 2.7 1.2.9 2.7 1.4 4.4 1.4a4.5 4.5 0 0 0 4.6-4.5c0-1.4-.9-2.6-2-2.6z" />
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
  draftOperation: "include" | "exclude";
  hasDrawStrokes: boolean;
  onReviewPhaseChange: (phase: "model" | "correction") => void;
  onCorrectionToolChange: (tool: CorrectionTool) => void;
  onHoverActionModeChange: (mode: HoverActionMode) => void;
  onDrawBrushSizeChange: (size: number) => void;
  onDraftOperationChange: (operation: "include" | "exclude") => void;
  onConfirmShape: () => void;
  onClearDrawing: () => void;
  showGroupConfirm?: boolean;
  showTestPoint?: boolean;
  /** ER Review uses group selection only -- hides the point-action buttons. */
  isErSegmentation?: boolean;
  canApplyGroupAction?: boolean;
  onApplyGroupAction?: (mode: GroupHoverActionMode) => void;
  /** Object polygon tool: a draft polygon is in progress. */
  polygonHasDraft?: boolean;
  /** Object polygon tool: the draft can be closed into a filled object. */
  polygonCanClose?: boolean;
  /** Close the draft and commit it as a filled object. */
  onClosePolygon?: () => void;
  /** Extra correction sub-tools, rendered alongside the drawing tools. */
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
  draftOperation,
  hasDrawStrokes,
  onCorrectionToolChange,
  onHoverActionModeChange,
  onDrawBrushSizeChange,
  onDraftOperationChange,
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

      {workflowMode === "review" && reviewPhase === "correction" && (
        <div className="mode-toolbar-group" aria-label="Drawing operation">
          <button
            type="button"
            className={draftOperation === "include" ? "active" : ""}
            aria-pressed={draftOperation === "include"}
            onClick={() => onDraftOperationChange("include")}
          >
            Include
          </button>
          <button
            type="button"
            className={draftOperation === "exclude" ? "active" : ""}
            aria-pressed={draftOperation === "exclude"}
            onClick={() => onDraftOperationChange("exclude")}
          >
            Exclude
          </button>
        </div>
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
              aria-label="Brush"
              title="Brush: paint an object area (adjust diameter below)"
            >
              <BrushIcon />
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
            {extraModes}
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
          <div className="mode-toolbar-group mode-tool-icons">
            <button
              type="button"
              className={`icon-tool-button ${correctionTool === "polygon" ? "active" : ""}`}
              onClick={() => onCorrectionToolChange("polygon")}
              aria-pressed={correctionTool === "polygon"}
              aria-label="Polygon"
              title="Polygon: click to place vertices, press R (or Close polygon) to create the object"
            >
              <PolygonIcon />
            </button>
            <button
              type="button"
              className={`icon-tool-button ${correctionTool === "draw" ? "active" : ""}`}
              onClick={() => onCorrectionToolChange("draw")}
              aria-pressed={correctionTool === "draw"}
              aria-label="Brush"
              title="Brush: paint an object area (adjust diameter below)"
            >
              <BrushIcon />
            </button>
            {extraModes}
          </div>

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

          {correctionTool === "polygon" && (
            <div className="mode-toolbar-group">
              {polygonHasDraft ? (
                <button disabled={!polygonCanClose} onClick={onClosePolygon}>
                  Close polygon (R)
                </button>
              ) : (
                <span className="mode-toolbar-hint">
                  Click to place vertices; press R to create the object.
                </span>
              )}
            </div>
          )}
        </>
      )}

      {extraTools && <div className="mode-toolbar-group">{extraTools}</div>}
    </div>
  );
}
