/**
 * Minimal labeling toolbar for the tissue-mask segmentation view.
 *
 * Unlike the standard review toolbar there is no Review/Correct split, no prompt
 * tool and no confirmed-area (training-mask) tool. Just three drawing tools that
 * edit one hand-drawn foreground mask:
 *   - Brush: paint strokes, then confirm them into the mask.
 *   - Polygon: click-to-trace a ring (R to close) added to the mask.
 *   - Exclude Polygon: click-to-trace a ring (R to close) carved out of the mask.
 */

import type { TissueTool } from "@/features/segmentation/screen/hooks/useTissueLabeling";
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

function BrushIcon() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M12 19l7-7a2.1 2.1 0 0 0-3-3l-7 7-1 4z" />
      <path d="M8 16l-4 4" />
    </svg>
  );
}

function PolygonIcon() {
  return (
    <svg {...ICON_PROPS}>
      <polygon points="12 3 21 9.5 17.5 20 6.5 20 3 9.5" />
    </svg>
  );
}

export interface TissueWorkflowControlsProps {
  tool: TissueTool;
  onToolChange: (tool: TissueTool) => void;
  operation: "include" | "exclude";
  onOperationChange: (operation: "include" | "exclude") => void;
  brushSize: number;
  onBrushSizeChange: (size: number) => void;
  canConfirmBrush: boolean;
  hasBrushStrokes: boolean;
  confirmingBrush: boolean;
  onConfirmBrush: () => void;
  onClearBrush: () => void;
  /** State of the polygon draft for whichever polygon tool is active. */
  polygonHasDraft: boolean;
  polygonCanClose: boolean;
  onClosePolygon: () => void;
  onClearPolygon: () => void;
}

export function TissueWorkflowControls({
  tool,
  onToolChange,
  operation,
  onOperationChange,
  brushSize,
  onBrushSizeChange,
  canConfirmBrush,
  hasBrushStrokes,
  confirmingBrush,
  onConfirmBrush,
  onClearBrush,
  polygonHasDraft,
  polygonCanClose,
  onClosePolygon,
  onClearPolygon,
}: TissueWorkflowControlsProps) {
  return (
    <div className="mode-toolbar">
      <div className="mode-toolbar-group" aria-label="Drawing operation">
        <button
          type="button"
          className={operation === "include" ? "active" : ""}
          aria-pressed={operation === "include"}
          onClick={() => onOperationChange("include")}
        >
          Include
        </button>
        <button
          type="button"
          className={operation === "exclude" ? "active" : ""}
          aria-pressed={operation === "exclude"}
          onClick={() => onOperationChange("exclude")}
        >
          Exclude
        </button>
      </div>
      <div className="mode-toolbar-group mode-tool-icons">
        <button
          type="button"
          className={`icon-tool-button ${tool === "brush" ? "active" : ""}`}
          onClick={() => onToolChange("brush")}
          aria-pressed={tool === "brush"}
          aria-label="Brush"
          title="Brush: paint the tissue mask (adjust diameter below), then confirm"
        >
          <BrushIcon />
        </button>
        <button
          type="button"
          className={`icon-tool-button ${tool === "polygon" ? "active" : ""}`}
          onClick={() => onToolChange("polygon")}
          aria-pressed={tool === "polygon"}
          aria-label="Polygon"
          title="Polygon: click to place vertices, press R (or connect the ends) to fill"
        >
          <PolygonIcon />
        </button>
      </div>

      {tool === "brush" && (
        <>
          <div className="mode-toolbar-group">
            <label className="prompt-edge-toggle" htmlFor="tissue-brush-size">
              Brush diameter
              <input
                id="tissue-brush-size"
                type="range"
                min={4}
                max={256}
                step={1}
                value={brushSize}
                onChange={(event) => onBrushSizeChange(Number(event.target.value))}
              />
              <span>{brushSize}px</span>
            </label>
          </div>
          <div className="mode-toolbar-group">
            <button disabled={!canConfirmBrush} onClick={onConfirmBrush}>
              {confirmingBrush ? "Adding..." : "Confirm Drawn Area"}
            </button>
            <button disabled={!hasBrushStrokes || confirmingBrush} onClick={onClearBrush}>
              Clear
            </button>
          </div>
        </>
      )}

      {tool === "polygon" && (
        <div className="mode-toolbar-group">
          {polygonHasDraft ? (
            <>
              <button disabled={!polygonCanClose} onClick={onClosePolygon}>
                {operation === "exclude" ? "Close & exclude (R)" : "Close polygon (R)"}
              </button>
              <button onClick={onClearPolygon}>Clear</button>
            </>
          ) : (
            <span className="mode-toolbar-hint">
              {operation === "exclude"
                ? "Click to place vertices; press R to carve out the area."
                : "Click to place vertices; press R to fill."}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
