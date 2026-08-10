/**
 * Panel for ROI annotation controls and status.
 */

import type { SegmentationRoi } from "@/shared/types";
import "./RoiAnnotationPanel.css";

interface RoiAnnotationPanelProps {
  activeRoi?: SegmentationRoi | null;
  roiPoints: Array<{ x: number; y: number; label: number; size: number }>;
  roiPointsSubmitted: number;
  roiComplete?: boolean;
  roiLabelMode: "positive" | "negative";
  brushSize: number;
  onRoiLabelModeChange: (mode: "positive" | "negative") => void;
  onBrushSizeChange: (size: number) => void;
  onSubmitRoiLabels: () => void;
  onClearRoiLabels: () => void;
  onReselectRoi?: () => void;
  onMarkRoiComplete?: () => void;
}

export function RoiAnnotationPanel({
  roiPoints,
  roiPointsSubmitted,
  roiLabelMode,
  brushSize,
  onRoiLabelModeChange,
  onBrushSizeChange,
  onSubmitRoiLabels,
  onClearRoiLabels,
}: RoiAnnotationPanelProps) {
  return (
    <div className="roi-annotation-panel">
      <div className="roi-meta">
        <span>Target: current segmentation</span>
        <span>
          Labels: {roiPoints.length}{" "}
          {roiPoints.length > roiPointsSubmitted
            ? `(pending ${roiPoints.length - roiPointsSubmitted})`
            : ""}
        </span>
      </div>
      <div className="roi-hint">
        Click and drag to paint positive or negative areas.
      </div>
      <div className="roi-tools">
        <button
          className={roiLabelMode === "positive" ? "active" : ""}
          onClick={() => onRoiLabelModeChange("positive")}
        >
          Positive
        </button>
        <button
          className={roiLabelMode === "negative" ? "active" : ""}
          onClick={() => onRoiLabelModeChange("negative")}
        >
          Negative
        </button>
        <div className="roi-brush">
          <label htmlFor="brush-size">
            Brush: {Math.round(brushSize)}px
          </label>
          <input
            id="brush-size"
            type="range"
            min={4}
            max={80}
            step={2}
            value={brushSize}
            onChange={(e) => onBrushSizeChange(Number(e.target.value))}
          />
        </div>
        <button
          onClick={onSubmitRoiLabels}
          disabled={roiPoints.length === roiPointsSubmitted}
        >
          Train RF
        </button>
        <button onClick={onClearRoiLabels} disabled={roiPoints.length === 0}>
          Clear Labels
        </button>
      </div>
    </div>
  );
}
