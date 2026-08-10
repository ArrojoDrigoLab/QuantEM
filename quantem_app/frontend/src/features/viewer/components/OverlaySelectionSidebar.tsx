import type { ReactNode } from "react";
import type { ImageSegmentation, StatusStage } from "@/shared/types";
import "./OverlaySelectionSidebar.css";

const PROCESSING_STAGES: StatusStage[] = [
  "UNSTARTED",
  "RUNNING_INFERENCE",
  "EXTRACTING_CANDIDATES",
];

function isProcessing(stage: StatusStage): boolean {
  return PROCESSING_STAGES.includes(stage);
}

function isCompleted(stage: StatusStage): boolean {
  return stage === "COMPLETED";
}

function getStageLabel(stage: StatusStage): string {
  switch (stage) {
    case "UNSTARTED":
      return "Pending";
    case "RUNNING_INFERENCE":
      return "Running inference...";
    case "EXTRACTING_CANDIDATES":
      return "Extracting candidates...";
    case "CANDIDATES_READY":
      return "Candidates ready";
    case "UPDATING":
      return "Updating...";
    case "COMPUTING_FEATURES":
      return "Computing features...";
    case "COMPLETED":
      return "Completed";
    case "FAILED":
      return "Failed";
    default:
      return stage;
  }
}

interface OverlaySelectionSidebarProps {
  overlays: Array<{
    segmentation: ImageSegmentation;
    enabled: boolean;
    color: string;
    opacity: number;
  }>;
  onToggle: (segmentationId: string) => void;
  onColorChange: (segmentationId: string, color: string) => void;
  onOpacityChange: (segmentationId: string, opacity: number) => void;
  createPanel?: ReactNode;
  onEditSegmentation?: (segmentationId: string) => void;
  /**
   * Ask to delete this segmentation. Opens a confirmation that reads live
   * counts first — this button only starts the question. Offered on every
   * resting state; a processing segmentation has a job on it, which the server
   * refuses anyway, so the button is not shown mid-run.
   */
  onDeleteSegmentation?: (segmentationId: string) => void;
  disabled?: boolean;
  statusMessage?: string;
}

export function OverlaySelectionSidebar({
  overlays,
  onToggle,
  onColorChange,
  onOpacityChange,
  createPanel,
  onEditSegmentation,
  onDeleteSegmentation,
  disabled,
  statusMessage,
}: OverlaySelectionSidebarProps) {
  const getTotalCount = (counts?: Record<string, number>) => {
    if (!counts) return null;
    return Object.values(counts).reduce((sum, value) => sum + (value ?? 0), 0);
  };

  return (
    <aside className="overlay-sidebar">
      <h3>Viewer Overlays</h3>
      <div className="overlay-sidebar-content">
        {disabled && (
          <div className="overlay-sidebar-disabled-mask">
            <span className="overlay-sidebar-disabled-message">
              {statusMessage || "Overlays unavailable"}
            </span>
          </div>
        )}
        {overlays.length === 0 ? (
          <div className="overlay-empty">No segmentations available.</div>
        ) : (
          <div className="overlay-list">
            {overlays.map((overlay) => {
              const stage = overlay.segmentation.status_stage;

              if (isCompleted(stage)) {
                return (
                  <div key={overlay.segmentation.id} className="overlay-item">
                    <div className="overlay-header">
                      <label className="overlay-toggle">
                        <input
                          type="checkbox"
                          checked={overlay.enabled}
                          onChange={() => onToggle(overlay.segmentation.id)}
                        />
                        <span className="overlay-name">
                          {overlay.segmentation.segmentation_type.long_name}
                        </span>
                      </label>
                      {onEditSegmentation && (
                        <button
                          type="button"
                          className="overlay-edit-button"
                          onClick={() =>
                            onEditSegmentation(overlay.segmentation.id)
                          }
                          title="Open labeling"
                        >
                          ✎
                        </button>
                      )}
                      {onDeleteSegmentation && (
                        <button
                          type="button"
                          className="overlay-delete-button"
                          onClick={() =>
                            onDeleteSegmentation(overlay.segmentation.id)
                          }
                          title={`Delete ${overlay.segmentation.segmentation_type.long_name}…`}
                          aria-label={`Delete ${overlay.segmentation.segmentation_type.long_name}`}
                        >
                          🗑
                        </button>
                      )}
                    </div>
                    <div className="overlay-count">
                      {getTotalCount(overlay.segmentation.segment_counts) ?? 0}{" "}
                      objects
                    </div>
                    <div className="overlay-controls">
                      <label className="overlay-color">
                        <span>Color</span>
                        <input
                          type="color"
                          value={overlay.color}
                          onChange={(event) =>
                            onColorChange(
                              overlay.segmentation.id,
                              event.target.value
                            )
                          }
                        />
                      </label>
                      <label className="overlay-opacity">
                        <span>Opacity</span>
                        <input
                          type="range"
                          min={0}
                          max={0.8}
                          step={0.05}
                          value={overlay.opacity}
                          onChange={(event) =>
                            onOpacityChange(
                              overlay.segmentation.id,
                              Number(event.target.value)
                            )
                          }
                        />
                        <span className="overlay-opacity-value">
                          {(overlay.opacity * 100).toFixed(0)}%
                        </span>
                      </label>
                    </div>
                  </div>
                );
              }

              if (isProcessing(stage)) {
                const progress = overlay.segmentation.status_progress;
                return (
                  <div
                    key={overlay.segmentation.id}
                    className="overlay-item overlay-item-processing"
                  >
                    <div className="overlay-header">
                      <span className="overlay-name">
                        {overlay.segmentation.segmentation_type.long_name}
                      </span>
                    </div>
                    <div className="overlay-status-label overlay-status-processing">
                      {getStageLabel(stage)}
                    </div>
                    {progress > 0 && (
                      <div className="overlay-progress-bar">
                        <div
                          className="overlay-progress-fill"
                          style={{ width: `${Math.round(progress)}%` }}
                        />
                      </div>
                    )}
                  </div>
                );
              }

              // Actionable: CANDIDATES_READY, UPDATING, COMPUTING_FEATURES, FAILED
              return (
                <div
                  key={overlay.segmentation.id}
                  className="overlay-item overlay-item-actionable"
                >
                  <div className="overlay-header">
                    <span className="overlay-name">
                      {overlay.segmentation.segmentation_type.long_name}
                    </span>
                  </div>
                  <div
                    className={`overlay-status-label ${stage === "FAILED" ? "overlay-status-failed" : "overlay-status-actionable"}`}
                  >
                    {getStageLabel(stage)}
                  </div>
                  {/* "Candidates ready" is what a finished run leaves behind
                      whether it found two hundred objects or none, so on its own
                      it reads as success over an empty overlay. `run_notice` is
                      the server's finding about exactly that case; the labeling
                      header carries the full list of what to check, so this only
                      says which case it is. */}
                  {overlay.segmentation.run_notice && (
                    <div className="overlay-status-note">
                      {overlay.segmentation.run_notice.message}
                    </div>
                  )}
                  {onEditSegmentation && (
                    <button
                      type="button"
                      className="overlay-edit-button-prominent"
                      onClick={() =>
                        onEditSegmentation(overlay.segmentation.id)
                      }
                    >
                      Edit / Label
                    </button>
                  )}
                  {onDeleteSegmentation && (
                    <button
                      type="button"
                      className="overlay-delete-button-text"
                      onClick={() =>
                        onDeleteSegmentation(overlay.segmentation.id)
                      }
                    >
                      Delete…
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
        {createPanel && (
          <div className="overlay-create-section">
            <h4>Add segmentation</h4>
            {createPanel}
          </div>
        )}
      </div>
    </aside>
  );
}
