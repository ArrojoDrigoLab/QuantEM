import type { ReactNode } from "react";
import type { ImageSegmentation, StatusStage } from "@/shared/types";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";
import { segmentationHasResults } from "@/shared/constants/segmentation";
import { OverlayBuildFailureNotice } from "./OverlayBuildFailureNotice";
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
    case "THRESHOLD_READY":
      return "Threshold ready";
    case "CANDIDATES_READY":
      return "Run finished";
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

/**
 * The server's reason a run failed, said out loud on the card.
 *
 * `status_error` has always been on the payload this sidebar already holds —
 * `"Model pack 'quantem:er' is not installed."` and the like — and the card
 * rendered the single word "Failed", which distinguishes a missing model from
 * a dead worker from a cancellation not at all. The text is reproduced
 * verbatim (`quantem.jobs.failure_reconcile` writes it) for the same reason
 * the labeling header reproduces it: it is the only string that says which of
 * those happened, and each wants a different response.
 *
 * Rendered on any card carrying the field, not only a `FAILED` one: the retry
 * path writes "Attempt N of M failed; retrying automatically. <error>" onto a
 * segmentation without moving its stage, so a card that only showed this when
 * the stage was FAILED would stay silent while attempts burned.
 */
function OverlayStatusError({
  stage,
  error,
}: {
  stage: StatusStage;
  error?: string | null;
}) {
  const reason = error?.trim();
  // A failed run says something even when the server left the field empty:
  // "Failed" alone is what this card used to be, and it names nothing.
  if (!reason && stage !== "FAILED") return null;
  return (
    <div className="overlay-status-error" role="status">
      {reason || "The server recorded no reason for it."}
    </div>
  );
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
  /**
   * Manifests whose overlay *build* failed, keyed by segmentation id.
   *
   * Distinct from `segmentation.status_error`, which is about the run: a run
   * can succeed and leave hundreds of objects while the raster the viewer
   * draws them from fails to rebuild, and that is exactly the case that used
   * to show "Overlay updating..." for ever with the reason on the wire and
   * nowhere on screen.
   */
  overlayBuildFailures?: Record<string, SegmentationOverlayManifest>;
  /** Refetch the manifest after a retry so the card leaves the failed state. */
  onOverlayBuildRetried?: (segmentationId: string) => void;
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
  overlayBuildFailures,
  onOverlayBuildRetried,
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

              // Every stage that holds drawable objects gets the same card:
              // a checkbox, a count, a colour and an opacity. It used to be
              // COMPLETED only -- a state reached exclusively by pressing
              // "Mark Image Done" on another screen -- so the result of a
              // finished run had no control on it at all.
              if (segmentationHasResults(stage)) {
                const finished = isCompleted(stage);
                return (
                  <div
                    key={overlay.segmentation.id}
                    className={
                      finished
                        ? "overlay-item"
                        : "overlay-item overlay-item-actionable"
                    }
                  >
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
                      {onEditSegmentation && finished && (
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
                    {!finished && (
                      <div className="overlay-status-label overlay-status-actionable">
                        {getStageLabel(stage)}
                      </div>
                    )}
                    <div className="overlay-count">
                      {getTotalCount(overlay.segmentation.segment_counts) ?? 0}{" "}
                      objects
                    </div>
                    {/* A finished run leaves this stage whether it found two
                        hundred objects or none, so the count alone can read as
                        success over an empty overlay. `run_notice` is the
                        server's finding about exactly that case. */}
                    {overlay.segmentation.run_notice && (
                      <div className="overlay-status-note">
                        {overlay.segmentation.run_notice.message}
                      </div>
                    )}
                    <OverlayStatusError
                      stage={stage}
                      error={overlay.segmentation.status_error}
                    />
                    {/* The run is fine; the picture is not. Above the colour
                        and opacity controls, because a user who cannot see
                        their objects reaches for those first and they will not
                        help. */}
                    {overlayBuildFailures?.[overlay.segmentation.id] && (
                      <OverlayBuildFailureNotice
                        manifest={overlayBuildFailures[overlay.segmentation.id]}
                        segmentationId={overlay.segmentation.id}
                        onRetried={() =>
                          onOverlayBuildRetried?.(overlay.segmentation.id)
                        }
                      />
                    )}
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
                    {/* The run has finished but nobody has looked at it yet, so
                        the call to action stays what it was: go and check the
                        objects. On a COMPLETED card the ✎ glyph in the header
                        does the same job more quietly. */}
                    {!finished && onEditSegmentation && (
                      <button
                        type="button"
                        className="overlay-edit-button-prominent"
                        onClick={() => onEditSegmentation(overlay.segmentation.id)}
                      >
                        Edit / Label
                      </button>
                    )}
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
                    <OverlayStatusError
                      stage={stage}
                      error={overlay.segmentation.status_error}
                    />
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

              // FAILED, and any stage this build does not know about: nothing
              // drawable, so no overlay controls -- only what happened and the
              // two ways out.
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
                  <OverlayStatusError
                    stage={stage}
                    error={overlay.segmentation.status_error}
                  />
                  {/* The server's finding about a run whose stage does not tell
                      the whole story -- it found nothing, or a re-run added
                      nothing. The labeling header carries the full list of what
                      to check; this only says which case it is. */}
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
