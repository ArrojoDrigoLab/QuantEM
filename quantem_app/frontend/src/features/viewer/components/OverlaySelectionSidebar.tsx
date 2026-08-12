import { useState, type ReactNode } from "react";
import type { ImageSegmentation, StatusStage } from "@/shared/types";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";
import { segmentationHasResults } from "@/shared/constants/segmentation";
import {
  segmentationDisplayName,
  segmentationShortName,
} from "@/shared/segmentationNames";
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

function OverlayStatusError({
  stage,
  error,
}: {
  stage: StatusStage;
  error?: string | null;
}) {
  const reason = error?.trim();
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
  onDeleteSegmentation?: (segmentationId: string) => void;
  overlayBuildFailures?: Record<string, SegmentationOverlayManifest>;
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
  const [expandedSegmentationId, setExpandedSegmentationId] = useState<string | null>(
    null
  );

  const getTotalCount = (counts?: Record<string, number>) => {
    if (!counts) return null;
    return Object.values(counts).reduce((sum, value) => sum + (value ?? 0), 0);
  };

  return (
    <aside className="overlay-sidebar">
      <h3>Segmentations</h3>
      <div className="overlay-sidebar-content">
        {disabled ? (
          <div className="overlay-sidebar-disabled-mask">
            <span className="overlay-sidebar-disabled-message">
              {statusMessage || "Overlays unavailable"}
            </span>
          </div>
        ) : null}
        <section aria-labelledby="existing-segmentations-heading">
          <h4 id="existing-segmentations-heading">Existing Segmentations</h4>
          {overlays.length === 0 ? (
            <div className="overlay-empty">No segmentations available.</div>
          ) : (
            <div className="overlay-list">
              {overlays.map((overlay) => {
                const segmentation = overlay.segmentation;
                const stage = segmentation.status_stage;
                const canDraw = segmentationHasResults(stage);
                const expanded = expandedSegmentationId === segmentation.id;
                const fullName = segmentationDisplayName(segmentation);
                const shortName = segmentationShortName(segmentation);
                const finished = isCompleted(stage);

                return (
                  <div
                    key={segmentation.id}
                    className={
                      canDraw
                        ? "overlay-item"
                        : "overlay-item overlay-item-actionable"
                    }
                  >
                    <div className="overlay-header">
                      <label className="overlay-toggle" title={fullName}>
                        <input
                          type="checkbox"
                          checked={canDraw && overlay.enabled}
                          disabled={!canDraw}
                          aria-label={`Show ${fullName}`}
                          onChange={() => onToggle(segmentation.id)}
                        />
                        <span className="overlay-name">{shortName}</span>
                      </label>
                      <button
                        type="button"
                        className="overlay-open-button"
                        onClick={() => onEditSegmentation?.(segmentation.id)}
                        disabled={!onEditSegmentation}
                        title="Edit Labels"
                        aria-label={`Edit Labels for ${fullName}`}
                      >
                        <OpenIcon />
                      </button>
                      <button
                        type="button"
                        className="overlay-accordion-button"
                        onClick={() =>
                          setExpandedSegmentationId((current) =>
                            current === segmentation.id ? null : segmentation.id
                          )
                        }
                        aria-expanded={expanded}
                        aria-label={`${expanded ? "Collapse" : "Expand"} ${fullName}`}
                      >
                        <ChevronIcon expanded={expanded} />
                      </button>
                    </div>
                    {expanded ? (
                      <div className="overlay-accordion-content">
                        {!finished ? (
                          <div
                            className={`overlay-status-label ${
                              stage === "FAILED"
                                ? "overlay-status-failed"
                                : isProcessing(stage)
                                  ? "overlay-status-processing"
                                  : "overlay-status-actionable"
                            }`}
                          >
                            {getStageLabel(stage)}
                          </div>
                        ) : null}
                        {canDraw ? (
                          <div className="overlay-count">
                            {getTotalCount(segmentation.segment_counts) ?? 0} objects
                          </div>
                        ) : null}
                        {isProcessing(stage) && segmentation.status_progress > 0 ? (
                          <div className="overlay-progress-bar">
                            <div
                              className="overlay-progress-fill"
                              style={{
                                width: `${Math.round(segmentation.status_progress)}%`,
                              }}
                            />
                          </div>
                        ) : null}
                        {segmentation.run_notice ? (
                          <div className="overlay-status-note">
                            {segmentation.run_notice.message}
                          </div>
                        ) : null}
                        <OverlayStatusError
                          stage={stage}
                          error={segmentation.status_error}
                        />
                        {overlayBuildFailures?.[segmentation.id] ? (
                          <OverlayBuildFailureNotice
                            manifest={overlayBuildFailures[segmentation.id]}
                            segmentationId={segmentation.id}
                            onRetried={() => onOverlayBuildRetried?.(segmentation.id)}
                          />
                        ) : null}
                        {canDraw ? (
                          <div className="overlay-controls">
                            <label className="overlay-color">
                              <span>Color</span>
                              <input
                                type="color"
                                value={overlay.color}
                                onChange={(event) =>
                                  onColorChange(segmentation.id, event.target.value)
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
                                    segmentation.id,
                                    Number(event.target.value)
                                  )
                                }
                              />
                              <span className="overlay-opacity-value">
                                {(overlay.opacity * 100).toFixed(0)}%
                              </span>
                            </label>
                          </div>
                        ) : null}
                        {onEditSegmentation ? (
                          <button
                            type="button"
                            className="overlay-edit-button-prominent"
                            onClick={() => onEditSegmentation(segmentation.id)}
                          >
                            Edit Labels
                          </button>
                        ) : null}
                        {onDeleteSegmentation ? (
                          <button
                            type="button"
                            className="overlay-delete-button-text"
                            onClick={() => onDeleteSegmentation(segmentation.id)}
                          >
                            Delete
                          </button>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          )}
        </section>
        {createPanel ? <div className="overlay-create-section">{createPanel}</div> : null}
      </div>
    </aside>
  );
}

function OpenIcon() {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      viewBox="0 0 24 24"
    >
      <path d="M14 3h7v7" />
      <path d="m21 3-9 9" />
      <path d="M19 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h6" />
    </svg>
  );
}

function ChevronIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      viewBox="0 0 24 24"
    >
      <path d={expanded ? "m6 15 6-6 6 6" : "m9 18 6-6-6-6"} />
    </svg>
  );
}
