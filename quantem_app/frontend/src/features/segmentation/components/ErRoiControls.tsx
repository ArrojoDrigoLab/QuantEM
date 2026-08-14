import { useMemo, useState } from "react";

import "./ErRoiControls.css";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { ROI_REVIEWED_LABEL, ROI_REVIEWED_TOOLTIP } from "@/shared/constants/confirmedArea";
import type { SegmentationRoi } from "@/shared/types/segmentation";

export interface ErRoiSection {
  placementActive: boolean;
  pendingRoiActive: boolean;
  relocatingRoiId: string | null;
  confirming: boolean;
  rois: SegmentationRoi[];
  activeRoiId: string | null;
  markingRoiId: string | null;
  deletingRoiId: string | null;
  activatingRoiId: string | null;
  testingRoiId: string | null;
  testDisabled?: boolean;
  testDisabledReason?: string;
  onStartPlacement: () => void;
  onEditRoi: (roi: SegmentationRoi) => void;
  onCancelPlacement: () => void;
  onConfirmRoi: () => void;
  onMarkRoiDone: (roiId: string, done: boolean) => void;
  onDeleteRoi: (roiId: string) => void;
  onActivateRoi: (roiId: string) => void;
  onTestRoi: (roiId: string) => void;
}

function roiDetails(roi: SegmentationRoi): string {
  return `${roi.width}×${roi.height} px at (${roi.x}, ${roi.y})`;
}

export function ErRoiControls({
  placementActive,
  pendingRoiActive,
  relocatingRoiId,
  confirming,
  rois,
  activeRoiId,
  markingRoiId,
  deletingRoiId,
  activatingRoiId,
  testingRoiId,
  testDisabled = false,
  testDisabledReason,
  onStartPlacement,
  onEditRoi,
  onCancelPlacement,
  onConfirmRoi,
  onMarkRoiDone,
  onDeleteRoi,
  onActivateRoi,
  onTestRoi,
}: ErRoiSection) {
  const [pendingDeleteRoi, setPendingDeleteRoi] = useState<SegmentationRoi | null>(null);
  const orderedRois = useMemo(
    () =>
      [...rois].sort(
        (left, right) =>
          left.created_at.localeCompare(right.created_at) || left.id.localeCompare(right.id)
      ),
    [rois]
  );
  const editing = relocatingRoiId !== null;
  const primaryLabel = pendingRoiActive
    ? confirming
      ? editing
        ? "Saving..."
        : "Creating..."
      : editing
        ? "Save Area"
        : "Create"
    : placementActive
      ? "Cancel"
      : "Add New";
  const onPrimaryClick = pendingRoiActive
    ? onConfirmRoi
    : placementActive
      ? onCancelPlacement
      : onStartPlacement;

  return (
    <div className="er-roi-controls">
      <div className="er-roi-heading">
        <h4>ROI</h4>
        <button
          type="button"
          className={`er-roi-heading-button ${placementActive ? "active" : ""}`}
          onClick={onPrimaryClick}
          disabled={confirming}
        >
          {primaryLabel}
        </button>
      </div>
      {placementActive && !pendingRoiActive && (
        <p className="er-roi-hint">Click to place the new ROI.</p>
      )}
      {editing && pendingRoiActive && (
        <p className="er-roi-hint">
          Drag an edge or corner to resize. Drag inside the ROI to move it.
        </p>
      )}
      {orderedRois.length > 0 && (
        <ul className="er-roi-list" aria-label="ROIs">
          {orderedRois.map((roi, index) => {
            const done = Boolean(roi.completed_for_segmentation);
            const isActive = roi.id === activeRoiId;
            const title = `ROI ${index + 1}: ${roi.width}x${roi.height} px`;
            const details = roiDetails(roi);
            return (
              <li key={roi.id} className={`er-roi-item ${isActive ? "active" : ""}`}>
                <div className="er-roi-item-summary">
                  <span className="er-roi-item-label" title={details}>
                    {title}
                  </span>
                  <button
                    type="button"
                    className="er-roi-edit-button"
                    disabled={done || placementActive || confirming}
                    aria-pressed={relocatingRoiId === roi.id}
                    title={done ? "Reopen this ROI before editing its area." : undefined}
                    onClick={() => onEditRoi(roi)}
                  >
                    {relocatingRoiId === roi.id ? "Editing" : "Edit Area"}
                  </button>
                </div>
                <div className="er-roi-item-actions">
                  <button
                    type="button"
                    className="er-roi-action-button"
                    disabled={activatingRoiId === roi.id}
                    onClick={() => onActivateRoi(roi.id)}
                  >
                    {activatingRoiId === roi.id ? "Opening..." : "Open"}
                  </button>
                  <button
                    type="button"
                    className="er-roi-action-button er-roi-test-button"
                    disabled={
                      done ||
                      testDisabled ||
                      testingRoiId !== null ||
                      placementActive ||
                      confirming
                    }
                    title={
                      done
                        ? "This ROI is marked done. Reopen it before testing the model."
                        : testDisabledReason ?? "Run the selected model on this ROI"
                    }
                    onClick={() => onTestRoi(roi.id)}
                  >
                    {testingRoiId === roi.id ? "Testing..." : "Test"}
                  </button>
                  <label className="er-roi-done-toggle" title={ROI_REVIEWED_TOOLTIP}>
                    <input
                      aria-label={`Mark ${title} done`}
                      type="checkbox"
                      checked={done}
                      disabled={markingRoiId === roi.id}
                      onChange={(event) => onMarkRoiDone(roi.id, event.target.checked)}
                    />
                    {ROI_REVIEWED_LABEL}
                  </label>
                  <button
                    type="button"
                    className="er-roi-delete-button"
                    aria-label={`Delete ${title}`}
                    title="Delete ROI"
                    disabled={deletingRoiId === roi.id}
                    onClick={() => setPendingDeleteRoi(roi)}
                  >
                    <svg
                      aria-hidden="true"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M3 6h18" />
                      <path d="M8 6V4h8v2" />
                      <path d="M19 6l-1 14H6L5 6" />
                      <path d="M10 11v5" />
                      <path d="M14 11v5" />
                    </svg>
                  </button>
                  {isActive && <span className="er-roi-active-badge">active</span>}
                </div>
              </li>
            );
          })}
        </ul>
      )}
      <ConfirmDialog
        isOpen={pendingDeleteRoi !== null}
        title="Delete ROI"
        message={
          pendingDeleteRoi
            ? `Delete ${roiDetails(pendingDeleteRoi)}? It will be removed from every segmentation on this image. Existing segment objects are kept.`
            : ""
        }
        confirmText={
          pendingDeleteRoi && deletingRoiId === pendingDeleteRoi.id
            ? "Deleting..."
            : "Delete ROI"
        }
        cancelText="Cancel"
        onConfirm={() => {
          if (pendingDeleteRoi) {
            onDeleteRoi(pendingDeleteRoi.id);
            setPendingDeleteRoi(null);
          }
        }}
        onCancel={() => setPendingDeleteRoi(null)}
      />
    </div>
  );
}
