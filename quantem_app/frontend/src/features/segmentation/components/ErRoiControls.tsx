/**
 * ER ROI windows, and the per-organelle tick box beside each one.
 *
 * That tick box used to read **Done (ER)**, which was the fourth thing in this
 * application called "complete" and the only one that collided with a
 * precondition. It writes `RoiSegmentationStatus.is_complete`, which nothing
 * outside this list reads. What "Adapt a model" requires is a `CompletedROI`
 * polygon, and only the Confirmed area tool makes one -- so a user who ticked
 * Done (ER) and opened the wizard was told *"No completed ROI on this image.
 * Mark the area you have finished annotating as complete"*, which is a
 * word-for-word instruction to repeat what they had just done.
 *
 * The flag stays -- with twenty 2048² windows on an image, a record of which
 * ones you have been through is worth keeping -- under a word the wizard does
 * not use, with the distinction written next to it rather than left to be
 * discovered on the step that refuses to proceed.
 */

import { useState } from "react";
import "./ErRoiControls.css";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import {
  CONFIRMED_AREA_HOW_TO,
  CONFIRMED_AREA_LABEL,
  ROI_REVIEWED_EXPLANATION,
  ROI_REVIEWED_LABEL,
  ROI_REVIEWED_TOOLTIP,
} from "@/shared/constants/confirmedArea";
import type { SegmentationRoi } from "@/shared/types/segmentation";

export interface ErRoiSection {
  placementActive: boolean;
  pendingRoiActive: boolean;
  confirming: boolean;
  rois: SegmentationRoi[];
  activeRoiId: string | null;
  markingRoiId: string | null;
  deletingRoiId: string | null;
  onStartPlacement: () => void;
  onCancelPlacement: () => void;
  onConfirmRoi: () => void;
  onMarkRoiDone: (roiId: string, done: boolean) => void;
  onDeleteRoi: (roiId: string) => void;
  /** Double-clicking a ROI row activates it and fits the viewer to it. */
  onActivateRoi: (roiId: string) => void;
}

function roiLabel(roi: SegmentationRoi): string {
  return `${roi.width}×${roi.height} at (${roi.x}, ${roi.y})`;
}

export function ErRoiControls({
  placementActive,
  pendingRoiActive,
  confirming,
  rois,
  activeRoiId,
  markingRoiId,
  deletingRoiId,
  onStartPlacement,
  onCancelPlacement,
  onConfirmRoi,
  onMarkRoiDone,
  onDeleteRoi,
  onActivateRoi,
}: ErRoiSection) {
  const [pendingDeleteRoi, setPendingDeleteRoi] = useState<SegmentationRoi | null>(null);

  const primaryLabel = pendingRoiActive
    ? confirming
      ? "Creating ROI..."
      : "Confirm 2048² ROI"
    : placementActive
      ? "Cancel placement"
      : "New 2048² ROI";
  const onPrimaryClick = pendingRoiActive
    ? onConfirmRoi
    : placementActive
      ? onCancelPlacement
      : onStartPlacement;

  return (
    <div className="er-roi-controls">
      <button
        type="button"
        className={`er-roi-button ${placementActive ? "active" : ""}`}
        onClick={onPrimaryClick}
        disabled={confirming}
      >
        {primaryLabel}
      </button>
      {placementActive && !pendingRoiActive && (
        <p className="er-roi-hint">
          Click a point in the image to place a 2048&times;2048 ROI.
        </p>
      )}
      {rois.length > 0 && (
        <ul className="er-roi-list">
          {rois.map((roi) => {
            const done = Boolean(roi.completed_for_segmentation);
            const isActive = roi.id === activeRoiId;
            return (
              <li
                key={roi.id}
                className={`er-roi-item ${isActive ? "active" : ""}`}
                title="Double-click to switch to this ROI"
                onDoubleClick={() => onActivateRoi(roi.id)}
              >
                <span className="er-roi-item-label">
                  {roi.width}&times;{roi.height} @ ({roi.x}, {roi.y})
                  {isActive && <span className="er-roi-active-badge">active</span>}
                </span>
                <div
                  className="er-roi-item-actions"
                  onDoubleClick={(event) => event.stopPropagation()}
                >
                  <label
                    className="er-roi-done-toggle"
                    title={ROI_REVIEWED_TOOLTIP}
                  >
                    <input
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
                    aria-label={`Delete ROI ${roiLabel(roi)}`}
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
                </div>
              </li>
            );
          })}
        </ul>
      )}
      {rois.length > 0 && (
        // Under the tick boxes it is about, and shown whether or not any are
        // ticked: the misreading happens at the moment of ticking, and the
        // wizard's refusal arrives on a different screen much later.
        <p className="er-roi-hint er-roi-reviewed-note" role="note">
          {ROI_REVIEWED_EXPLANATION} {CONFIRMED_AREA_HOW_TO}
        </p>
      )}
      <ConfirmDialog
        isOpen={pendingDeleteRoi !== null}
        title="Delete ROI"
        message={
          pendingDeleteRoi
            ? `Delete ROI ${roiLabel(pendingDeleteRoi)}? This removes the ROI window and its ` +
              `"${ROI_REVIEWED_LABEL}" mark for every organelle. Segments already labeled inside it ` +
              `are not deleted, and any ${CONFIRMED_AREA_LABEL.toLowerCase()} you drew is a separate ` +
              `shape and is not affected.`
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
