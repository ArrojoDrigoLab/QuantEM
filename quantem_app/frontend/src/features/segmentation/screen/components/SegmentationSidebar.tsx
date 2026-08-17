import type { ComponentProps, ReactNode } from "react";
import {
  ErRoiControls,
  type ErRoiSection,
} from "@/features/segmentation/components/ErRoiControls";
import { WorkflowModeToolbar } from "@/features/segmentation/components/WorkflowModeToolbar";
import {
  TissueWorkflowControls,
  type TissueWorkflowControlsProps,
} from "@/features/segmentation/components/TissueWorkflowControls";
import type {
  WorkflowMode,
} from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import type { HoverActionMode, GroupHoverActionMode } from "@/hooks/useHoverSelection";
import type { CorrectionTool } from "@/shared/types/segmentation";
import {
  OVERLAY_DISPLAY_LABELS,
  type FailedOverlayBundle,
} from "@/features/segmentation/screen/hooks/overlay/useOverlayManifestState";
// Imported across features on purpose. The viewer and the labeling screen fail
// the same way for the same reason, and finding V4 was the labeling screen
// saying *nothing* while the viewer said everything -- two renderers of one
// message is how that gap reopens.
import { OverlayBuildFailureNotice } from "@/features/viewer/components/OverlayBuildFailureNotice";
import { IncludeLevelDial } from "@/features/segmentation/components/threshold";

interface TissueSection extends TissueWorkflowControlsProps {
  enabled: boolean;
}

interface ReviewSection {
  workflowMode: WorkflowMode;
  reviewPhase: "model" | "correction";
  correctionTool: CorrectionTool;
  hoverActionMode: HoverActionMode;
  drawBrushSize: number;
  draftOperation: "include" | "exclude";
  hasDrawStrokes: boolean;
  supportsPointFeedback: boolean;
  isErSegmentation: boolean;
  canApplyGroupAction: boolean;
  /** ER polygon tool: a draft polygon is in progress (enables Close/Clear). */
  polygonHasDraft: boolean;
  /** ER polygon tool: the draft can be closed into a filled object. */
  polygonCanClose: boolean;
  onReviewPhaseChange: (phase: "model" | "correction") => void;
  onCorrectionToolChange: (tool: CorrectionTool) => void;
  onHoverActionModeChange: (mode: HoverActionMode) => void;
  onDrawBrushSizeChange: (size: number) => void;
  onDraftOperationChange: (operation: "include" | "exclude") => void;
  onClearDrawing: () => void;
  onConfirmShape: () => void;
  /** ER polygon tool: close the draft and commit it as a filled ER object. */
  onClosePolygon: () => void;
  onApplyGroupAction: (mode: GroupHoverActionMode) => void;
  /**
   * Rendered into the toolbar's `extraModes` slot -- controls that change what
   * a click on the image means. Box-to-object (`features/sam`) is the first
   * user; the slot exists so a package adds a tool from its own file instead of
   * opening `WorkflowModeToolbar.tsx`.
   */
  extraModes?: ReactNode;
  /** The toolbar's `extraTools` slot: controls acting on the current selection. */
  extraTools?: ReactNode;
}

interface LayersSection {
  /** The named-model preview bundle is being rebuilt. */
  modelOverlayUpdating: boolean;
  /** The source-less confirmed-display bundle is being rebuilt. */
  confirmedOverlayUpdating: boolean;
  /**
   * Every bundle whose build the server has given up on, each tagged with the
   * display it belongs to. A list rather than a flag plus a manifest: the two
   * bundles fail independently, and one card per failure is the only way the
   * second one gets said at all and the only way each retry button points at
   * the bundle its card describes.
   */
  failedOverlays: FailedOverlayBundle[];
  /** Needed to ask for the build again; null before a segmentation is chosen. */
  overlaySegmentationId: string | null;
  /** Restart polling and reread the manifest once a retry is accepted. */
  onOverlayBuildRetried: () => void;
}

interface ViewSection {
  leftNavigateMode: boolean;
  onLeftNavigateModeChange: (value: boolean) => void;
  showConfirmedPanel: boolean;
  onShowConfirmedPanelChange: (value: boolean) => void;
  isGroupActionMode: boolean;
  activeGroupActionVerb: "confirm" | "reject";
  groupSelectionCount: number;
}

type IncludeLevelSection = ComponentProps<typeof IncludeLevelDial>;

export interface SegmentationSidebarProps {
  /** Manual tissue-mask toolbar (brush/polygon with Include/Exclude). */
  tissue: TissueSection;
  review: ReviewSection;
  layers: LayersSection;
  view: ViewSection;
  /** Rectangular ROI controls (1024² default + per-organelle "mark ROI done"). */
  erRoi?: ErRoiSection;
  /**
   * The include-level dial. Absent until a segmentation is selected, and
   * the dial itself greys out when there is no stored probability map to
   * re-read, so it never offers a move it cannot make.
   */
  includeLevel?: IncludeLevelSection;
}

export function SegmentationSidebar({
  tissue,
  review,
  layers,
  view,
  erRoi,
  includeLevel,
}: SegmentationSidebarProps) {
  const { enabled: isTissueSegmentation, ...tissueControls } = tissue;
  const {
    workflowMode,
    reviewPhase,
    correctionTool,
    hoverActionMode,
    drawBrushSize,
    draftOperation,
    hasDrawStrokes,
    supportsPointFeedback,
    isErSegmentation,
    canApplyGroupAction,
    polygonHasDraft,
    polygonCanClose,
    onReviewPhaseChange,
    onCorrectionToolChange,
    onHoverActionModeChange,
    onDrawBrushSizeChange,
    onDraftOperationChange,
    onClearDrawing,
    onConfirmShape,
    onClosePolygon,
    onApplyGroupAction,
    extraModes,
    extraTools,
  } = review;
  const {
    modelOverlayUpdating,
    confirmedOverlayUpdating,
    failedOverlays,
    overlaySegmentationId,
    onOverlayBuildRetried,
  } = layers;
  const {
    leftNavigateMode,
    onLeftNavigateModeChange,
    showConfirmedPanel,
    onShowConfirmedPanelChange,
    isGroupActionMode,
    activeGroupActionVerb,
    groupSelectionCount,
  } = view;
  return (
    <aside className="labeling-sidebar">
        <section className="labeling-sidebar-section">
          <h3>Labeling View</h3>
          {erRoi && <ErRoiControls {...erRoi} />}
          {includeLevel ? <IncludeLevelDial {...includeLevel} /> : null}
          {isTissueSegmentation ? (
            <TissueWorkflowControls {...tissueControls} />
          ) : (
            <WorkflowModeToolbar
              workflowMode={workflowMode}
              reviewPhase={reviewPhase}
              correctionTool={correctionTool}
              hoverActionMode={hoverActionMode}
              drawBrushSize={drawBrushSize}
              draftOperation={draftOperation}
              hasDrawStrokes={hasDrawStrokes}
              onReviewPhaseChange={onReviewPhaseChange}
              onCorrectionToolChange={onCorrectionToolChange}
              onHoverActionModeChange={onHoverActionModeChange}
              onDrawBrushSizeChange={onDrawBrushSizeChange}
              onDraftOperationChange={onDraftOperationChange}
              onClearDrawing={onClearDrawing}
              onConfirmShape={onConfirmShape}
              polygonHasDraft={polygonHasDraft}
              polygonCanClose={polygonCanClose}
              onClosePolygon={onClosePolygon}
              showGroupConfirm={supportsPointFeedback}
              isErSegmentation={isErSegmentation}
              canApplyGroupAction={canApplyGroupAction}
              onApplyGroupAction={onApplyGroupAction}
              extraModes={extraModes}
              extraTools={extraTools}
            />
          )}
        </section>
        <section className="labeling-sidebar-section">
          <label className="view-confirmed-toggle" htmlFor="navigate-toggle">
            <input
              id="navigate-toggle"
              type="checkbox"
              checked={leftNavigateMode}
              onChange={(event) => onLeftNavigateModeChange(event.target.checked)}
            />
            Navigate (A)
          </label>
          {leftNavigateMode && (
            // Actionable, not just descriptive: choosing a tool now leaves
            // Navigate automatically, but the A shortcut can put it back
            // mid-session and the way out has to be one click from the notice.
            <div className="group-confirm-hint">
              <span>
                Navigate mode is on: clicks and drags pan the image instead of
                labeling.
              </span>
              <button
                type="button"
                className="navigate-hint-action"
                onClick={() => onLeftNavigateModeChange(false)}
              >
                Turn off Navigate
              </button>
            </div>
          )}
          <label className="view-confirmed-toggle" htmlFor="view-confirmed-toggle">
            <input
              id="view-confirmed-toggle"
              type="checkbox"
              checked={showConfirmedPanel}
              onChange={(event) => onShowConfirmedPanelChange(event.target.checked)}
            />
            View confirmed
          </label>
          {isGroupActionMode && !leftNavigateMode && (
            <div className="group-confirm-hint">
              Drag a box in labeling view, then press Enter, Space, or the active
              group button to {activeGroupActionVerb}. Delete clears the selection.
            </div>
          )}
          {isGroupActionMode && !leftNavigateMode && groupSelectionCount > 0 && (
            <div className="group-confirm-count">Selected: {groupSelectionCount}</div>
          )}
          {workflowMode === "review" &&
            (modelOverlayUpdating || confirmedOverlayUpdating) && (
              <div className="group-confirm-hint" role="status">
                {modelOverlayUpdating && confirmedOverlayUpdating
                  ? "Model preview and confirmed displays are updating. Saved objects remain ready for analysis."
                  : confirmedOverlayUpdating
                    ? "Confirmed display is updating. Saved objects remain ready for analysis."
                    : "Model preview display is updating."}
              </div>
            )}
          {/*
            Finding V4. Until this block existed the labeling screen said
            nothing at all about a failed overlay while the review canvas
            quietly went on drawing a raster that no longer matched the objects.
            It is shown in every workflow mode, unlike the updating line above:
            a failed build is not a passing state, and leaving review does not
            make it go away.

            The line above is *not* mutually exclusive with this one, which it
            was while both were derived from a single manifest. The model
            preview and the confirmed display are separate bundles with
            separate rebuild jobs, so one can be failing while the other is
            still building -- which is why the line above names the display it
            is about and each card below names its own. Two unnamed messages
            about "the overlay" would read as contradicting each other.
          */}
          {overlaySegmentationId &&
            failedOverlays.map(({ role, manifest }) => (
              <OverlayBuildFailureNotice
                key={role}
                manifest={manifest}
                segmentationId={overlaySegmentationId}
                displayLabel={OVERLAY_DISPLAY_LABELS[role]}
                onRetried={onOverlayBuildRetried}
              />
            ))}
        </section>
    </aside>
  );
}
