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
import type {
  CorrectionTool,
  SegmentationOverlayManifest,
} from "@/shared/types/segmentation";
import type { LeftPanelLayerStyles } from "@/features/segmentation/overlays/segments";
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
  usesRasterReviewOverlay: boolean;
  showCandidateBorders: boolean;
  onShowCandidateBordersChange: (value: boolean) => void;
  showConfirmedBorders: boolean;
  onShowConfirmedBordersChange: (value: boolean) => void;
  leftPanelLayerStyles: LeftPanelLayerStyles;
  onCandidateStrokeWidthChange: (value: number) => void;
  onCandidateFillOpacityChange: (value: number) => void;
  onConfirmedStrokeWidthChange: (value: number) => void;
  onConfirmedFillOpacityChange: (value: number) => void;
  overlayUpdating: boolean;
  /** True only for a build the server has given up on. */
  overlayBuildFailed: boolean;
  /** The failed manifest, carrying the reason and the two revisions. */
  overlayManifest: SegmentationOverlayManifest | null;
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
  /** Manual tissue-mask toolbar (brush + polygon + exclude-polygon). */
  tissue: TissueSection;
  review: ReviewSection;
  layers: LayersSection;
  view: ViewSection;
  /** ER-only ROI controls (2048² creation + per-organelle "mark ROI done"). */
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
    onClearDrawing,
    onConfirmShape,
    onClosePolygon,
    onApplyGroupAction,
    extraModes,
    extraTools,
  } = review;
  const {
    usesRasterReviewOverlay,
    showCandidateBorders,
    onShowCandidateBordersChange,
    showConfirmedBorders,
    onShowConfirmedBordersChange,
    leftPanelLayerStyles,
    onCandidateStrokeWidthChange,
    onCandidateFillOpacityChange,
    onConfirmedStrokeWidthChange,
    onConfirmedFillOpacityChange,
    overlayUpdating,
    overlayBuildFailed,
    overlayManifest,
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
              hasDrawStrokes={hasDrawStrokes}
              onReviewPhaseChange={onReviewPhaseChange}
              onCorrectionToolChange={onCorrectionToolChange}
              onHoverActionModeChange={onHoverActionModeChange}
              onDrawBrushSizeChange={onDrawBrushSizeChange}
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
          {workflowMode === "review" && overlayUpdating && (
            <div className="group-confirm-hint">Overlay updating.</div>
          )}
          {/*
            Finding V4. The line above is mutually exclusive with this block --
            `overlayUpdating` is false for a build the server has given up on --
            and until now that meant the labeling screen said nothing at all
            about a failed overlay while the review canvas quietly went on
            drawing a raster that no longer matched the objects. It is shown in
            every workflow mode, unlike "Overlay updating.": a failed build is
            not a passing state, and leaving review does not make it go away.
          */}
          {overlayBuildFailed && overlayManifest && overlaySegmentationId && (
            <OverlayBuildFailureNotice
              manifest={overlayManifest}
              segmentationId={overlaySegmentationId}
              onRetried={onOverlayBuildRetried}
            />
          )}
        </section>
        <details className="labeling-sidebar-section labeling-sidebar-collapsible" open>
          <summary className="labeling-sidebar-collapsible-title">Layers</summary>
          <div className="layer-style-controls">
            <section className="layer-style-group">
              <h4>Candidates</h4>
              {usesRasterReviewOverlay ? (
                <div className="layer-style-control">
                  <label
                    className="view-confirmed-toggle"
                    htmlFor="candidate-borders-toggle"
                  >
                    <input
                      id="candidate-borders-toggle"
                      type="checkbox"
                      checked={showCandidateBorders}
                      onChange={(event) =>
                        onShowCandidateBordersChange(event.target.checked)
                      }
                    />
                    Borders
                  </label>
                </div>
              ) : (
                <div className="layer-style-control">
                  <label htmlFor="candidate-border-thickness">Border thickness</label>
                  <div className="layer-style-slider-row">
                    <input
                      id="candidate-border-thickness"
                      type="range"
                      min={0.5}
                      max={8}
                      step={0.5}
                      value={leftPanelLayerStyles.candidateStrokeWidth}
                      onChange={(event) =>
                        onCandidateStrokeWidthChange(Number(event.target.value))
                      }
                    />
                    <span>{leftPanelLayerStyles.candidateStrokeWidth.toFixed(1)}px</span>
                  </div>
                </div>
              )}
              <div className="layer-style-control">
                <label htmlFor="candidate-fill-opacity">Fill opacity</label>
                <div className="layer-style-slider-row">
                  <input
                    id="candidate-fill-opacity"
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={leftPanelLayerStyles.candidateFillOpacity}
                    onChange={(event) =>
                      onCandidateFillOpacityChange(Number(event.target.value))
                    }
                  />
                  <span>
                    {(leftPanelLayerStyles.candidateFillOpacity * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </section>
            <section className="layer-style-group">
              <h4>Confirmed</h4>
              {usesRasterReviewOverlay ? (
                <div className="layer-style-control">
                  <label
                    className="view-confirmed-toggle"
                    htmlFor="confirmed-borders-toggle"
                  >
                    <input
                      id="confirmed-borders-toggle"
                      type="checkbox"
                      checked={showConfirmedBorders}
                      onChange={(event) =>
                        onShowConfirmedBordersChange(event.target.checked)
                      }
                    />
                    Borders
                  </label>
                </div>
              ) : (
                <div className="layer-style-control">
                  <label htmlFor="confirmed-border-thickness">Border thickness</label>
                  <div className="layer-style-slider-row">
                    <input
                      id="confirmed-border-thickness"
                      type="range"
                      min={0.5}
                      max={8}
                      step={0.5}
                      value={leftPanelLayerStyles.confirmedStrokeWidth}
                      onChange={(event) =>
                        onConfirmedStrokeWidthChange(Number(event.target.value))
                      }
                    />
                    <span>{leftPanelLayerStyles.confirmedStrokeWidth.toFixed(1)}px</span>
                  </div>
                </div>
              )}
              <div className="layer-style-control">
                <label htmlFor="confirmed-fill-opacity">Fill opacity</label>
                <div className="layer-style-slider-row">
                  <input
                    id="confirmed-fill-opacity"
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={leftPanelLayerStyles.confirmedFillOpacity}
                    onChange={(event) =>
                      onConfirmedFillOpacityChange(Number(event.target.value))
                    }
                  />
                  <span>
                    {(leftPanelLayerStyles.confirmedFillOpacity * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </section>
          </div>
        </details>
    </aside>
  );
}
