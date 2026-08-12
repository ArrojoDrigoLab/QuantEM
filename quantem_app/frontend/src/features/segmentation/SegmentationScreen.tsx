import { useCallback, useEffect, useMemo, useRef } from "react";
import { useDrawing } from "@/hooks/useDrawing";
import { useUncertainSegments } from "@/features/segmentation/hooks/useUncertainSegments";
import { SegmentationHeader } from "@/features/segmentation/components/SegmentationHeader";
import { SegmentationRightPanel } from "@/features/segmentation/components/SegmentationRightPanel";
import { SegmentationLeftPanel } from "@/features/segmentation/components/SegmentationLeftPanel";
import { CompletedRoiBackgroundNotice } from "@/features/segmentation/components/CompletedRoiBackgroundNotice";
import { SegmentationSidebar } from "@/features/segmentation/screen/components/SegmentationSidebar";
import { SegmentationJobBanner } from "@/features/segmentation/screen/components/SegmentationJobBanner";
import { SegmentationScreenState } from "@/features/segmentation/screen/components/SegmentationScreenState";
import { useSegmentationRouteState } from "@/features/segmentation/screen/hooks/useSegmentationRouteState";
import { useSegmentationOverlayState } from "@/features/segmentation/screen/hooks/useSegmentationOverlayState";
import { useSegmentationHoverQuery } from "@/features/segmentation/screen/hooks/useSegmentationHoverQuery";
import { useSegmentationProcessingState } from "@/features/segmentation/screen/hooks/useSegmentationProcessingState";
import { useSegmentationFeedback } from "@/features/segmentation/screen/hooks/useSegmentationFeedback";
import {
  isBackgroundCountWarning,
  useCompletedRoiWorkflow,
} from "@/features/segmentation/screen/hooks/useCompletedRoiWorkflow";
import { useErRoiWorkflow } from "@/features/segmentation/screen/hooks/useErRoiWorkflow";
import { useErPolygonWorkflow } from "@/features/segmentation/screen/hooks/useErPolygonWorkflow";
import { useConfirmedGeometrySubmission } from "@/features/segmentation/screen/hooks/useConfirmedGeometrySubmission";
import { useSegmentationReviewWorkflow } from "@/features/segmentation/screen/hooks/useSegmentationReviewWorkflow";
import { useSegmentationInteractionRouter } from "@/features/segmentation/screen/hooks/useSegmentationInteractionRouter";
import { useReviewSamBoxController } from "@/features/segmentation/screen/hooks/review/useReviewSamBoxController";
import { useSegmentationKeyboardShortcuts } from "@/features/segmentation/screen/hooks/useSegmentationKeyboardShortcuts";
import { useSegmentationScreenUiState } from "@/features/segmentation/screen/hooks/useSegmentationScreenUiState";
import { useRemoveAreaWorkflow } from "@/features/segmentation/screen/hooks/useRemoveAreaWorkflow";
import { useTissueLabeling } from "@/features/segmentation/screen/hooks/useTissueLabeling";
import { useSegmentationScreenViewModels } from "@/features/segmentation/screen/hooks/useSegmentationScreenViewModels";
import { useModelCatalogue } from "@/features/models/useModelCatalogue";
import {
  packIdForSourceModel,
  runnabilityForPackId,
} from "@/features/models/runnable";
import { appliedAdapterState } from "@/features/models/appliedAdapter";
import { scaleMismatchForPack } from "@/features/models/scaleMismatch";
import { clearSegmentationManualLabels } from "@/shared/api/segmentations/annotations";
import { resolvePixelSize } from "@/shared/pixelSize";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { useRestartBlocker } from "@/features/update/restartGuardHooks";
import { CONFIRMED_AREA_EXPLANATION } from "@/shared/constants/confirmedArea";
import type { Point } from "@/utils/geometry";
import "./SegmentationScreen.css";

export function SegmentationScreen() {
  const route = useSegmentationRouteState();
  const ui = useSegmentationScreenUiState({
    currentSegmentationId: route.currentSegmentationId,
  });
  const drawing = useDrawing();

  // Whether the model the run button would use can be loaded here. Without
  // this the button cheerfully queued a run that died on a missing encoder,
  // and the queue banner then replaced the error that explained it.
  const { catalogue: modelCatalogue } = useModelCatalogue();
  const modelRunnability = useMemo(
    () =>
      runnabilityForPackId(
        modelCatalogue,
        packIdForSourceModel(route.activeSourceModel)
      ),
    [modelCatalogue, route.activeSourceModel]
  );

  // Whether this segmentation has an adapted model applied, and whether the
  // selected source model means the run will actually go through it. The
  // labeling header is the only screen that starts a run, and it used to say
  // nothing about either.
  const appliedAdapter = useMemo(
    () =>
      appliedAdapterState(
        modelCatalogue,
        route.currentSegmentation?.id ?? null,
        route.activeSourceModel
      ),
    [modelCatalogue, route.currentSegmentation?.id, route.activeSourceModel]
  );

  // Whether starting a run here would put a pack that declares a working
  // resolution onto an image with no pixel size. The create-segmentation
  // dialog has asked about this for a while; the run button on this screen is
  // the other door into the identical inference pass and asked nothing.
  const runScaleMismatch = useMemo(
    () =>
      scaleMismatchForPack(
        modelCatalogue,
        packIdForSourceModel(route.activeSourceModel),
        route.image ? resolvePixelSize(route.image).valueNm : undefined
      ),
    [modelCatalogue, route.activeSourceModel, route.image]
  );

  const isPointInsideImageBounds = useCallback(
    (point: Point): boolean => {
      if (!route.image) return false;
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return false;
      return (
        point.x >= 0 &&
        point.y >= 0 &&
        point.x <= route.image.width &&
        point.y <= route.image.height
      );
    },
    [route.image]
  );

  const refetchLeftSegmentsRef = useRef<() => Promise<void>>(() => Promise.resolve());
  const refetchLeftSegmentsProxy = useCallback(() => refetchLeftSegmentsRef.current(), []);

  const overlay = useSegmentationOverlayState({
    currentSegmentationId: route.currentSegmentationId,
    activeSourceModel: route.activeSourceModel,
    segmentationInternalName: route.segmentationInternalName,
    refetchSegmentations: route.refetchSegmentations,
    refetchLeftSegments: refetchLeftSegmentsProxy,
    useSmoothedSegmentGeometry: route.useSmoothedSegmentGeometry,
  });
  const { optimistic: overlayOptimistic } = overlay;
  const { refresh: overlayRefresh } = overlay;

  const hover = useSegmentationHoverQuery({
    currentSegmentation: route.currentSegmentation,
    activeSourceModel: route.activeSourceModel,
  });

  const confirmedSubmission = useConfirmedGeometrySubmission({
    currentSegmentation: route.currentSegmentation,
    registerAnnotationActivity: overlayRefresh.registerAnnotationActivity,
    stageOptimisticSegments: overlayOptimistic.stageOptimisticSegments,
    clearOptimisticSegments: overlayOptimistic.clearOptimisticSegments,
    stageOptimisticRevisionTargets: overlayOptimistic.stageOptimisticRevisionTargets,
    getOptimisticTargetRevision: overlayOptimistic.getOptimisticTargetRevision,
    handleOverlayMutationRefresh: overlayRefresh.handleOverlayMutationRefresh,
  });

  const review = useSegmentationReviewWorkflow({
    currentSegmentation: route.currentSegmentation,
    activeSourceModel: route.activeSourceModel,
    isErSegmentation: route.isErSegmentation,
    supportsPointFeedback: route.supportsPointFeedback,
    hoverSegments: hover.hoverSegments,
    highlightedSegmentId: hover.highlightedSegmentId,
    hoverPoint: hover.hoverPoint,
    hoverActionMode: hover.hoverActionMode,
    setHoverActionMode: hover.setHoverActionMode,
    clearHoverInteraction: hover.clearHoverInteraction,
    applyLabelOverrides: overlayOptimistic.applyLabelOverrides,
    applyOptimisticLabel: overlayOptimistic.applyOptimisticLabel,
    rollbackOptimisticLabel: overlayOptimistic.rollbackOptimisticLabel,
    stageOptimisticRevisionTargets: overlayOptimistic.stageOptimisticRevisionTargets,
    getOptimisticTargetRevision: overlayOptimistic.getOptimisticTargetRevision,
    handleOverlayMutationRefresh: overlayRefresh.handleOverlayMutationRefresh,
    registerAnnotationActivity: overlayRefresh.registerAnnotationActivity,
    showErrorToast: ui.showErrorToast,
    showNoticeToast: ui.showNoticeToast,
    exitNavigateMode: ui.exitNavigateMode,
    drawing,
    submitConfirmedGeometriesOptimistically:
      confirmedSubmission.submitConfirmedGeometriesOptimistically,
  });
  const reviewMode = review.mode;
  const reviewDraw = review.draw;
  const reviewGroup = review.group;
  const reviewPointActions = review.pointActions;

  const { uncertainSegments, refetchUncertainSegments } = useUncertainSegments(
    route.currentSegmentation,
    reviewMode.workflowMode,
    ui.uncertainLimit,
    route.activeSourceModel
  );
  const leftSegments = useMemo(
    () => (reviewMode.workflowMode === "uncertain" ? uncertainSegments : []),
    [reviewMode.workflowMode, uncertainSegments]
  );
  const tooManyLeft = leftSegments.length > 500;
  const refetchLeftSegments = useCallback(() => {
    if (reviewMode.workflowMode === "uncertain") {
      return refetchUncertainSegments();
    }
    return Promise.resolve();
  }, [refetchUncertainSegments, reviewMode.workflowMode]);

  useEffect(() => {
    refetchLeftSegmentsRef.current = refetchLeftSegments;
  }, [refetchLeftSegments]);

  const processing = useSegmentationProcessingState({
    currentSegmentation: route.currentSegmentation,
    activeSourceModel: route.activeSourceModel,
    supportsPointFeedback: route.supportsPointFeedback,
    supportsInstanceParams: route.supportsInstanceParams,
    currentInstanceParams: route.currentInstanceParams,
    refetchSegmentations: route.refetchSegmentations,
    refreshSegmentViews: overlayRefresh.refreshSegmentViews,
  });

  const feedback = useSegmentationFeedback({
    currentSegmentationId: route.currentSegmentationId,
    supportsPointFeedback: route.supportsPointFeedback,
    refreshSegmentViews: overlayRefresh.refreshSegmentViews,
  });

  const completedRoiActive =
    reviewMode.workflowMode === "review" &&
    reviewMode.correctionMode.reviewPhase === "correction" &&
    reviewMode.correctionMode.correctionTool === "completed_roi";

  const completedRoi = useCompletedRoiWorkflow({
    currentSegmentation: route.currentSegmentation,
    active: completedRoiActive,
    isPointInsideImageBounds,
    registerAnnotationActivity: overlayRefresh.registerAnnotationActivity,
    showErrorToast: ui.showErrorToast,
  });

  const erRoi = useErRoiWorkflow({
    currentSegmentationId: route.currentSegmentationId,
    enabled: !route.isTissueSegmentation && Boolean(route.currentSegmentationId),
    image: route.image,
    isPointInsideImageBounds,
    refetchSegmentationRois: processing.refetchSegmentationRois,
    registerAnnotationActivity: overlayRefresh.registerAnnotationActivity,
    onRoiConfirmed: useCallback(() => {
      // After creating the ROI, hand off to Correct mode (the new ROI is the
      // active ROI, so the viewer auto-fits to it).
      reviewMode.handleReviewPhaseChange("correction");
    }, [reviewMode]),
    showErrorToast: ui.showErrorToast,
  });

  const erPolygonActive =
    route.isErSegmentation &&
    reviewMode.workflowMode === "review" &&
    reviewMode.correctionMode.reviewPhase === "correction" &&
    reviewMode.correctionMode.correctionTool === "polygon";

  const erPolygon = useErPolygonWorkflow({
    currentSegmentation: route.currentSegmentation,
    active: erPolygonActive,
    isPointInsideImageBounds,
    registerAnnotationActivity: overlayRefresh.registerAnnotationActivity,
    showErrorToast: ui.showErrorToast,
    showNoticeToast: ui.showNoticeToast,
    submitConfirmedGeometriesOptimistically:
      confirmedSubmission.submitConfirmedGeometriesOptimistically,
  });

  const removeArea = useRemoveAreaWorkflow({
    currentSegmentation: route.currentSegmentation,
    currentSegmentationId: route.currentSegmentationId,
    registerAnnotationActivity: overlayRefresh.registerAnnotationActivity,
    handleOverlayMutationRefresh: overlayRefresh.handleOverlayMutationRefresh,
    clearHoverInteraction: hover.clearHoverInteraction,
    showErrorToast: ui.showErrorToast,
    showNoticeToast: ui.showNoticeToast,
  });

  const tissue = useTissueLabeling({
    currentSegmentation: route.currentSegmentation,
    currentSegmentationId: route.currentSegmentationId,
    enabled: route.isTissueSegmentation,
    isPointInsideImageBounds,
    registerAnnotationActivity: overlayRefresh.registerAnnotationActivity,
    showErrorToast: ui.showErrorToast,
    showNoticeToast: ui.showNoticeToast,
    drawing,
    submitConfirmedGeometriesOptimistically:
      confirmedSubmission.submitConfirmedGeometriesOptimistically,
    refreshSegmentViews: overlayRefresh.refreshSegmentViews,
    setOverlayManifestPollingEnabled: overlay.manifest.setOverlayManifestPollingEnabled,
    clearHoverInteraction: hover.clearHoverInteraction,
  });

  // These shapes exist only in React until their respective Save/Confirm
  // action succeeds. A desktop update replaces the running application, so it
  // must wait for the user to finish or clear them rather than discarding an
  // annotation-in-progress.
  const hasUnsavedAnnotationDraft =
    Boolean(drawing.pendingPolygon?.length) ||
    drawing.brushStrokes.length > 0 ||
    Boolean(erRoi.pendingRoi) ||
    completedRoi.hasDraft ||
    erPolygon.hasDraft ||
    removeArea.removeAreaDrawing.brushStrokes.length > 0 ||
    tissue.addPolygon.hasDraft ||
    tissue.excludePolygon.hasDraft;
  useRestartBlocker(
    hasUnsavedAnnotationDraft,
    "Finish or clear the unsaved annotation draft before QuantEM restarts."
  );

  const handleCorrectionDrawComplete = useCallback(
    (points: Point[]) => {
      overlayRefresh.registerAnnotationActivity();
      drawing.handleDrawComplete(points);
    },
    [drawing, overlayRefresh]
  );

  const handleCorrectionBrushStroke = useCallback(
    (points: Point[]) => {
      overlayRefresh.registerAnnotationActivity();
      drawing.handleBrushStroke(points);
    },
    [drawing, overlayRefresh]
  );

  // TODO(quantem): the "add" correction tool (paint a stroke to add one object,
  // auto-refined server-side) has no QuantEM endpoint yet.
  // Until an organelle-agnostic add endpoint exists the stroke is a no-op and
  // the tool is not offered in the toolbar.
  const handleAddStroke = useCallback(() => {}, []);

  const handleRoiPlacementClick = useCallback(
    (point: Point) => {
      erRoi.setPendingRoi(erRoi.resolvePendingRoi(point));
    },
    [erRoi]
  );

  // Box-to-object. Owns its own pointer gestures and its own weights; the
  // screen only tells it where it is allowed to run and what to refresh.
  const samBox = useReviewSamBoxController({
    currentSegmentationId: route.currentSegmentationId,
    workflowMode: reviewMode.workflowMode,
    correctionMode: reviewMode.correctionMode,
    leftNavigateMode: ui.leftNavigateMode,
    isTissueSegmentation: route.isTissueSegmentation,
    onCorrectionToolChange: reviewMode.handleCorrectionToolChange,
    onOverlayMutation: (overlayMutation) =>
      overlayRefresh.handleOverlayMutationRefresh(
        (overlayMutation ?? null) as Parameters<
          typeof overlayRefresh.handleOverlayMutationRefresh
        >[0]
      ),
    showErrorToast: ui.showErrorToast,
    showNoticeToast: ui.showNoticeToast,
    registerAnnotationActivity: overlayRefresh.registerAnnotationActivity,
  });

  const interactions = useSegmentationInteractionRouter({
    currentSegmentationId: route.currentSegmentationId,
    leftNavigateMode: ui.leftNavigateMode,
    roiPlacementActive: erRoi.placementActive,
    isPointInsideImageBounds,
    applyLabelOverrides: overlayOptimistic.applyLabelOverrides,
    scheduleHoverSegmentQuery: hover.scheduleHoverSegmentQuery,
    clearHoverInteraction: hover.clearHoverInteraction,
    onRoiPlacementClick: handleRoiPlacementClick,
    completedRoi: {
      isActive: completedRoi.active,
      handlePolygonClick: completedRoi.handlePolygonClick,
      handlePolygonMouseMove: completedRoi.handlePolygonMouseMove,
    },
    erPolygon: {
      isActive: erPolygon.active,
      handlePolygonClick: erPolygon.handlePolygonClick,
      handlePolygonMouseMove: erPolygon.handlePolygonMouseMove,
    },
    tissue: {
      enabled: route.isTissueSegmentation,
      polygon: {
        isActive: tissue.activePolygonTool !== null,
        handlePolygonClick: tissue.activePolygonTool?.handlePolygonClick ?? (() => {}),
        handlePolygonMouseMove:
          tissue.activePolygonTool?.handlePolygonMouseMove ?? (() => {}),
      },
    },
    samBox: {
      isActive: samBox.isActive,
      handleImagePress: samBox.handleImagePress,
      handleImageDrag: samBox.handleImageDrag,
      handleImageRelease: samBox.handleImageRelease,
    },
    review: {
      hoverActionMode: hover.hoverActionMode,
      leftMode: reviewMode.leftMode,
      workflowMode: reviewMode.workflowMode,
      isGroupActionMode: reviewMode.isGroupActionModeActive,
      group: {
        handleImagePress: reviewGroup.handleGroupImagePress,
        handleImageDrag: reviewGroup.handleGroupImageDrag,
        handleImageRelease: reviewGroup.handleGroupImageRelease,
      },
      pointActions: {
        handleApply: reviewPointActions.handleApplyPointAction,
      },
    },
  });

  /**
   * The keyboard's half of the canvas reform.
   *
   * The object under the pointer is the object a key acts on, so a decision
   * costs one keystroke instead of a round trip to the sidebar and back --
   * which, at a few hundred objects an image, was most of the work.
   */
  const pointVerbs = useMemo(
    () => ({
      hoverPoint: hover.hoverPoint,
      hasHoverTarget: hover.hoverSegments.length > 0,
      keep: (point: Point) =>
        reviewPointActions.handleApplyPointAction(point, "confirm"),
      remove: (point: Point) =>
        reviewPointActions.handleApplyPointAction(point, "reject"),
      unmark: (point: Point) =>
        reviewPointActions.handleResetConfirmedToCandidate(point),
    }),
    [hover.hoverPoint, hover.hoverSegments.length, reviewPointActions]
  );

  useSegmentationKeyboardShortcuts({
    leftNavigateMode: ui.leftNavigateMode,
    toggleLeftNavigateMode: ui.toggleLeftNavigateMode,
    cycleHoverIndex: hover.cycleHoverIndex,
    pointVerbs,
    drawing,
    removeArea: {
      mode: removeArea.rightPanelRemoveMode,
      clearDrawing: removeArea.removeAreaDrawing.clearDrawing,
      canApply: removeArea.canApplyRemoveArea,
      handleApply: removeArea.handleApplyRemoveArea,
    },
    completedRoi: {
      isActive: completedRoi.active,
      canClosePolygon: completedRoi.canClosePolygon,
      hasDraft: completedRoi.hasDraft,
      handleClosePolygon: completedRoi.handleClosePolygon,
      clearDraft: completedRoi.clearDraft,
    },
    erPolygon: {
      isActive: erPolygon.active,
      canClosePolygon: erPolygon.canClosePolygon,
      hasDraft: erPolygon.hasDraft,
      handleClosePolygon: erPolygon.handleClosePolygon,
      clearDraft: erPolygon.clearDraft,
    },
    tissue: {
      enabled: route.isTissueSegmentation,
      polygonCanClose: tissue.activePolygonTool?.canClosePolygon ?? false,
      polygonHasDraft: tissue.activePolygonTool?.hasDraft ?? false,
      canConfirmBrush: tissue.canConfirmBrush,
      handleClosePolygon: () => {
        void tissue.activePolygonTool?.handleClosePolygon();
      },
      clearPolygon: () => {
        tissue.activePolygonTool?.clearDraft();
      },
      handleConfirmBrush: tissue.handleConfirmBrush,
    },
    review: {
      isGroupActionMode: reviewMode.isGroupActionModeActive,
      clearGroupSelection: reviewGroup.clearGroupBboxSelection,
      groupSelectionBBox: reviewGroup.groupSelectionBBox,
      groupBboxHighlightedSegmentIds: reviewGroup.groupBboxHighlightedSegmentIds,
      handleBatchGroupAction: reviewGroup.handleBatchGroupAction,
      activeGroupActionLabelState: reviewMode.activeGroupActionLabelState,
      leftMode: reviewMode.leftMode,
      correctionTool: reviewMode.correctionMode.correctionTool,
      handleAcceptPolygon: reviewDraw.handleAcceptPolygon,
    },
  });

  /**
   * The recovery from calibrated-after-the-fact, wired to a screen at last.
   *
   * `POST /api/segmentations/<id>/labels/clear` deletes every CONFIRMED and
   * EXCLUDED object; the analysis bundle's own caveat names it as the required
   * first step ("the objects have to go first") because a re-run drops any new
   * candidate landing on a confirmed or excluded object, so re-running without
   * clearing changes nothing. Clearing and re-running are one intention here —
   * the point of deleting the mis-scaled set is the set stamped with the pixel
   * size the image records now — so this does both, in that order. Errors are
   * left to the header's confirm dialog, which stays open and prints them.
   */
  const handleClearMislabeledObjects = useCallback(async () => {
    const segmentationId = route.currentSegmentationId;
    if (!segmentationId) return;
    const response = await clearSegmentationManualLabels(segmentationId);
    overlayRefresh.handleOverlayMutationRefresh(response.overlay ?? null);
    await route.refetchSegmentations();
    await processing.handleApplyFullImage();
  }, [
    overlayRefresh,
    processing,
    route,
  ]);

  const handleLeftViewportChange = useCallback(
    (nextViewport: Parameters<typeof route.publishFromViewer>[1]) => {
      overlayRefresh.registerAnnotationActivity();
      route.publishFromViewer("left", nextViewport);
    },
    [overlayRefresh, route]
  );

  const handleRightViewportChange = useCallback(
    (nextViewport: Parameters<typeof route.publishFromViewer>[1]) => {
      overlayRefresh.registerAnnotationActivity();
      route.publishFromViewer("right", nextViewport);
    },
    [overlayRefresh, route]
  );

  const viewModels = useSegmentationScreenViewModels({
    route,
    ui,
    overlay,
    review,
    completedRoi,
    erRoi,
    erPolygon,
    processing,
    feedback,
    hover,
    interactions,
    samBox,
    drawing,
    removeArea,
    tissue,
    modelRunnability,
    appliedAdapter,
    runScaleMismatch,
    onClearMislabeledObjects: handleClearMislabeledObjects,
    leftSegments,
    tooManyLeft,
    refetchUncertainSegments,
    handleCorrectionDrawComplete,
    handleCorrectionBrushStroke,
    handleAddStroke,
    handleLeftViewportChange,
    handleRightViewportChange,
  });

  if (!route.selectedImageId) {
    return <SegmentationScreenState kind="no-selection" />;
  }
  if (route.imageLoading || route.segmentationsLoading) {
    return <SegmentationScreenState kind="loading" />;
  }
  if (!route.image) {
    return <SegmentationScreenState kind="image-error" />;
  }
  if (!route.preprocessReady) {
    return (
      <SegmentationScreenState
        kind="preprocess"
        preprocessLabel={route.preprocessLabel}
      />
    );
  }
  if (route.visibleSegmentations.length === 0) {
    return <SegmentationScreenState kind="empty" image={route.image} />;
  }
  if (
    !viewModels.headerProps ||
    !viewModels.leftPanelProps ||
    !viewModels.rightPanelProps ||
    !viewModels.sidebarProps
  ) {
    return null;
  }

  return (
    <div className="segmentation-screen">
      {processing.shouldShowProcessingStatus && (
        <SegmentationJobBanner jobs={processing.processingJobs} />
      )}
      <SegmentationHeader {...viewModels.headerProps} />
      <main
        className={`segmentation-main ${
          ui.showConfirmedPanel ? "show-confirmed" : "hide-confirmed"
        }`}
      >
        <ConfirmDialog
          isOpen={completedRoi.saveDialogOpen}
          title={
            completedRoi.mode === "exclude"
              ? "Remove from the confirmed area"
              : "Add to the confirmed area"
          }
          message={
            completedRoi.mode === "exclude"
              ? `Remove this polygon from the confirmed area for this segmentation? Overlapping confirmed regions will be trimmed (or split) accordingly. ${CONFIRMED_AREA_EXPLANATION}`
              : `Add this polygon to the confirmed area for this segmentation? Overlapping confirmed regions will be merged into one. ${CONFIRMED_AREA_EXPLANATION}`
          }
          details={
            completedRoi.mode === "include" ? (
              <CompletedRoiBackgroundNotice
                pending={completedRoi.pendingBackground}
              />
            ) : null
          }
          detailsTone={
            isBackgroundCountWarning(completedRoi.pendingBackground)
              ? "warning"
              : "default"
          }
          confirmText={
            completedRoi.saving
              ? "Saving..."
              : completedRoi.mode === "exclude"
                ? "Remove area"
                : "Add area"
          }
          cancelText="Cancel"
          onConfirm={() => {
            void completedRoi.confirmSave();
          }}
          onCancel={completedRoi.cancelSave}
        />
        <SegmentationSidebar
          {...viewModels.sidebarProps}
          // Wired here rather than in the view-model hook because the dial
          // needs `refreshSegmentViews`, which lives on this screen: a
          // re-extract replaces the whole candidate set and renumbers the
          // result, so the ID-map overlay and the segment lists are both stale
          // afterwards. The dial deliberately knows nothing about drawing.
          includeLevel={
            route.currentSegmentation
              ? {
                  segmentationId: route.currentSegmentation.id,
                  sourceModel: route.activeSourceModel,
                  segmentationInternalName: route.segmentationInternalName,
                  statusStage: route.currentSegmentation.status_stage,
                  onSourceModelChange: route.handleSourceModelChange,
                  onRunQueued: () => {
                    void Promise.all([
                      processing.refetchJobs(),
                      route.refetchSegmentations(),
                    ]);
                  },
                  onRunFinished: () => {
                    void route.refetchSegmentations();
                  },
                  onReextracted: overlayRefresh.refreshSegmentViews,
                }
              : undefined
          }
        />
        <SegmentationLeftPanel {...viewModels.leftPanelProps} />
        {ui.showConfirmedPanel && (
          <SegmentationRightPanel {...viewModels.rightPanelProps} />
        )}
        {ui.toast && (
          // `role="alert"` interrupts a screen reader, which is right for a
          // failure and wrong for "that worked, and here is what it did":
          // a notice announces politely instead.
          <div
            className={`segmentation-toast segmentation-toast-${ui.toast.tone}`}
            role={ui.toast.tone === "error" ? "alert" : "status"}
          >
            <span>{ui.toast.message}</span>
            <button type="button" onClick={ui.dismissToast}>
              Dismiss
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
