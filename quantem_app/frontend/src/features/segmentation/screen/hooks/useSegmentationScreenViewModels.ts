import { useEffect, useMemo, useState } from "react";
import { useDrawing } from "@/hooks/useDrawing";
import { useCompletedRoiWorkflow } from "@/features/segmentation/screen/hooks/useCompletedRoiWorkflow";
import { useErRoiWorkflow } from "@/features/segmentation/screen/hooks/useErRoiWorkflow";
import { useErPolygonWorkflow } from "@/features/segmentation/screen/hooks/useErPolygonWorkflow";
import { useRemoveAreaWorkflow } from "@/features/segmentation/screen/hooks/useRemoveAreaWorkflow";
import { useTissueLabeling } from "@/features/segmentation/screen/hooks/useTissueLabeling";
import { useSegmentationFeedback } from "@/features/segmentation/screen/hooks/useSegmentationFeedback";
import { useSegmentationHoverQuery } from "@/features/segmentation/screen/hooks/useSegmentationHoverQuery";
import { useSegmentationInteractionRouter } from "@/features/segmentation/screen/hooks/useSegmentationInteractionRouter";
import { useSegmentationOverlayState } from "@/features/segmentation/screen/hooks/useSegmentationOverlayState";
import { useSegmentationProcessingState } from "@/features/segmentation/screen/hooks/useSegmentationProcessingState";
import { useSegmentationReviewWorkflow } from "@/features/segmentation/screen/hooks/useSegmentationReviewWorkflow";
import { useSegmentationRouteState } from "@/features/segmentation/screen/hooks/useSegmentationRouteState";
import { useSegmentationScreenUiState } from "@/features/segmentation/screen/hooks/useSegmentationScreenUiState";
import {
  buildLeftPanelWorkflowState,
  buildSegmentationHeaderProps,
  buildSegmentationLeftPanelProps,
  buildSegmentationRightPanelProps,
  buildSegmentationRoiViewModel,
  buildSegmentationSidebarProps,
} from "@/features/segmentation/screen/viewModels";
import { generateCompletedRoiDraftOverlays } from "@/features/segmentation/overlays/completedRois";
import type { Point } from "@/utils/geometry";
import type { SegmentObject } from "@/shared/types";
import type { Runnability } from "@/features/models/runnable";
import { packIdForSourceModel } from "@/features/models/runnable";
import type { AppliedAdapterState } from "@/features/models/appliedAdapter";
import type { ModelCatalogue } from "@/shared/types/finetune";
import type { ReviewSamBoxController } from "@/features/segmentation/screen/hooks/review/useReviewSamBoxController";

interface UseSegmentationScreenViewModelsArgs {
  route: ReturnType<typeof useSegmentationRouteState>;
  ui: ReturnType<typeof useSegmentationScreenUiState>;
  overlay: ReturnType<typeof useSegmentationOverlayState>;
  review: ReturnType<typeof useSegmentationReviewWorkflow>;
  completedRoi: ReturnType<typeof useCompletedRoiWorkflow>;
  erRoi: ReturnType<typeof useErRoiWorkflow>;
  erPolygon: ReturnType<typeof useErPolygonWorkflow>;
  processing: ReturnType<typeof useSegmentationProcessingState>;
  feedback: ReturnType<typeof useSegmentationFeedback>;
  hover: ReturnType<typeof useSegmentationHoverQuery>;
  interactions: ReturnType<typeof useSegmentationInteractionRouter>;
  /** Box-to-object: its toolbar controls and its live/pending rectangles. */
  samBox: ReviewSamBoxController;
  drawing: ReturnType<typeof useDrawing>;
  removeArea: ReturnType<typeof useRemoveAreaWorkflow>;
  tissue: ReturnType<typeof useTissueLabeling>;
  /** Whether the selected source model can be loaded here. */
  modelRunnability: Runnability;
  modelCatalogue: ModelCatalogue | null;
  /** The adapter applied to this segmentation, and whether it is in force. */
  appliedAdapter: AppliedAdapterState | null;
  /**
   * Delete every reviewed object and queue a fresh run — the recovery for a
   * pixel size typed in after the objects were made. The header renders the
   * button beside the objects-pixel-size warning chip and owns the confirm.
   */
  onClearMislabeledObjects: () => Promise<void>;
  leftSegments: SegmentObject[];
  tooManyLeft: boolean;
  refetchUncertainSegments: () => Promise<void>;
  handleCorrectionDrawComplete: (points: Point[]) => void;
  handleCorrectionBrushStroke: (points: Point[]) => void;
  handleAddStroke: (points: Point[]) => void;
  handleLeftViewportChange: (
    nextViewport: Parameters<ReturnType<typeof useSegmentationRouteState>["publishFromViewer"]>[1]
  ) => void;
  handleRightViewportChange: (
    nextViewport: Parameters<ReturnType<typeof useSegmentationRouteState>["publishFromViewer"]>[1]
  ) => void;
}

export function useSegmentationScreenViewModels({
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
  modelCatalogue,
  appliedAdapter,
  onClearMislabeledObjects,
  leftSegments,
  tooManyLeft,
  refetchUncertainSegments,
  handleCorrectionDrawComplete,
  handleCorrectionBrushStroke,
  handleAddStroke,
  handleLeftViewportChange,
  handleRightViewportChange,
}: UseSegmentationScreenViewModelsArgs) {
  const [roiFocusRequest, setRoiFocusRequest] = useState<{
    roiId: string;
    revision: number;
  } | null>(null);

  // A focus request belongs only to the labeling screen and segmentation in
  // which Open was clicked. Entering another view must start at the full image.
  useEffect(() => {
    setRoiFocusRequest(null);
  }, [route.currentSegmentationId, route.selectedAssetId]);

  return useMemo(() => {
    if (!route.image) {
      return {
        headerProps: null,
        leftPanelProps: null,
        rightPanelProps: null,
        sidebarProps: null,
        instanceParamsBarProps: null,
      };
    }
    const overlayManifest = overlay.manifest;
    const overlayLayers = overlay.layers;
    const overlayOptimistic = overlay.optimistic;
    const reviewMode = review.mode;
    const reviewDraw = review.draw;
    const reviewGroup = review.group;
    const reviewPointActions = review.pointActions;
    const reviewDerived = review.derived;
    const focusedRoi = roiFocusRequest
      ? (processing.segmentationRois ?? []).find(
          (roi) => roi.id === roiFocusRequest.roiId
        ) ?? null
      : null;

    const disableCorrectionBrush = removeArea.rightPanelRemoveMode === "area";
    const leftPanelWorkflowState = buildLeftPanelWorkflowState({
      isTissueSegmentation: route.isTissueSegmentation,
      tissueTool: tissue.tool,
      leftNavigateMode: ui.leftNavigateMode,
      workflowMode: reviewMode.workflowMode,
      leftMode: reviewMode.leftMode,
      correctionMode: reviewMode.correctionMode,
      isGroupActionModeActive: reviewMode.isGroupActionModeActive,
      roiPlacementActive: erRoi.placementActive,
      samBoxActive: samBox.isSelected,
    });

    const headerProps = buildSegmentationHeaderProps({
      image: route.image,
      currentSegmentation: route.currentSegmentation,
      sourceModelOptions: route.sourceModelOptions,
      activeSourceModel: route.activeSourceModel,
      // What the raster on screen was actually built from, straight off the
      // manifest -- not the selector, which is a request rather than a fact.
      displayedSourceModel: overlayManifest.overlayManifest?.source_model ?? null,
      fineTuneEligibilityRevision: [
        ...completedRoi.items.map((item) => item.id),
        ...(processing.segmentationRois ?? [])
          .filter((roi) => Boolean(roi.completed_for_segmentation))
          .map((roi) => roi.id),
      ]
        .sort()
        .join(":"),
      onBackToHome: route.handleBackToHome,
      onBackToExperiment: route.handleBackToExperiment,
      onBackToViewer: route.handleOpenViewer,
      onSourceModelChange: route.handleSourceModelChange,
      onToggleSegmentationComplete: processing.handleToggleSegmentationComplete,
      isApplyingFull: processing.isApplyingFull,
      isApplyingActiveRoi: processing.isRerunningRoi,
      hasQueuedOrRunningOrganelleTask: processing.hasQueuedOrRunningOrganelleTask,
      modelCatalogue,
      appliedAdapter,
      onClearMislabeledObjects,
    });

    const leftPanelProps = buildSegmentationLeftPanelProps({
      base: {
        layerControls: {
          usesRasterOverlay: overlayManifest.usesRasterReviewOverlay,
          candidates: {
            strokeWidth: overlayLayers.leftPanelLayerStyles.candidateStrokeWidth,
            fillOpacity: overlayLayers.leftPanelLayerStyles.candidateFillOpacity,
            showBorders: overlayLayers.showCandidateBorders,
            onStrokeWidthChange:
              overlayLayers.updateLayerStyles.setCandidateStrokeWidth,
            onFillOpacityChange:
              overlayLayers.updateLayerStyles.setCandidateFillOpacity,
            onShowBordersChange: overlayLayers.setShowCandidateBorders,
          },
          confirmed: {
            strokeWidth: overlayLayers.leftPanelLayerStyles.confirmedStrokeWidth,
            fillOpacity: overlayLayers.leftPanelLayerStyles.confirmedFillOpacity,
            showBorders: overlayLayers.showConfirmedBorders,
            onStrokeWidthChange:
              overlayLayers.updateLayerStyles.setConfirmedStrokeWidth,
            onFillOpacityChange:
              overlayLayers.updateLayerStyles.setConfirmedFillOpacity,
            onShowBordersChange: overlayLayers.setShowConfirmedBorders,
          },
        },
        viewer: {
          image: route.image,
          segmentationTypeInternalName: route.segmentationInternalName,
          useSmoothedGeometry: route.useSmoothedSegmentGeometry,
          layerStyles: overlayLayers.leftPanelLayerStyles,
          viewport: route.viewport,
          onViewportChange: handleLeftViewportChange,
          overlayNgffLayers: [],
          idMapOverlays:
            reviewMode.workflowMode === "review" ? overlayLayers.leftIdMapOverlays : [],
          onOverlayRevisionDisplayed: overlayManifest.handleLeftOverlayRevisionDisplayed,
          transientFitBounds: focusedRoi
            ? {
                x: focusedRoi.x,
                y: focusedRoi.y,
                width: focusedRoi.width,
                height: focusedRoi.height,
              }
            : null,
          transientFitBoundsKey: focusedRoi
            ? `${route.image.id}:${route.currentSegmentationId}:${focusedRoi.id}:${roiFocusRequest?.revision}`
            : null,
        },
        workflow: leftPanelWorkflowState,
        segments: {
          items: [],
          highlightedSegmentId: null,
          hoverPoint: null,
          hoverCount: 0,
          tooMany: tooManyLeft,
          // Image clicks drive confirm/reject-object and group selection through
          // the interaction router.
          onClick: interactions.onLeftClick,
          onPress: interactions.onLeftImagePress,
          onDrag: interactions.onLeftImageDrag,
          onRelease: interactions.onLeftImageRelease,
          onMouseMove: interactions.onLeftMouseMove,
          onMouseLeave: interactions.onLeftMouseLeave,
          groupSelectionBBox: reviewGroup.groupSelectionBBox,
          groupHighlightedSegmentIds: reviewGroup.groupBboxHighlightedSegmentIds,
        },
        roi: buildSegmentationRoiViewModel(
          processing.activeRoi,
          processing.segmentationRois ?? []
        ),
        drawing: {
          pendingPolygon: drawing.pendingPolygon,
          brushStrokes: drawing.brushStrokes,
          brushSize: drawing.brushSize,
          onDrawComplete: handleCorrectionDrawComplete,
          onBrushStroke: handleCorrectionBrushStroke,
          onEraseStroke: reviewDraw.handleEraseStroke,
          onAddStroke: handleAddStroke,
          onAccept: reviewDraw.handleAcceptPolygon,
          onCancel: drawing.clearDrawing,
        },
        uncertain: {
          limit: ui.uncertainLimit,
          onLimitChange: ui.setUncertainLimit,
          onRefresh: refetchUncertainSegments,
        },
        completedRoi: {
          active: completedRoi.active,
          loading: completedRoi.loading,
          mode: completedRoi.mode,
          items: completedRoi.items,
          polygons: completedRoi.polygons,
          liveSectionPoints: completedRoi.liveSectionPoints,
          hasDraft: completedRoi.hasDraft,
          canClosePolygon: completedRoi.canClosePolygon,
          canSave: completedRoi.canSave,
          isSaving: completedRoi.saving,
          onModeChange: completedRoi.setMode,
          onClosePolygon: () => {
            void completedRoi.handleClosePolygon();
          },
          onRequestSave: completedRoi.requestSave,
          onClear: completedRoi.clearDraft,
        },
        feedback: {
          items: feedback.feedbackLog,
        },
        overlays: {
          disableCorrectionBrush,
          hideRoiOverlayId: erRoi.pendingRoi?.roiId ?? null,
          extraTransientOverlays: [
            ...overlayOptimistic.optimisticTransientOverlays,
            ...samBox.overlays,
            ...erRoi.pendingRoiOverlays,
            ...(erPolygon.active
              ? generateCompletedRoiDraftOverlays(
                  erPolygon.polygons,
                  erPolygon.liveSectionPoints,
                  "include"
                ).map((overlay) => ({ ...overlay, id: `er-polygon-${overlay.id}` }))
              : []),
            ...(tissue.enabled && tissue.activePolygonTool
              ? generateCompletedRoiDraftOverlays(
                  tissue.activePolygonTool.polygons,
                  tissue.activePolygonTool.liveSectionPoints,
                  tissue.operation
                ).map((overlay) => ({ ...overlay, id: `tissue-polygon-${overlay.id}` }))
              : []),
          ],
        },
      },
      leftSegments,
      applyLabelOverrides: overlayOptimistic.applyLabelOverrides,
      workflowMode: reviewMode.workflowMode,
      reviewInteractionSegments: reviewDerived.reviewInteractionSegments,
      hoverPoint: hover.hoverPoint,
      hoverSegments: hover.hoverSegments,
      highlightedSegmentId: hover.highlightedSegmentId,
      groupSelectionBBox: reviewGroup.groupSelectionBBox,
      groupBboxHighlightedSegmentIds: reviewGroup.groupBboxHighlightedSegmentIds,
    });

    const rightPanelProps = buildSegmentationRightPanelProps({
      image: route.image,
      segmentationTypeInternalName: route.segmentationInternalName,
      useSmoothedGeometry: route.useSmoothedSegmentGeometry,
      viewport: route.viewport,
      onViewportChange: handleRightViewportChange,
      confirmedSegments: overlayOptimistic.optimisticConfirmed,
      tooManyRight: false,
      activeRoi: processing.activeRoi,
      rois: processing.segmentationRois ?? [],
      removeMode: removeArea.rightPanelRemoveMode,
      onRemoveModeChange: removeArea.setRightPanelRemoveMode,
      onRemoveObjectClick: (segmentId) => {
        if (!segmentId) return;
        void reviewPointActions.handleDeleteConfirmedObject(segmentId);
      },
      removeAreaBrushSize: removeArea.removeAreaDrawing.brushSize,
      onRemoveAreaBrushSizeChange: removeArea.removeAreaDrawing.setBrushSize,
      removeAreaBrushStrokes: removeArea.removeAreaDrawing.brushStrokes,
      onRemoveAreaBrushStroke: removeArea.handleRemoveAreaBrushStroke,
      canApplyRemoveArea: removeArea.canApplyRemoveArea,
      onApplyRemoveArea: () => {
        void removeArea.handleApplyRemoveArea();
      },
      removingArea: removeArea.isRemovingArea,
      layerControls: {
        usesRasterOverlay: overlayManifest.usesRasterReviewOverlay,
        confirmed: {
          strokeWidth: overlayLayers.rightPanelConfirmedStyle.strokeWidth,
          fillOpacity: overlayLayers.rightPanelConfirmedStyle.fillOpacity,
          showBorders: overlayLayers.showRightConfirmedBorders,
          onStrokeWidthChange: overlayLayers.updateRightLayerStyle.setStrokeWidth,
          onFillOpacityChange: overlayLayers.updateRightLayerStyle.setFillOpacity,
          onShowBordersChange: overlayLayers.setShowRightConfirmedBorders,
        },
      },
      overlayNgffLayers: [],
      idMapOverlays:
        reviewMode.workflowMode === "review" ? overlayLayers.rightIdMapOverlays : [],
      onOverlayRevisionDisplayed: overlayManifest.handleRightOverlayRevisionDisplayed,
    });

    const instanceParamsBarProps = {
      enabled: route.supportsInstanceParams,
      draft: processing.instanceParamsDraft,
      isSaving: processing.isSavingInstanceParams,
      hasQueuedOrRunningOrganelleTask: processing.hasQueuedOrRunningOrganelleTask,
      onChange: processing.updateInstanceParam,
      onSave: () => {
        void processing.handleSaveInstanceParams();
      },
    };

    const selectedPackId = packIdForSourceModel(route.activeSourceModel);
    const testDisabled =
      !selectedPackId ||
      modelRunnability.state === "blocked" ||
      route.currentSegmentation?.status_stage === "COMPLETED" ||
      route.currentSegmentation?.is_complete === true ||
      processing.isApplyingFull ||
      processing.isRerunningRoi ||
      processing.hasQueuedOrRunningOrganelleTask;
    const testDisabledReason = !selectedPackId
      ? "Select QuantEM or OmniEM before testing an ROI."
      : modelRunnability.state === "blocked"
        ? modelRunnability.reason ?? "The selected model cannot run here."
        : route.currentSegmentation?.status_stage === "COMPLETED" ||
            route.currentSegmentation?.is_complete === true
          ? "This segmentation is locked. Unlock it before testing an ROI."
          : processing.isApplyingFull ||
              processing.isRerunningRoi ||
              processing.hasQueuedOrRunningOrganelleTask
            ? "Processing in progress"
            : undefined;

    const sidebarProps = buildSegmentationSidebarProps({
      tissue: {
        enabled: route.isTissueSegmentation,
        tool: tissue.tool,
        onToolChange: tissue.setTool,
        operation: tissue.operation,
        onOperationChange: tissue.setOperation,
        brushSize: tissue.brushSize,
        onBrushSizeChange: tissue.setBrushSize,
        canConfirmBrush: tissue.canConfirmBrush,
        hasBrushStrokes: tissue.hasBrushStrokes,
        confirmingBrush: tissue.confirmingBrush,
        onConfirmBrush: () => {
          void tissue.handleConfirmBrush();
        },
        onClearBrush: tissue.handleClearBrush,
        polygonHasDraft: tissue.activePolygonTool?.hasDraft ?? false,
        polygonCanClose: tissue.activePolygonTool?.canClosePolygon ?? false,
        onClosePolygon: () => {
          void tissue.activePolygonTool?.handleClosePolygon();
        },
        onClearPolygon: () => {
          tissue.activePolygonTool?.clearDraft();
        },
      },
      review: {
        workflowMode: reviewMode.workflowMode,
        reviewPhase: reviewMode.correctionMode.reviewPhase,
        correctionTool: reviewMode.correctionMode.correctionTool,
        hoverActionMode: hover.hoverActionMode,
        drawBrushSize: drawing.brushSize,
        draftOperation: drawing.draftOperation,
        hasDrawStrokes: drawing.brushStrokes.length > 0,
        supportsPointFeedback: route.supportsPointFeedback,
        isErSegmentation: route.isErSegmentation,
        canApplyGroupAction: reviewGroup.groupBboxHighlightedSegmentIds.length > 0,
        polygonHasDraft: erPolygon.hasDraft,
        polygonCanClose: erPolygon.canClosePolygon,
        onReviewPhaseChange: reviewMode.handleReviewPhaseChange,
        onCorrectionToolChange: reviewMode.handleCorrectionToolChange,
        // Not the raw setter: picking an action mode also leaves Navigate, or
        // the first box-drag pans instead of selecting.
        onHoverActionModeChange: reviewMode.handleHoverActionModeChange,
        onDrawBrushSizeChange: drawing.setBrushSize,
        onDraftOperationChange: drawing.setDraftOperation,
        onClearDrawing: drawing.clearDrawing,
        onConfirmShape: () => {
          void reviewDraw.handleAcceptPolygon();
        },
        onClosePolygon: () => {
          void erPolygon.handleClosePolygon();
        },
        onApplyGroupAction: (mode) => {
          reviewGroup.handleToolbarGroupAction(mode);
        },
        extraModes: samBox.controls,
      },
      layers: {
        overlayUpdating: overlayManifest.overlayUpdating,
        // Finding V4: the sidebar could say a build was in progress but had no
        // way to say one had failed, so a terminal failure showed as silence.
        overlayBuildFailed: overlayManifest.overlayBuildFailed,
        overlayManifest: overlayManifest.overlayManifest,
        overlaySegmentationId: route.currentSegmentation?.id ?? null,
        onOverlayBuildRetried: overlayManifest.handleOverlayBuildRetried,
      },
      view: {
        leftNavigateMode: ui.leftNavigateMode,
        onLeftNavigateModeChange: ui.setLeftNavigateMode,
        showConfirmedPanel: ui.showConfirmedPanel,
        onShowConfirmedPanelChange: ui.setShowConfirmedPanel,
        isGroupActionMode: reviewMode.isGroupActionModeActive,
        activeGroupActionVerb:
          reviewMode.activeGroupActionLabelState === "EXCLUDED" ? "reject" : "confirm",
        groupSelectionCount: reviewGroup.groupBboxHighlightedSegmentIds.length,
      },
    });

    return {
      headerProps,
      leftPanelProps,
      rightPanelProps,
      sidebarProps: route.currentSegmentation && !route.isTissueSegmentation
        ? {
            ...sidebarProps,
            erRoi: {
              placementActive: erRoi.placementActive,
              pendingRoiActive: erRoi.pendingRoi !== null,
              relocatingRoiId: erRoi.relocatingRoiId,
              confirming: erRoi.confirming,
              rois: processing.segmentationRois ?? [],
              activeRoiId: processing.activeRoi?.id ?? null,
              markingRoiId: erRoi.markingRoiId,
              deletingRoiId: erRoi.deletingRoiId,
              activatingRoiId: erRoi.activatingRoiId,
              testingRoiId: processing.rerunningRoiId,
              testDisabled,
              testDisabledReason,
              onStartPlacement: erRoi.startPlacement,
              onEditRoi: erRoi.editRoi,
              onCancelPlacement: erRoi.cancelPlacement,
              onConfirmRoi: () => {
                void erRoi.confirmRoi();
              },
              onMarkRoiDone: (roiId: string, done: boolean) => {
                void erRoi.markRoiDone(roiId, done);
              },
              onDeleteRoi: (roiId: string) => {
                void erRoi.deleteRoi(roiId);
              },
              onActivateRoi: (roiId: string) => {
                void (async () => {
                  if (processing.activeRoi?.id !== roiId) {
                    const activated = await erRoi.activateRoi(roiId);
                    if (!activated) return;
                  }
                  setRoiFocusRequest((previous) => ({
                    roiId,
                    revision: (previous?.revision ?? 0) + 1,
                  }));
                })();
              },
              onTestRoi: (roiId: string) => {
                void (async () => {
                  if (processing.activeRoi?.id !== roiId) {
                    const activated = await erRoi.activateRoi(roiId);
                    if (!activated) return;
                  }
                  await processing.handleRerunRoi(roiId);
                })();
              },
            },
          }
        : sidebarProps,
      instanceParamsBarProps,
    };
  }, [
    completedRoi,
    drawing,
    erRoi,
    erPolygon,
    feedback.feedbackLog,
    handleAddStroke,
    handleCorrectionBrushStroke,
    handleCorrectionDrawComplete,
    handleLeftViewportChange,
    handleRightViewportChange,
    onClearMislabeledObjects,
    interactions,
    samBox,
    leftSegments,
    modelRunnability,
    modelCatalogue,
    appliedAdapter,
    overlay,
    processing,
    roiFocusRequest,
    refetchUncertainSegments,
    removeArea,
    tissue,
    review,
    route,
    tooManyLeft,
    ui,
    hover,
  ]);
}
