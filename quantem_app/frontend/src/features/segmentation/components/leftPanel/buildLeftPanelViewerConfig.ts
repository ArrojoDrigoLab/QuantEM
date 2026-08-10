import type { ComponentProps } from "react";
import { getAssetNgffUrl } from "@/shared/api/assets";
import { ADD_BRUSH_DIAMETER } from "@/features/segmentation/screen/utils/constants";
import { ImageViewer } from "@/viewer/components/ImageViewer";
import type { OverlayScene } from "@/viewer/overlays/types";
import type { SegmentationLeftPanelProps } from "@/features/segmentation/components/leftPanel/types";

type LeftPanelViewerConfigArgs = Pick<
  SegmentationLeftPanelProps,
  "viewer" | "workflow" | "segments" | "roi" | "drawing" | "overlays"
> & {
  overlayScene: OverlayScene;
};

export function buildLeftPanelViewerConfig({
  viewer,
  workflow,
  segments,
  roi,
  drawing,
  overlays,
  overlayScene,
}: LeftPanelViewerConfigArgs): ComponentProps<typeof ImageViewer> {
  const viewerHighlightedId = workflow.groupConfirmActive
    ? null
    : segments.highlightedSegmentId;
  const showHoverCursor =
    workflow.mode === "review" &&
    workflow.leftMode === "hover" &&
    segments.hoverCount > 0 &&
    !workflow.groupConfirmActive;
  const disablePanForGroup =
    workflow.mode === "review" && workflow.groupConfirmActive && !workflow.navigateMode;
  const isAddMode =
    workflow.mode === "review" &&
    workflow.reviewPhase === "correction" &&
    workflow.correctionTool === "add";
  // While placing an ER ROI, the click must reach the placement handler: keep
  // clicks live (even in navigate mode), force brush/draw off, and disable pan
  // so a placement click is never swallowed as a brush stroke or a pan gesture.
  const roiPlacementActive = workflow.roiPlacementActive;
  const fitBounds = viewer.transientFitBounds
    ? viewer.transientFitBounds
    : roi.activeRoi
      ? {
          x: roi.activeRoi.x,
          y: roi.activeRoi.y,
          width: roi.activeRoi.width,
          height: roi.activeRoi.height,
        }
      : null;
  const fitBoundsKey = viewer.transientFitBounds
    ? viewer.transientFitBoundsKey ?? "transient-fit"
    : roi.activeRoi
      ? `${viewer.image.id}:${roi.activeRoi.id}`
      : null;

  return {
    image: {
      ngffUrl: getAssetNgffUrl(viewer.image.id, null),
      width: viewer.image.width,
      height: viewer.image.height,
    },
    className: "viewer-container",
    viewport: {
      state: viewer.viewport ?? undefined,
      onChange: viewer.onViewportChange,
      disablePan: disablePanForGroup || roiPlacementActive,
      fitBounds,
      fitBoundsKey,
    },
    overlays: {
      persistent: overlayScene.persistent,
      transient: overlayScene.transient,
      rasterLayers: viewer.overlayNgffLayers ?? [],
      idMapOverlays: viewer.idMapOverlay ? [viewer.idMapOverlay] : [],
      onRasterRevisionDisplayed: viewer.onOverlayRevisionDisplayed,
    },
    interactions: {
      onImageClick:
        workflow.navigateMode && !roiPlacementActive ? undefined : segments.onClick,
      onImageMouseMove: workflow.navigateMode ? undefined : segments.onMouseMove,
      onImageMouseLeave: workflow.navigateMode ? undefined : segments.onMouseLeave,
      onImagePress: workflow.navigateMode ? undefined : segments.onPress,
      onImageDrag: workflow.navigateMode ? undefined : segments.onDrag,
      onImageRelease: workflow.navigateMode ? undefined : segments.onRelease,
      draw: {
        enabled:
          !roiPlacementActive &&
          !workflow.navigateMode &&
          workflow.leftMode === "draw" &&
          !(
            workflow.mode === "review" &&
            workflow.reviewPhase === "correction" &&
            (workflow.correctionTool === "draw" ||
              workflow.correctionTool === "erase")
          ),
        onComplete: drawing.onDrawComplete,
      },
      brush: {
        enabled:
          !roiPlacementActive &&
          !workflow.navigateMode &&
          (workflow.leftMode === "annotate" ||
            isAddMode ||
            (workflow.mode === "review" &&
              workflow.reviewPhase === "correction" &&
              (workflow.correctionTool === "draw" ||
                workflow.correctionTool === "erase") &&
              !overlays.disableCorrectionBrush)),
        size: isAddMode
          ? ADD_BRUSH_DIAMETER
          : workflow.mode === "review" &&
              workflow.reviewPhase === "correction" &&
              (workflow.correctionTool === "draw" ||
                workflow.correctionTool === "erase")
            ? drawing.brushSize
            : roi.brushSize,
        color: isAddMode
          ? "#33cc66"
          : workflow.mode === "review" &&
              workflow.reviewPhase === "correction" &&
              workflow.correctionTool === "erase"
            ? "#ff5d5d"
            : workflow.mode === "review" &&
                workflow.reviewPhase === "correction" &&
                workflow.correctionTool === "draw"
              ? "#ffb000"
              : roi.brushColor,
        onStroke: isAddMode
          ? drawing.onAddStroke
          : workflow.mode === "review" &&
              workflow.reviewPhase === "correction" &&
              workflow.correctionTool === "erase"
            ? drawing.onEraseStroke
            : workflow.mode === "review" &&
                workflow.reviewPhase === "correction" &&
                workflow.correctionTool === "draw"
              ? overlays.disableCorrectionBrush
                ? undefined
                : drawing.onBrushStroke
              : roi.onBrushStroke,
      },
    },
    highlighting: {
      highlightedSegmentId: viewerHighlightedId,
      hoverCursor: !workflow.navigateMode && showHoverCursor,
      cursorMode: !workflow.navigateMode && workflow.targetCursorActive ? "target" : undefined,
      hoverBadge: workflow.navigateMode
        ? undefined
        : { point: segments.hoverPoint, count: segments.hoverCount },
    },
  };
}
