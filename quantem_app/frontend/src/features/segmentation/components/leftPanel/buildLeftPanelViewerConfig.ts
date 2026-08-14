import type { ComponentProps } from "react";
import { getAssetNgffUrl } from "@/shared/api/assets";
import { ADD_BRUSH_DIAMETER } from "@/features/segmentation/screen/utils/constants";
import { ImageViewer } from "@/viewer/components/ImageViewer";
import { resolvePixelSize } from "@/shared/pixelSize";
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
  // Placement remains armed behind Navigate, but Navigate always owns the
  // canvas until the user turns it off.
  const roiPlacementActive = workflow.roiPlacementActive;
  // Box-to-object owns the entire drag. In particular, the normal draw brush
  // must be off or it consumes pointer-up before the box tool can submit.
  const samBoxActive = workflow.samBoxActive;
  // Being active controls which ROI receives labeling actions; it is not a
  // camera command. Only an explicit Open action supplies transient bounds.
  // This keeps a freshly entered labeling view fitted to the whole image.
  const fitBounds = viewer.transientFitBounds ?? null;
  const fitBoundsKey = fitBounds
    ? viewer.transientFitBoundsKey ?? "transient-fit"
    : null;

  return {
    image: {
      ngffUrl: getAssetNgffUrl(viewer.image.id, null),
      width: viewer.image.width,
      height: viewer.image.height,
      // Drives the scale bar. `resolvePixelSize` returns null for an
      // uncalibrated image and the canvas then draws no bar, which is the
      // honest outcome: this screen's whole job is measurements in nanometres.
      pixelSizeNm: resolvePixelSize(viewer.image).valueNm,
    },
    className: "viewer-container",
    viewport: {
      state: viewer.viewport ?? undefined,
      onChange: viewer.onViewportChange,
      disablePan:
        !workflow.navigateMode &&
        Boolean(disablePanForGroup || roiPlacementActive || samBoxActive),
      fitBounds,
      fitBoundsKey,
      // Doubling both ROI dimensions makes the square occupy about half of
      // the pane's limiting (shorter) axis, leaving useful image context.
      fitBoundsPaddingRatio: fitBounds ? 1 : undefined,
    },
    overlays: {
      persistent: overlayScene.persistent,
      transient: overlayScene.transient,
      rasterLayers: viewer.overlayNgffLayers ?? [],
      idMapOverlays: viewer.idMapOverlays ?? [],
      onRasterRevisionDisplayed: viewer.onOverlayRevisionDisplayed,
    },
    interactions: {
      mode: workflow.navigateMode ? "navigate" : undefined,
      onImageClick:
        workflow.navigateMode || samBoxActive ? undefined : segments.onClick,
      onImageMouseMove: workflow.navigateMode ? undefined : segments.onMouseMove,
      onImageMouseLeave: workflow.navigateMode ? undefined : segments.onMouseLeave,
      onImagePress: workflow.navigateMode ? undefined : segments.onPress,
      onImageDrag: workflow.navigateMode ? undefined : segments.onDrag,
      onImageRelease: workflow.navigateMode ? undefined : segments.onRelease,
      draw: {
        enabled:
          !roiPlacementActive &&
          !samBoxActive &&
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
          !samBoxActive &&
          !workflow.navigateMode &&
          (isAddMode ||
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
      hoverCursor: !workflow.navigateMode && !samBoxActive && showHoverCursor,
      cursorMode: !workflow.navigateMode && workflow.targetCursorActive ? "target" : undefined,
      hoverBadge: workflow.navigateMode
        ? undefined
        : { point: segments.hoverPoint, count: segments.hoverCount },
    },
  };
}
