import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DeckGL, type DeckGLRef } from "@deck.gl/react";
import type { Point } from "@/utils/geometry";
import { OrthographicView } from "@deck.gl/core";
import type { ImageViewerProps } from "@/viewer/imageViewerTypes";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";
import { ViewerCursorOverlay } from "@/viewer/components/internal/ViewerCursorOverlay";
import { ViewerSvgOverlay } from "@/viewer/components/internal/ViewerSvgOverlay";
import { ViewerZSlider } from "@/viewer/components/internal/ViewerZSlider";
import { buildViewerDeckLayers } from "@/viewer/components/internal/buildViewerDeckLayers";
import { getDepthFromSource } from "@/viewer/components/internal/vivUtils";
import { useViewerBaseImageLoader } from "@/viewer/components/internal/useViewerBaseImageLoader";
import { useViewerContainerMetrics } from "@/viewer/components/internal/useViewerContainerMetrics";
import { useViewerCursorState } from "@/viewer/components/internal/useViewerCursorState";
import { useViewerDrawBrushState } from "@/viewer/components/internal/useViewerDrawBrushState";
import { useViewerPointerInteractions } from "@/viewer/components/internal/useViewerPointerInteractions";
import { useViewerRasterOverlayLoader } from "@/viewer/components/internal/useViewerRasterOverlayLoader";
import { useViewerIdMapLoaders } from "@/viewer/components/internal/useViewerIdMapLoader";
import { useViewerViewportState } from "@/viewer/components/internal/useViewerViewportState";
import { sceneToOverlayList } from "@/viewer/overlays/scene";
import "./ImageViewer.css";

const EMPTY_ID_MAP_OVERLAYS: ViewerIdMapOverlaySpec[] = [];

export function ImageViewer({
  image,
  className,
  viewport,
  overlays,
  interactions,
  highlighting,
}: ImageViewerProps) {
  const { containerRef, containerSize } = useViewerContainerMetrics();
  const interactionLayerRef = useRef<HTMLDivElement | null>(null);
  const deckRef = useRef<DeckGLRef | null>(null);
  const { loaderData, inferredSize } = useViewerBaseImageLoader(image.ngffUrl);
  const volumeDepth = useMemo(
    () => (loaderData?.[0] ? getDepthFromSource(loaderData[0]) : 1),
    [loaderData]
  );
  const [zIndex, setZIndex] = useState(0);
  // Reset the z-plane when the underlying image changes.
  useEffect(() => {
    setZIndex(0);
  }, [image.ngffUrl]);
  const clampedZIndex = Math.min(Math.max(0, zIndex), Math.max(0, volumeDepth - 1));
  const persistentOverlays = useMemo(() => overlays?.persistent ?? [], [overlays?.persistent]);
  const transientOverlays = useMemo(() => overlays?.transient ?? [], [overlays?.transient]);
  const overlayNgffLayers = useMemo(() => overlays?.rasterLayers ?? [], [overlays?.rasterLayers]);
  // Stabilize the idMapOverlays reference by CONTENT. The labeling panel rebuilds
  // this wrapper array every render even when the (memoized) specs inside are
  // unchanged. An unstable reference forces the `deckLayers` memo to recompute
  // every render, which rebuilds the base-image MultiscaleImageLayer with a fresh
  // `selections` array -- viv keys `updateTriggers.getTileData` on `selections`,
  // so deck then invalidates and refetches the entire base-image tileset on every
  // incidental re-render (endless base-tile loading). Keeping a stable reference
  // unless an element actually changes prevents that.
  const rawIdMapOverlays = overlays?.idMapOverlays ?? EMPTY_ID_MAP_OVERLAYS;
  const idMapOverlaysRef = useRef<ViewerIdMapOverlaySpec[]>(rawIdMapOverlays);
  if (
    rawIdMapOverlays.length !== idMapOverlaysRef.current.length ||
    rawIdMapOverlays.some((spec, index) => spec !== idMapOverlaysRef.current[index])
  ) {
    idMapOverlaysRef.current = rawIdMapOverlays;
  }
  const idMapOverlays = idMapOverlaysRef.current;
  const bitmapOverlays = useMemo(() => overlays?.bitmapOverlays ?? [], [overlays?.bitmapOverlays]);
  const drawMode = interactions?.draw?.enabled ?? false;
  const brushMode = interactions?.brush?.enabled ?? false;
  const brushSize = interactions?.brush?.size ?? 12;
  const brushColor = interactions?.brush?.color ?? "#33cc66";

  const { overlayLoaderDataByUrl, displayedOverlayNgffLayers } = useViewerRasterOverlayLoader({
    overlayNgffLayers,
    onOverlayRevisionDisplayed: overlays?.onRasterRevisionDisplayed,
  });
  const idMapDataById = useViewerIdMapLoaders(idMapOverlays);
  const resolvedImageWidth = image.width ?? inferredSize?.width ?? 1;
  const resolvedImageHeight = image.height ?? inferredSize?.height ?? 1;
  const viewportState = useViewerViewportState({
    viewportState: viewport?.state,
    initialViewport: viewport?.initialState,
    onViewportChange: viewport?.onChange,
    fitBounds: viewport?.fitBounds ?? null,
    fitBoundsKey: viewport?.fitBoundsKey ?? null,
    fitBoundsPaddingRatio: viewport?.fitBoundsPaddingRatio ?? 0.05,
    containerSize,
    resolvedImageWidth,
    resolvedImageHeight,
  });
  const drawState = useViewerDrawBrushState({
    drawMode,
    brushMode,
    brushSize,
    brushColor,
    onDrawComplete: interactions?.draw?.onComplete,
    onBrushStroke: interactions?.brush?.onStroke,
  });
  const cursorState = useViewerCursorState({
    brushMode,
    brushSize,
    cursorMode: highlighting?.cursorMode,
    hoverBadge: highlighting?.hoverBadge,
    metrics: viewportState.metrics,
  });

  const overlayScene = useMemo(
    () => ({ persistent: persistentOverlays, transient: transientOverlays }),
    [persistentOverlays, transientOverlays]
  );
  const svgTransientOverlays = useMemo(() => {
    const list = [...overlayScene.transient];
    if (drawState.drawPreviewOverlay) list.push(drawState.drawPreviewOverlay);
    if (drawState.brushPreviewOverlay) list.push(drawState.brushPreviewOverlay);
    return list;
  }, [drawState.brushPreviewOverlay, drawState.drawPreviewOverlay, overlayScene.transient]);

  useEffect(() => {
    const duplicateIds = new Set<string>();
    const seenIds = new Set<string>();
    for (const overlay of sceneToOverlayList(overlayScene)) {
      if (seenIds.has(overlay.id)) {
        duplicateIds.add(overlay.id);
      } else {
        seenIds.add(overlay.id);
      }
    }
    if (duplicateIds.size > 0) {
      console.warn(
        "[ImageViewer] Duplicate overlay IDs across persistent/transient layers detected. The first matching ID in scene order will win for click/highlight behavior:",
        Array.from(duplicateIds)
      );
    }
  }, [overlayScene]);

  // Resolve the object under a screen point from the pickable ID-map raster.
  // Lets clicks select individual objects directly off the overlay, with no
  // vector rendering -- the "object level" of the raster<->object swap.
  const pickRasterObjectId = useCallback((screenPoint: Point): string | null => {
    const deck = deckRef.current;
    if (!deck) return null;
    const info = deck.pickObject({ x: screenPoint.x, y: screenPoint.y, radius: 2 });
    const picked = info?.object as { uuid?: string | null } | null | undefined;
    return picked?.uuid ?? null;
  }, []);

  const pointerInteractions = useViewerPointerInteractions({
    containerRef,
    interactionLayerRef,
    pickRasterObjectId,
    metrics: viewportState.metrics,
    localViewport: viewportState.localViewport,
    disablePan: viewport?.disablePan ?? false,
    resolvedImageWidth,
    setViewport: viewportState.setViewport,
    onImageClick: interactions?.onImageClick,
    onImagePress: interactions?.onImagePress,
    onImageDrag: interactions?.onImageDrag,
    onImageRelease: interactions?.onImageRelease,
    onImageMove: interactions?.onImageMove,
    onImageMouseMove: interactions?.onImageMouseMove,
    onImageMouseLeave: interactions?.onImageMouseLeave,
    onShapeClick: interactions?.onShapeClick,
    overlayScene,
    drawMode,
    brushMode,
    drawState,
    cursorState,
  });

  const deckLayers = useMemo(
    () =>
      buildViewerDeckLayers({
        loaderData,
        displayedOverlayNgffLayers,
        overlayLoaderDataByUrl,
        idMapOverlays,
        idMapDataById,
        bitmapOverlays,
        zIndex: clampedZIndex,
      }),
    [
      bitmapOverlays,
      clampedZIndex,
      displayedOverlayNgffLayers,
      idMapDataById,
      idMapOverlays,
      loaderData,
      overlayLoaderDataByUrl,
    ]
  );

  const containerClass = `image-viewer ${className || ""} ${brushMode ? "brush-active" : ""}`.trim();
  const cursorStyle =
    brushMode || highlighting?.cursorMode === "target"
      ? "none"
      : highlighting?.hoverCursor
        ? "pointer"
        : viewport?.disablePan
          ? "crosshair"
          : "";

  return (
    <div ref={containerRef} className={containerClass} style={{ cursor: cursorStyle }}>
      <DeckGL
        ref={deckRef}
        style={{ position: "absolute", inset: "0" }}
        views={new OrthographicView({ id: "viv-main" })}
        viewState={viewportState.deckViewState}
        controller={false}
        layers={deckLayers}
      />
      <div
        ref={interactionLayerRef}
        style={{ position: "absolute", inset: 0, touchAction: "none" }}
        onClick={pointerInteractions.handleClick}
        onDoubleClick={pointerInteractions.handleDoubleClick}
        onPointerDown={pointerInteractions.handlePointerDown}
        onPointerMove={pointerInteractions.handlePointerMove}
        onPointerUp={pointerInteractions.handlePointerUp}
        onPointerCancel={pointerInteractions.handlePointerUp}
        onPointerLeave={pointerInteractions.handleMouseLeave}
      >
        <ViewerSvgOverlay
          metrics={viewportState.metrics}
          persistentOverlays={overlayScene.persistent}
          transientOverlays={svgTransientOverlays}
          highlightedSegmentId={highlighting?.highlightedSegmentId ?? null}
        />
        <ViewerCursorOverlay
          hoverBadge={highlighting?.hoverBadge}
          hoverBadgeStyle={cursorState.hoverBadgeStyle}
          overlayCursorState={cursorState.overlayCursorState}
          brushColor={brushColor}
        />
      </div>
      <ViewerZSlider
        depth={volumeDepth}
        value={clampedZIndex}
        onChange={setZIndex}
        planeIndices={image.zPlaneIndices}
      />
    </div>
  );
}
