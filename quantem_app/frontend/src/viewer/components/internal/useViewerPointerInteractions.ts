import { useCallback, useEffect, useRef } from "react";
import { findSceneOverlayIdAtPoint } from "@/viewer/overlays/hitTest";
import type { Point } from "@/utils/geometry";
import type { ViewMetrics } from "@/viewer/components/internal/viewerMath";
import { screenToImagePoint } from "@/viewer/components/internal/viewerMath";
import type { ViewportState } from "@/viewer/types";

const MOUSE_MOVE_THROTTLE_MS = 60;
const CLICK_MOVE_TOLERANCE_PX = 4;

export function useViewerPointerInteractions(config: {
  containerRef: React.RefObject<HTMLDivElement | null>;
  interactionLayerRef: React.RefObject<HTMLDivElement | null>;
  metrics: ViewMetrics | null;
  localViewport: ViewportState | null;
  disablePan: boolean;
  resolvedImageWidth: number;
  setViewport: (nextViewport: ViewportState, emit?: boolean) => void;
  onImageClick?: (point: Point) => void;
  onImagePress?: (point: Point, screenPoint: Point) => void;
  onImageDrag?: (point: Point, screenPoint: Point) => void;
  onImageRelease?: (point: Point, screenPoint: Point) => void;
  onImageMove?: (point: Point, screenPoint: Point) => void;
  onImageMouseMove?: (point: Point) => void;
  onImageMouseLeave?: () => void;
  onShapeClick?: (segmentId: string | null) => void;
  /** Resolve the object id under a screen point from the pickable ID-map raster. */
  pickRasterObjectId?: (screenPoint: Point) => string | null;
  overlayScene: { persistent: import("@/viewer/types").SegmentOverlay[]; transient: import("@/viewer/types").SegmentOverlay[] };
  drawMode: boolean;
  brushMode: boolean;
  drawState: ReturnType<typeof import("@/viewer/components/internal/useViewerDrawBrushState").useViewerDrawBrushState>;
  cursorState: ReturnType<typeof import("@/viewer/components/internal/useViewerCursorState").useViewerCursorState>;
}) {
  const {
    containerRef,
    interactionLayerRef,
    metrics,
    localViewport,
    disablePan,
    resolvedImageWidth,
    setViewport,
    onImageClick,
    onImagePress,
    onImageDrag,
    onImageRelease,
    onImageMove,
    onImageMouseMove,
    onImageMouseLeave,
    onShapeClick,
    pickRasterObjectId,
    overlayScene,
    drawMode,
    brushMode,
    drawState,
    cursorState,
  } = config;
  const activePointerIdRef = useRef<number | null>(null);
  const isPanningRef = useRef(false);
  const lastPointerScreenRef = useRef<Point | null>(null);
  const pointerDownStartRef = useRef<Point | null>(null);
  const movedSincePointerDownRef = useRef(false);
  const isBrushingRef = useRef(false);
  const lastMouseMoveTsRef = useRef(0);
  const wheelStateRef = useRef({
    metrics,
    localViewport,
    resolvedImageWidth,
  });

  useEffect(() => {
    wheelStateRef.current = {
      metrics,
      localViewport,
      resolvedImageWidth,
    };
  }, [localViewport, metrics, resolvedImageWidth]);

  const screenPointFromMouseEvent = useCallback((event: MouseEvent | PointerEvent): Point | null => {
    const container = containerRef.current;
    if (!container) return null;
    const rect = container.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }, [containerRef]);

  const toImagePoint = useCallback(
    (screenPoint: Point): Point | null => {
      if (!metrics) return null;
      return screenToImagePoint(metrics, screenPoint);
    },
    [metrics]
  );

  const maybeEmitMouseMove = useCallback(
    (imagePoint: Point) => {
      const now = Date.now();
      if (now - lastMouseMoveTsRef.current < MOUSE_MOVE_THROTTLE_MS) return;
      lastMouseMoveTsRef.current = now;
      onImageMouseMove?.(imagePoint);
    },
    [onImageMouseMove]
  );

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!metrics) return;
      const screenPoint = screenPointFromMouseEvent(event.nativeEvent);
      if (!screenPoint) return;
      const imagePoint = toImagePoint(screenPoint);
      if (!imagePoint) return;

      cursorState.setIsPointerInside(true);
      activePointerIdRef.current = event.pointerId;
      pointerDownStartRef.current = screenPoint;
      lastPointerScreenRef.current = screenPoint;
      movedSincePointerDownRef.current = false;

      if (brushMode) {
        isBrushingRef.current = true;
        drawState.startBrushStroke(imagePoint);
      } else {
        isPanningRef.current = !disablePan;
        onImagePress?.(imagePoint, screenPoint);
      }

      maybeEmitMouseMove(imagePoint);
      cursorState.updateOverlayCursor(screenPoint, imagePoint);
      (event.target as Element).setPointerCapture(event.pointerId);
    },
    [
      brushMode,
      cursorState,
      disablePan,
      drawState,
      maybeEmitMouseMove,
      metrics,
      onImagePress,
      screenPointFromMouseEvent,
      toImagePoint,
    ]
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!metrics) return;
      const screenPoint = screenPointFromMouseEvent(event.nativeEvent);
      if (!screenPoint) return;
      const imagePoint = toImagePoint(screenPoint);
      if (!imagePoint) return;

      cursorState.setIsPointerInside(true);
      cursorState.setLastMouseScreen(screenPoint);
      onImageMove?.(imagePoint, screenPoint);
      maybeEmitMouseMove(imagePoint);
      cursorState.updateOverlayCursor(screenPoint, imagePoint);

      if (drawMode && drawState.drawPoints.length > 0) {
        drawState.setDrawPreviewPoint(imagePoint);
      }

      if (activePointerIdRef.current !== event.pointerId) return;
      const previousScreen = lastPointerScreenRef.current;
      lastPointerScreenRef.current = screenPoint;
      if (!previousScreen) return;

      const dx = screenPoint.x - previousScreen.x;
      const dy = screenPoint.y - previousScreen.y;
      const downStart = pointerDownStartRef.current;
      if (downStart) {
        const dragDistance = Math.hypot(
          screenPoint.x - downStart.x,
          screenPoint.y - downStart.y
        );
        if (dragDistance >= CLICK_MOVE_TOLERANCE_PX) {
          movedSincePointerDownRef.current = true;
        }
      }

      if (brushMode && isBrushingRef.current) {
        drawState.appendBrushStroke(imagePoint);
        onImageDrag?.(imagePoint, screenPoint);
        return;
      }

      onImageDrag?.(imagePoint, screenPoint);
      if (!isPanningRef.current || disablePan || !localViewport) return;

      const panX = (dx / metrics.containerWidth) * metrics.visibleWidth;
      const panY = (dy / metrics.containerHeight) * metrics.visibleHeight;
      setViewport(
        {
          ...localViewport,
          centerX: localViewport.centerX - panX / resolvedImageWidth,
          centerY: localViewport.centerY - panY / resolvedImageWidth,
        },
        true
      );
    },
    [
      brushMode,
      cursorState,
      disablePan,
      drawMode,
      drawState,
      localViewport,
      maybeEmitMouseMove,
      metrics,
      onImageDrag,
      onImageMove,
      resolvedImageWidth,
      screenPointFromMouseEvent,
      setViewport,
      toImagePoint,
    ]
  );

  const handlePointerUp = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!metrics || activePointerIdRef.current !== event.pointerId) return;
      const screenPoint = screenPointFromMouseEvent(event.nativeEvent);
      const imagePoint = screenPoint ? toImagePoint(screenPoint) : null;

      if (brushMode && isBrushingRef.current) {
        isBrushingRef.current = false;
        drawState.finishBrushStroke();
      } else if (screenPoint && imagePoint) {
        onImageRelease?.(imagePoint, screenPoint);
      }

      activePointerIdRef.current = null;
      isPanningRef.current = false;
      lastPointerScreenRef.current = null;
    },
    [brushMode, drawState, metrics, onImageRelease, screenPointFromMouseEvent, toImagePoint]
  );

  const handleMouseLeave = useCallback(() => {
    cursorState.setIsPointerInside(false);
    cursorState.setLastMouseScreen(null);
    onImageMouseLeave?.();
  }, [cursorState, onImageMouseLeave]);

  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!metrics || brushMode) return;
      const downStart = pointerDownStartRef.current;
      if (downStart && movedSincePointerDownRef.current) {
        pointerDownStartRef.current = null;
        return;
      }
      pointerDownStartRef.current = null;

      const screenPoint = screenPointFromMouseEvent(event.nativeEvent);
      if (!screenPoint) return;
      const imagePoint = toImagePoint(screenPoint);
      if (!imagePoint) return;

      if (onShapeClick) {
        const clickedSegmentId = findSceneOverlayIdAtPoint(imagePoint, overlayScene);
        if (clickedSegmentId) {
          onShapeClick(clickedSegmentId);
          return;
        }
        // No vector hit: fall back to the pickable ID-map raster in review/select
        // modes (brush mode already returned above; skip while drawing) so any
        // object stays selectable straight off the overlay -- the "object level"
        // of the raster<->object swap, with no vectors rendered.
        if (!drawMode) {
          const rasterSegmentId = pickRasterObjectId?.(screenPoint) ?? null;
          if (rasterSegmentId) {
            onShapeClick(rasterSegmentId);
            return;
          }
        }
        onShapeClick(null);
      }

      if (drawMode) {
        drawState.completeDraw(imagePoint);
        return;
      }

      onImageClick?.(imagePoint);
    },
    [
      brushMode,
      drawMode,
      drawState,
      metrics,
      onImageClick,
      onShapeClick,
      pickRasterObjectId,
      overlayScene,
      screenPointFromMouseEvent,
      toImagePoint,
    ]
  );

  const handleDoubleClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!metrics || !localViewport) return;
      event.preventDefault();
      const screenPoint = screenPointFromMouseEvent(event.nativeEvent);
      if (!screenPoint) return;
      const imagePoint = toImagePoint(screenPoint);
      if (!imagePoint) return;

      const nextZoom = localViewport.zoom * 2;
      const nextVisibleWidth = resolvedImageWidth / nextZoom;
      const nextVisibleHeight = nextVisibleWidth / (metrics.containerWidth / metrics.containerHeight);
      const nextCenterX =
        imagePoint.x + (0.5 - screenPoint.x / metrics.containerWidth) * nextVisibleWidth;
      const nextCenterY =
        imagePoint.y + (0.5 - screenPoint.y / metrics.containerHeight) * nextVisibleHeight;

      setViewport(
        {
          ...localViewport,
          centerX: nextCenterX / resolvedImageWidth,
          centerY: nextCenterY / resolvedImageWidth,
          zoom: nextZoom,
        },
        true
      );
    },
    [localViewport, metrics, resolvedImageWidth, screenPointFromMouseEvent, setViewport, toImagePoint]
  );

  const handleWheel = useCallback(
    (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      const { metrics, localViewport, resolvedImageWidth } = wheelStateRef.current;
      if (!metrics || !localViewport) return;
      const screenPoint = screenPointFromMouseEvent(event);
      if (!screenPoint) return;
      const imagePoint = screenToImagePoint(metrics, screenPoint);
      if (!imagePoint) return;

      const zoomFactor = Math.exp(-event.deltaY * 0.0015);
      const nextZoom = Math.max(0.05, Math.min(200, localViewport.zoom * zoomFactor));
      const nextVisibleWidth = resolvedImageWidth / nextZoom;
      const nextVisibleHeight = nextVisibleWidth / (metrics.containerWidth / metrics.containerHeight);
      const nextCenterX =
        imagePoint.x + (0.5 - screenPoint.x / metrics.containerWidth) * nextVisibleWidth;
      const nextCenterY =
        imagePoint.y + (0.5 - screenPoint.y / metrics.containerHeight) * nextVisibleHeight;

      setViewport(
        {
          ...localViewport,
          centerX: nextCenterX / resolvedImageWidth,
          centerY: nextCenterY / resolvedImageWidth,
          zoom: nextZoom,
        },
        true
      );
    },
    [screenPointFromMouseEvent, setViewport]
  );

  useEffect(() => {
    const interactionLayer = interactionLayerRef.current;
    if (!interactionLayer) return;

    // React's synthetic wheel listeners are passive, so use a native listener
    // for viewer zoom to keep preventDefault() valid and suppress page scroll.
    interactionLayer.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      interactionLayer.removeEventListener("wheel", handleWheel);
    };
  }, [handleWheel, interactionLayerRef]);

  return {
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handleMouseLeave,
    handleClick,
    handleDoubleClick,
  };
}
