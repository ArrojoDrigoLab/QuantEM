import { useCallback, useEffect, useRef, useState } from "react";
import { findSceneOverlayIdAtPoint } from "@/viewer/overlays/hitTest";
import type { Point } from "@/utils/geometry";
import type { ViewMetrics } from "@/viewer/components/internal/viewerMath";
import {
  clampViewportToImage,
  screenToImagePoint,
} from "@/viewer/components/internal/viewerMath";
import { panKeyState } from "@/viewer/panKeyState";
import type { ViewportState } from "@/viewer/types";

const MOUSE_MOVE_THROTTLE_MS = 60;
const CLICK_MOVE_TOLERANCE_PX = 4;
const MIDDLE_BUTTON = 1;

export function useViewerPointerInteractions(config: {
  containerRef: React.RefObject<HTMLDivElement | null>;
  interactionLayerRef: React.RefObject<HTMLDivElement | null>;
  metrics: ViewMetrics | null;
  localViewport: ViewportState | null;
  disablePan: boolean;
  resolvedImageWidth: number;
  resolvedImageHeight: number;
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
    resolvedImageHeight,
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
  const panStartedWithPanKeyRef = useRef(false);
  const lastPointerScreenRef = useRef<Point | null>(null);
  const pointerDownStartRef = useRef<Point | null>(null);
  const movedSincePointerDownRef = useRef(false);
  const isBrushingRef = useRef(false);
  const lastMouseMoveTsRef = useRef(0);
  const [panKeyHeld, setPanKeyHeld] = useState(false);
  const wheelStateRef = useRef({
    metrics,
    localViewport,
    resolvedImageWidth,
    resolvedImageHeight,
  });

  useEffect(() => {
    wheelStateRef.current = {
      metrics,
      localViewport,
      resolvedImageWidth,
      resolvedImageHeight,
    };
  }, [localViewport, metrics, resolvedImageHeight, resolvedImageWidth]);

  useEffect(() => panKeyState.subscribe(setPanKeyHeld), []);

  /**
   * Who owns a plain left-button drag.
   *
   * The canvas reform's rule is that clicking the image does what the active
   * tool says, so the left button cannot also be the pan gesture whenever a
   * tool is armed: on the labeling screen the first stroke of a correction used
   * to slide the image instead of drawing. Pan therefore moves to the two
   * gestures nothing else claims -- hold space and drag, or drag with the
   * middle button.
   *
   * Where nothing is armed there is no competition, and taking left-drag away
   * would be a pure loss: the plain viewer and Navigate mode both pass no click
   * or press handler at all, and dragging the image is the only thing a left
   * drag could sensibly mean there. So the gesture is decided by whether a tool
   * is listening, not by a flag a caller has to remember to set.
   */
  const toolOwnsLeftButton = Boolean(
    onImageClick || onImagePress || onShapeClick || drawMode || brushMode
  );

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

      // A pan gesture is exclusive: it must not also start a brush stroke or a
      // group-selection box, or holding space to reposition the image would
      // paint a line across whatever it passed over.
      const panGesture =
        !disablePan &&
        (panKeyState.isPanKeyHeld() ||
          event.button === MIDDLE_BUTTON ||
          !toolOwnsLeftButton);
      panStartedWithPanKeyRef.current = panGesture && panKeyState.isPanKeyHeld();

      if (panGesture) {
        isPanningRef.current = true;
      } else if (brushMode) {
        isBrushingRef.current = true;
        drawState.startBrushStroke(imagePoint);
      } else {
        isPanningRef.current = false;
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
      toolOwnsLeftButton,
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

      if (isPanningRef.current) {
        if (disablePan || !localViewport) return;
        const panX = (dx / metrics.containerWidth) * metrics.visibleWidth;
        const panY = (dy / metrics.containerHeight) * metrics.visibleHeight;
        if (panStartedWithPanKeyRef.current && (dx !== 0 || dy !== 0)) {
          // Tell the keyboard layer this space press was a pan, so releasing
          // the key does not also keep the object under the cursor.
          panKeyState.markSpacePan();
        }
        setViewport(
          clampViewportToImage(
            {
              ...localViewport,
              centerX: localViewport.centerX - panX / resolvedImageWidth,
              centerY: localViewport.centerY - panY / resolvedImageWidth,
            },
            resolvedImageWidth,
            resolvedImageHeight
          ),
          true
        );
        return;
      }

      if (brushMode && isBrushingRef.current) {
        drawState.appendBrushStroke(imagePoint);
        onImageDrag?.(imagePoint, screenPoint);
        return;
      }

      onImageDrag?.(imagePoint, screenPoint);
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
      resolvedImageHeight,
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

      if (isPanningRef.current) {
        // A pan claimed this gesture at pointer-down; no tool ever saw it.
      } else if (brushMode && isBrushingRef.current) {
        isBrushingRef.current = false;
        drawState.finishBrushStroke();
      } else if (screenPoint && imagePoint) {
        onImageRelease?.(imagePoint, screenPoint);
      }

      activePointerIdRef.current = null;
      isPanningRef.current = false;
      panStartedWithPanKeyRef.current = false;
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
      // Space is held: this gesture belonged to the pan, and a keep must not
      // also fire off the same press.
      if (panKeyState.isPanKeyHeld()) {
        pointerDownStartRef.current = null;
        return;
      }
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
        clampViewportToImage(
          {
            ...localViewport,
            centerX: nextCenterX / resolvedImageWidth,
            centerY: nextCenterY / resolvedImageWidth,
            zoom: nextZoom,
          },
          resolvedImageWidth,
          resolvedImageHeight
        ),
        true
      );
    },
    [
      localViewport,
      metrics,
      resolvedImageHeight,
      resolvedImageWidth,
      screenPointFromMouseEvent,
      setViewport,
      toImagePoint,
    ]
  );

  const handleWheel = useCallback(
    (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      const { metrics, localViewport, resolvedImageWidth, resolvedImageHeight } =
        wheelStateRef.current;
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
        clampViewportToImage(
          {
            ...localViewport,
            centerX: nextCenterX / resolvedImageWidth,
            centerY: nextCenterY / resolvedImageWidth,
            zoom: nextZoom,
          },
          resolvedImageWidth,
          resolvedImageHeight
        ),
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
    /** Space is down: the canvas shows a grab cursor and tools stand aside. */
    panKeyHeld,
    /** Whether a plain left drag pans, for the cursor and for tests. */
    leftDragPans: !toolOwnsLeftButton,
  };
}
