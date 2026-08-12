import { useCallback, useMemo, useState } from "react";
import type { Point } from "@/utils/geometry";
import type { ViewMetrics } from "@/viewer/components/internal/viewerMath";

interface OverlayCursorState {
  x: number;
  y: number;
  outerSize: number;
  borderWidth: number;
  visible: boolean;
  variant: "brush" | "target";
}

// The cursor's opening represents the brush diameter in image space. Keep the
// ring itself a constant screen-space width, so a small brush at low zoom stays
// small instead of becoming a thick, fixed-diameter circle.
const BRUSH_CURSOR_BORDER_WIDTH = 2;
const MIN_BRUSH_CURSOR_OPENING = 1;
const TARGET_CURSOR_OUTER_SIZE = 20;
const TARGET_CURSOR_BORDER_WIDTH = 2;

export function useViewerCursorState(config: {
  brushMode: boolean;
  brushSize: number;
  cursorMode?: "target";
  hoverBadge?: { point: Point | null; count: number };
  metrics: ViewMetrics | null;
}) {
  const { brushMode, brushSize, cursorMode, hoverBadge, metrics } = config;
  const [lastMouseScreen, setLastMouseScreen] = useState<Point | null>(null);
  const [isPointerInside, setIsPointerInside] = useState(false);
  const [overlayCursorState, setOverlayCursorState] = useState<OverlayCursorState>({
    x: 0,
    y: 0,
    outerSize: MIN_BRUSH_CURSOR_OPENING + BRUSH_CURSOR_BORDER_WIDTH * 2,
    borderWidth: BRUSH_CURSOR_BORDER_WIDTH,
    visible: false,
    variant: "brush",
  });

  const updateOverlayCursor = useCallback(
    (screenPoint: Point, imagePoint: Point) => {
      if (!metrics || (!brushMode && cursorMode !== "target")) {
        setOverlayCursorState((prev) => ({ ...prev, visible: false }));
        return;
      }
      if (cursorMode === "target" && !brushMode) {
        setOverlayCursorState({
          x: screenPoint.x,
          y: screenPoint.y,
          outerSize: TARGET_CURSOR_OUTER_SIZE,
          borderWidth: TARGET_CURSOR_BORDER_WIDTH,
          visible: isPointerInside && Number.isFinite(imagePoint.x) && Number.isFinite(imagePoint.y),
          variant: "target",
        });
        return;
      }
      const rawBrushPixels = (brushSize / metrics.visibleWidth) * metrics.containerWidth;
      // `boxSizing: border-box` on the ring means its CSS width includes the
      // border. Add the two fixed-width borders so the *open* diameter remains
      // exactly the scaled brush diameter. A one-pixel floor is only for the
      // physically unrepresentable case where the brush is sub-pixel on screen.
      const brushSizeInPixels = Math.max(rawBrushPixels, MIN_BRUSH_CURSOR_OPENING);
      const outerSize = brushSizeInPixels + BRUSH_CURSOR_BORDER_WIDTH * 2;
      setOverlayCursorState({
        x: screenPoint.x,
        y: screenPoint.y,
        outerSize,
        borderWidth: BRUSH_CURSOR_BORDER_WIDTH,
        visible: isPointerInside && Number.isFinite(imagePoint.x) && Number.isFinite(imagePoint.y),
        variant: "brush",
      });
    },
    [brushMode, brushSize, cursorMode, isPointerInside, metrics]
  );

  const hoverBadgeStyle = useMemo(() => {
    if (!hoverBadge || hoverBadge.count <= 1 || !lastMouseScreen) {
      return { display: "none" } as const;
    }
    return {
      display: "block",
      left: `${lastMouseScreen.x + 10}px`,
      top: `${lastMouseScreen.y - 10}px`,
    };
  }, [hoverBadge, lastMouseScreen]);

  return {
    lastMouseScreen,
    setLastMouseScreen,
    isPointerInside,
    setIsPointerInside,
    overlayCursorState,
    updateOverlayCursor,
    hoverBadgeStyle,
  };
}
