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

const BRUSH_CURSOR_MIN_OUTER_SIZE = 20;
const BRUSH_CURSOR_MIN_BORDER_WIDTH = 2;
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
    outerSize: BRUSH_CURSOR_MIN_OUTER_SIZE,
    borderWidth: BRUSH_CURSOR_MIN_BORDER_WIDTH,
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
      const brushSizeInPixels = Math.max(rawBrushPixels, 1);
      let outerSize: number;
      let borderWidth: number;
      if (brushSizeInPixels < BRUSH_CURSOR_MIN_OUTER_SIZE) {
        outerSize = BRUSH_CURSOR_MIN_OUTER_SIZE;
        borderWidth = (BRUSH_CURSOR_MIN_OUTER_SIZE - brushSizeInPixels) / 2;
      } else {
        outerSize = brushSizeInPixels + BRUSH_CURSOR_MIN_BORDER_WIDTH * 2;
        borderWidth = BRUSH_CURSOR_MIN_BORDER_WIDTH;
      }
      setOverlayCursorState({
        x: screenPoint.x,
        y: screenPoint.y,
        outerSize,
        borderWidth,
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

