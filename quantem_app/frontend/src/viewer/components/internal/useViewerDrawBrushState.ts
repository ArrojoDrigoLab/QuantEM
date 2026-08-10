import { useCallback, useMemo, useState } from "react";
import type { Point } from "@/utils/geometry";
import type { SegmentOverlay } from "@/viewer/types";

function toClosedGeometry(points: Point[]): Point[] {
  if (points.length < 2) return points;
  const first = points[0];
  const last = points[points.length - 1];
  if (!first || !last) return points;
  if (first.x === last.x && first.y === last.y) return points;
  return [...points, first];
}

export function useViewerDrawBrushState(config: {
  drawMode: boolean;
  brushMode: boolean;
  brushSize: number;
  brushColor: string;
  onDrawComplete?: (points: Point[]) => void;
  onBrushStroke?: (points: Point[]) => void;
}) {
  const { drawMode, brushMode, brushSize, brushColor, onDrawComplete, onBrushStroke } = config;
  const [drawPoints, setDrawPoints] = useState<Point[]>([]);
  const [drawPreviewPoint, setDrawPreviewPoint] = useState<Point | null>(null);
  const [brushPoints, setBrushPoints] = useState<Point[]>([]);
  const [brushPreviewPoint, setBrushPreviewPoint] = useState<Point | null>(null);

  const addBrushPoint = useCallback(
    (nextPoint: Point, points: Point[]) => {
      const last = points[points.length - 1];
      if (!last) return [...points, nextPoint];
      const minDistance = Math.max(brushSize * 0.25, 2);
      const dx = nextPoint.x - last.x;
      const dy = nextPoint.y - last.y;
      if (Math.hypot(dx, dy) < minDistance) return points;
      return [...points, nextPoint];
    },
    [brushSize]
  );

  const drawPreviewOverlay = useMemo(() => {
    if (!drawMode || drawPoints.length === 0 || !drawPreviewPoint) return null;
    const geometry = [...drawPoints, drawPreviewPoint];
    return {
      id: "draw-preview",
      geometry,
      shape: "polyline" as const,
      fillColor: "transparent",
      fillOpacity: 0,
      strokeColor: "#ff0000",
      strokeOpacity: 0.9,
      strokeWidth: 2,
    } satisfies SegmentOverlay;
  }, [drawMode, drawPoints, drawPreviewPoint]);

  const brushPreviewOverlay = useMemo(() => {
    if (!brushMode || brushPoints.length === 0) return null;
    const geometry = brushPreviewPoint ? [...brushPoints, brushPreviewPoint] : [...brushPoints];
    return {
      id: "brush-preview",
      geometry,
      shape: "polyline" as const,
      fillColor: "transparent",
      fillOpacity: 0,
      strokeColor: brushColor,
      strokeOpacity: 0.6,
      strokeWidth: Math.max(brushSize, 1),
    } satisfies SegmentOverlay;
  }, [brushMode, brushColor, brushPoints, brushPreviewPoint, brushSize]);

  return {
    drawPoints,
    setDrawPoints,
    drawPreviewPoint,
    setDrawPreviewPoint,
    brushPoints,
    setBrushPoints,
    brushPreviewPoint,
    setBrushPreviewPoint,
    addBrushPoint,
    drawPreviewOverlay,
    brushPreviewOverlay,
    completeDraw(imagePoint: Point) {
      if (drawPoints.length === 0) {
        setDrawPoints([imagePoint]);
      } else {
        onDrawComplete?.(toClosedGeometry([...drawPoints, imagePoint]));
        setDrawPoints([]);
        setDrawPreviewPoint(null);
      }
    },
    startBrushStroke(imagePoint: Point) {
      setBrushPoints([imagePoint]);
      setBrushPreviewPoint(imagePoint);
    },
    appendBrushStroke(imagePoint: Point) {
      setBrushPoints((prev) => addBrushPoint(imagePoint, prev));
      setBrushPreviewPoint(imagePoint);
    },
    finishBrushStroke() {
      setBrushPoints((prev) => {
        if (prev.length > 0) {
          onBrushStroke?.(prev);
        }
        return [];
      });
      setBrushPreviewPoint(null);
    },
  };
}

