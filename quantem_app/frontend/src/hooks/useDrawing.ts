/**
 * Hook for managing freehand drawing state.
 */

import { useState, useCallback } from "react";
import type { Point } from "@/utils/geometry";
import { brushStrokesToConnectedPolygonRings } from "@/utils/brushMask";

export type DraftOperation = "include" | "exclude";

export interface BrushStroke {
  id: string;
  label: number;
  size: number;
  points: Point[];
  operation: DraftOperation;
}

export function useDrawing() {
  const [pendingPolygon, setPendingPolygon] = useState<Point[] | null>(null);
  const [brushSize, setBrushSize] = useState(24);
  const [brushStrokes, setBrushStrokes] = useState<BrushStroke[]>([]);
  const [draftOperation, setDraftOperation] = useState<DraftOperation>("include");
  const [pendingPolygonOperation, setPendingPolygonOperation] =
    useState<DraftOperation>("include");

  const handleDrawComplete = useCallback((points: Point[]) => {
    // Close polygon by adding first point if not already closed
    if (points.length > 0 && (points[0].x !== points[points.length - 1].x || points[0].y !== points[points.length - 1].y)) {
      setPendingPolygon([...points, points[0]]);
    } else {
      setPendingPolygon(points);
    }
    setPendingPolygonOperation(draftOperation);
  }, [draftOperation]);

  const handleBrushStroke = useCallback(
    (points: Point[]) => {
      if (!points || points.length === 0) return;
      const id = `stroke-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      setBrushStrokes((prev) => [
        ...prev,
        {
          id,
          label: draftOperation === "include" ? 1 : 0,
          size: brushSize,
          points,
          operation: draftOperation,
        },
      ]);
    },
    [brushSize, draftOperation]
  );

  const getBrushPolygonRings = useCallback(() => {
    return (["include", "exclude"] as const).flatMap((operation) =>
      brushStrokesToConnectedPolygonRings(
        brushStrokes.filter(
          (stroke) =>
            (stroke.operation ?? (stroke.label === 0 ? "exclude" : "include")) ===
            operation
        )
      ).map((polygon) => ({ ...polygon, operation }))
    );
  }, [brushStrokes]);

  const getBrushPolygons = useCallback(
    () => getBrushPolygonRings().map((polygon) => polygon.exterior),
    [getBrushPolygonRings]
  );

  /** Remove any un-committed brush strokes the eraser path passes over. */
  const eraseBrushStrokesAt = useCallback(
    (eraserPoints: Point[], eraserSize: number) => {
      if (!eraserPoints || eraserPoints.length === 0) return;
      setBrushStrokes((prev) =>
        prev.filter((stroke) => {
          const radius = (stroke.size + eraserSize) / 2;
          const radiusSq = radius * radius;
          return !stroke.points.some((sp) =>
            eraserPoints.some((ep) => {
              const dx = sp.x - ep.x;
              const dy = sp.y - ep.y;
              return dx * dx + dy * dy <= radiusSq;
            })
          );
        })
      );
    },
    []
  );

  const clearDrawing = useCallback(() => {
    setPendingPolygon(null);
    setBrushStrokes([]);
  }, []);

  return {
    pendingPolygon,
    pendingPolygonOperation,
    brushSize,
    setBrushSize,
    brushStrokes,
    draftOperation,
    setDraftOperation,
    handleDrawComplete,
    handleBrushStroke,
    getBrushPolygons,
    getBrushPolygonRings,
    eraseBrushStrokesAt,
    clearDrawing,
  };
}
