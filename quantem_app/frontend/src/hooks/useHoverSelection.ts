/**
 * Hover-selection state driven by backend point queries.
 */

import { useState, useMemo, useCallback } from "react";
import type { Point } from "@/utils/geometry";
import type { SegmentObject } from "@/shared/types";

export type GroupHoverActionMode = "group-confirm" | "group-reject";
export type HoverActionMode =
  | "confirm"
  | "reject"
  | GroupHoverActionMode
  | "test";

function sortHoverSegments(segments: SegmentObject[]): SegmentObject[] {
  return [...segments].sort((a, b) => {
    if (a.confidence_score !== null && b.confidence_score !== null) {
      return b.confidence_score - a.confidence_score;
    }
    if (a.confidence_score !== null) return -1;
    if (b.confidence_score !== null) return 1;
    return a.id.localeCompare(b.id);
  });
}

export function useHoverSelection() {
  const [hoverSegments, setHoverSegments] = useState<SegmentObject[]>([]);
  const [hoverIndex, setHoverIndex] = useState<number>(0);
  const [hoverActionMode, setHoverActionMode] = useState<HoverActionMode>("confirm");
  const [hoverPoint, setHoverPoint] = useState<Point | null>(null);

  const highlightedSegmentId = useMemo(() => {
    if (hoverIndex < 0 || hoverIndex >= hoverSegments.length) {
      return null;
    }
    return hoverSegments[hoverIndex]?.id ?? null;
  }, [hoverIndex, hoverSegments]);

  const findSegmentsAtPoint = useCallback((point: Point, segments: SegmentObject[]) => {
    if (!segments || segments.length === 0) {
      setHoverSegments([]);
      setHoverIndex(0);
      setHoverPoint(null);
      return;
    }
    setHoverSegments(sortHoverSegments(segments));
    setHoverIndex(0);
    setHoverPoint(point);
  }, []);

  const cycleHoverIndex = useCallback(
    (direction: "next" | "prev") => {
      setHoverIndex((prev) => {
        if (!hoverSegments.length) return 0;
        return direction === "next"
          ? (prev + 1) % hoverSegments.length
          : (prev - 1 + hoverSegments.length) % hoverSegments.length;
      });
    },
    [hoverSegments.length]
  );

  const clearHover = useCallback(() => {
    setHoverSegments((prev) => (prev.length ? [] : prev));
    setHoverIndex((prev) => (prev !== 0 ? 0 : prev));
    setHoverPoint((prev) => (prev ? null : prev));
  }, []);

  return {
    hoverSegments,
    hoverIndex,
    highlightedSegmentId,
    hoverActionMode,
    hoverPoint,
    setHoverActionMode,
    findSegmentsAtPoint,
    cycleHoverIndex,
    clearHover,
  };
}
