import type { BBox } from "@/shared/types/common";
import type { ViewerFitBounds } from "@/viewer/types";

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function bboxToFitBounds(bbox: BBox): ViewerFitBounds {
  return {
    x: bbox.x0,
    y: bbox.y0,
    width: Math.max(1, bbox.x1 - bbox.x0),
    height: Math.max(1, bbox.y1 - bbox.y0),
  };
}
