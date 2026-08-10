import type { BBox } from "@/shared/types/common";
import type { Point } from "@/utils/geometry";

export function circlePoints(center: Point, radius: number, segments: number): Point[] {
  const points: Point[] = [];
  for (let index = 0; index < segments; index += 1) {
    const angle = (index / segments) * Math.PI * 2;
    points.push({
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius,
    });
  }
  points.push(points[0]);
  return points;
}

export function bboxToGeometry(bbox: BBox): Point[] {
  return [
    { x: bbox.x0, y: bbox.y0 },
    { x: bbox.x1, y: bbox.y0 },
    { x: bbox.x1, y: bbox.y1 },
    { x: bbox.x0, y: bbox.y1 },
    { x: bbox.x0, y: bbox.y0 },
  ];
}

export function boundsToGeometry(bounds: {
  x: number;
  y: number;
  width: number;
  height: number;
}): Point[] {
  return [
    { x: bounds.x, y: bounds.y },
    { x: bounds.x + bounds.width, y: bounds.y },
    { x: bounds.x + bounds.width, y: bounds.y + bounds.height },
    { x: bounds.x, y: bounds.y + bounds.height },
    { x: bounds.x, y: bounds.y },
  ];
}
