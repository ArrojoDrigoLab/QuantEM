import type { BBox } from "@/shared/types/common";
import type { Point } from "@/utils/geometry";
import { buildClosedPolygonRing } from "@/shared/geometry/draftGeometry/polygons";
import { cutClosedRingWithLasso } from "@/shared/geometry/draftGeometry/cut";
import {
  EDGE_EPSILON,
} from "@/shared/geometry/draftGeometry/shared";
import type { DraftPolygon } from "@/shared/geometry/draftGeometry/types";

function patchBBoxToRing(patchBBox: BBox): Point[] {
  return [
    { x: patchBBox.x0, y: patchBBox.y0 },
    { x: patchBBox.x1, y: patchBBox.y0 },
    { x: patchBBox.x1, y: patchBBox.y1 },
    { x: patchBBox.x0, y: patchBBox.y1 },
    { x: patchBBox.x0, y: patchBBox.y0 },
  ];
}

export function pointInsidePatch(point: Point, patchBBox: BBox) {
  return (
    point.x >= patchBBox.x0 - EDGE_EPSILON &&
    point.x <= patchBBox.x1 + EDGE_EPSILON &&
    point.y >= patchBBox.y0 - EDGE_EPSILON &&
    point.y <= patchBBox.y1 + EDGE_EPSILON
  );
}

export function normalizePatchBBox(patchBBox: BBox): BBox | null {
  const x0 = Math.min(patchBBox.x0, patchBBox.x1);
  const y0 = Math.min(patchBBox.y0, patchBBox.y1);
  const x1 = Math.max(patchBBox.x0, patchBBox.x1);
  const y1 = Math.max(patchBBox.y0, patchBBox.y1);
  if (x1 - x0 <= EDGE_EPSILON || y1 - y0 <= EDGE_EPSILON) {
    return null;
  }
  return { x0, y0, x1, y1 };
}

export function buildPatchBBoxFromPoints(start: Point, end: Point): BBox | null {
  return normalizePatchBBox({
    x0: start.x,
    y0: start.y,
    x1: end.x,
    y1: end.y,
  });
}

export function patchBBoxIntersectsImage(
  patchBBox: BBox,
  imageWidth: number,
  imageHeight: number
) {
  return (
    patchBBox.x0 >= 0 - EDGE_EPSILON &&
    patchBBox.y0 >= 0 - EDGE_EPSILON &&
    patchBBox.x1 <= imageWidth + EDGE_EPSILON &&
    patchBBox.y1 <= imageHeight + EDGE_EPSILON
  );
}

export function patchBBoxWithinLimit(patchBBox: BBox, limitPx: number) {
  return (
    patchBBox.x1 - patchBBox.x0 <= limitPx + EDGE_EPSILON &&
    patchBBox.y1 - patchBBox.y0 <= limitPx + EDGE_EPSILON
  );
}

export function validateAutomaticPatchSelection(
  polygon: DraftPolygon,
  patchBBox: BBox
): string | null {
  const ring = buildClosedPolygonRing(polygon);
  if (!ring) {
    return "Patch refine requires one closed polygon.";
  }
  const cutResult = cutClosedRingWithLasso(ring, patchBBoxToRing(patchBBox));
  if (cutResult.kind === "untouched") {
    return "The selected patch must intersect the current polygon boundary.";
  }
  if (cutResult.kind === "error") {
    return cutResult.message;
  }
  return null;
}

