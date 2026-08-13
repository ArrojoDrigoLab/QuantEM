import type { SegmentOverlay } from "@/viewer/types";
import type { OverlayScene } from "@/viewer/overlays/types";
import { pointInPolygon, type Point } from "@/utils/geometry";

export function findOverlayIdAtPoint(
  point: Point,
  overlays: SegmentOverlay[]
): string | null {
  for (const overlay of overlays) {
    if (
      pointInPolygon(point, overlay.geometry) &&
      !(overlay.holes ?? []).some((hole) => pointInPolygon(point, hole))
    ) {
      return overlay.id;
    }
  }
  return null;
}

export function findSceneOverlayIdAtPoint(
  point: Point,
  scene: OverlayScene
): string | null {
  return findOverlayIdAtPoint(point, [...scene.persistent, ...scene.transient]);
}
