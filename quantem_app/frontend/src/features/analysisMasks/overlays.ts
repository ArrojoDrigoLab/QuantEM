import type { Point } from "@/utils/geometry";
import type { SegmentOverlay } from "@/viewer/types";
import type { AnalysisMaskObject } from "./types";

function points(ring: number[][]): Point[] {
  return ring.flatMap((coordinate) =>
    coordinate.length >= 2
      ? [{ x: Number(coordinate[0]), y: Number(coordinate[1]) }]
      : []
  );
}

function polygonOverlays(
  object: AnalysisMaskObject,
  polygons: number[][][][],
  active: boolean
): SegmentOverlay[] {
  return polygons.flatMap((rings, index) => {
    const exterior = rings[0] ? points(rings[0]) : [];
    if (exterior.length < 4) return [];
    const holes = rings.slice(1).map(points).filter((ring) => ring.length >= 4);
    return [
      {
        id: `analysis-mask-object-${object.id}-${index}`,
        geometry: exterior,
        ...(holes.length > 0 ? { holes } : {}),
        fillColor: object.color,
        fillOpacity: 0.1,
        strokeColor: object.color,
        strokeOpacity: 0.95,
        strokeWidth: active ? 3 : 2,
      },
    ];
  });
}

export function analysisMaskObjectOverlays(
  object: AnalysisMaskObject,
  active: boolean = false
): SegmentOverlay[] {
  if (!object.geometry) return [];
  const polygons =
    object.geometry.type === "Polygon"
      ? [object.geometry.coordinates]
      : object.geometry.coordinates;
  return polygonOverlays(object, polygons, active);
}

export function analysisMaskObjectsOverlays(
  objects: AnalysisMaskObject[],
  activeObjectId: string | null
): SegmentOverlay[] {
  return objects.flatMap((object) =>
    analysisMaskObjectOverlays(object, object.id === activeObjectId)
  );
}
