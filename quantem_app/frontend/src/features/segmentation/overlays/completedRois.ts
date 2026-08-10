import {
  buildClosedPolygonRing,
  type DraftPolygon,
} from "@/shared/geometry/draftGeometry";
import type { CompletedRoi } from "@/shared/types/segmentation";
import type { Point } from "@/utils/geometry";
import type { SegmentOverlay } from "@/viewer/types";

export function generateCompletedRoiOverlays(items: CompletedRoi[]): SegmentOverlay[] {
  return items
    .filter((item) => item.polygon_coords.length >= 4)
    .map((item) => {
      const holes = (item.holes ?? [])
        .filter((ring) => ring.length >= 4)
        .map((ring) => ring.map(([x, y]) => ({ x, y })));
      return {
        id: `completed-roi-${item.id}`,
        geometry: item.polygon_coords.map(([x, y]) => ({ x, y })),
        ...(holes.length > 0 ? { holes } : {}),
        fillColor: "#f59e0b",
        fillOpacity: 0.08,
        strokeColor: "#f59e0b",
        strokeOpacity: 0.95,
        strokeWidth: 2.25,
      };
    });
}

interface CompletedRoiDraftColors {
  strokeColor: string;
  fillColor: string;
}

const COMPLETED_ROI_DRAFT_COLORS: Record<"include" | "exclude", CompletedRoiDraftColors> = {
  include: { strokeColor: "#f97316", fillColor: "#fb923c" },
  exclude: { strokeColor: "#ef4444", fillColor: "#ef4444" },
};

export function generateCompletedRoiDraftOverlays(
  polygons: DraftPolygon[],
  liveSectionPoints: Point[],
  mode: "include" | "exclude" = "include"
): SegmentOverlay[] {
  const { strokeColor, fillColor } = COMPLETED_ROI_DRAFT_COLORS[mode];

  const closedOverlays = polygons
    .filter((polygon) => polygon.closed)
    .map((polygon) => buildClosedPolygonRing(polygon))
    .filter((ring): ring is Point[] => Array.isArray(ring) && ring.length >= 4)
    .map((ring, index) => ({
      id: `completed-roi-draft-closed-${index}`,
      geometry: ring,
      fillColor,
      fillOpacity: 0.06,
      strokeColor,
      strokeOpacity: 0.98,
      strokeWidth: 2.5,
    }));

  const openSectionOverlays = polygons.flatMap((polygon) =>
    polygon.closed
      ? []
      : polygon.segments.map((segment) => ({
          id: `completed-roi-draft-section-${segment.id}`,
          geometry: segment.points,
          fillColor: "transparent",
          fillOpacity: 0,
          strokeColor,
          strokeOpacity: 0.98,
          strokeWidth: 2.5,
          shape: "polyline" as const,
        }))
  );

  const liveOverlay =
    liveSectionPoints.length >= 2
      ? [
          {
            id: "completed-roi-draft-live",
            geometry: liveSectionPoints,
            fillColor: "transparent",
            fillOpacity: 0,
            strokeColor,
            strokeOpacity: 0.98,
            strokeWidth: 2.5,
            shape: "polyline" as const,
          },
        ]
      : [];

  return [...closedOverlays, ...openSectionOverlays, ...liveOverlay];
}
