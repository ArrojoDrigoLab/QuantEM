import type { SegmentOverlay } from "@/viewer/types";
import type { SegmentationRoi } from "@/shared/types/segmentation";
import type { Point } from "@/utils/geometry";
import { brushStrokesToConnectedPolygons } from "@/utils/brushMask";
import { boundsToGeometry, circlePoints } from "@/features/segmentation/overlays/shared";

export interface RoiStroke {
  id: string;
  label: number;
  size: number;
  points: Point[];
}

export function generateRoiFrameOverlay(
  bounds:
    | {
        x: number;
        y: number;
        width: number;
        height: number;
      }
    | null,
  id: string = "roi-frame"
): SegmentOverlay | null {
  if (!bounds) return null;
  return {
    id,
    geometry: boundsToGeometry(bounds),
    fillColor: "transparent",
    fillOpacity: 0,
    strokeColor: "#ffd166",
    strokeOpacity: 0.9,
    strokeWidth: 3,
  };
}

export function generateRoiOverlays(activeRoi: SegmentationRoi | null): SegmentOverlay[] {
  const overlay = activeRoi
    ? generateRoiFrameOverlay({
        x: activeRoi.x,
        y: activeRoi.y,
        width: activeRoi.width,
        height: activeRoi.height,
      })
    : null;
  return overlay ? [overlay] : [];
}

export function generateRoiStrokeOverlays(roiStrokes: RoiStroke[]): SegmentOverlay[] {
  if (roiStrokes.length === 0) {
    return [];
  }

  return roiStrokes.flatMap((stroke) => {
    const isPositive = stroke.label === 1;
    const strokeColor = isPositive ? "#33cc66" : "#ff5d5d";

    if (stroke.points.length < 2) {
      const point = stroke.points[0];
      if (!point) return [];
      return [
        {
          id: `roi-stroke-${stroke.id}`,
          geometry: circlePoints(point, Math.max(stroke.size / 2, 2), 12),
          fillColor: strokeColor,
          fillOpacity: 0.6,
          strokeColor: "#111",
          strokeOpacity: 0.7,
          strokeWidth: 1,
        },
      ];
    }

    return [
      {
        id: `roi-stroke-${stroke.id}`,
        geometry: stroke.points,
        shape: "polyline",
        fillColor: "transparent",
        fillOpacity: 0,
        strokeColor,
        strokeOpacity: 0.65,
        strokeWidth: Math.max(stroke.size, 2),
      },
    ];
  });
}

export function generateDrawStrokeOverlays(brushStrokes: RoiStroke[]): SegmentOverlay[] {
  return brushStrokesToConnectedPolygons(brushStrokes).map((geometry, index) => ({
    id: `draw-stroke-component-${index}`,
    geometry,
    fillColor: "#33cc66",
    fillOpacity: 0.35,
    strokeColor: "#2aa957",
    strokeOpacity: 0.95,
    strokeWidth: 1.5,
  }));
}
