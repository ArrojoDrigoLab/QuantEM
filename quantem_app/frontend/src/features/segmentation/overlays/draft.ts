import type { SegmentOverlay } from "@/viewer/types";
import type { BBox } from "@/shared/types/common";
import type { Point } from "@/utils/geometry";
import { bboxToGeometry } from "@/features/segmentation/overlays/shared";

/** The drag-box drawn while selecting a group of objects to confirm/reject. */
export function generateSelectionBBoxOverlay(bbox: BBox | null): SegmentOverlay | null {
  if (!bbox) return null;
  return {
    id: "right-selection-bbox",
    geometry: bboxToGeometry(bbox),
    fillColor: "#00ffff",
    fillOpacity: 0.05,
    strokeColor: "#00ffff",
    strokeOpacity: 0.9,
    strokeWidth: 2,
  };
}

/** The freehand polygon being drawn but not yet accepted. */
export function generatePendingPolygonOverlay(
  pendingPolygon: Point[] | null
): SegmentOverlay | null {
  if (!pendingPolygon) return null;
  return {
    id: "pending-polygon",
    geometry: pendingPolygon,
    fillColor: "#ff0000",
    fillOpacity: 0.3,
    strokeColor: "#ff0000",
    strokeOpacity: 0.9,
    strokeWidth: 2,
  };
}
