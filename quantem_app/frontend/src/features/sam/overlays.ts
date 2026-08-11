/** The two rectangles this tool draws while the user is working. */

import { bboxToGeometry } from "@/features/segmentation/overlays/shared";
import type { BBox } from "@/shared/types/common";
import type { SegmentOverlay } from "@/viewer/types";

/** Ids are namespaced so they cannot collide with a segment uuid.
 *
 * The viewer recolours any overlay whose id matches the highlighted segment,
 * and warns on duplicates, so a prefixed literal is the safe choice.
 */
const LIVE_ID = "sam-box-live";
const PENDING_ID = "sam-box-pending";

/** The rubber band under the cursor. Cyan, matching the other drag box. */
export function samLiveBoxOverlay(bbox: BBox | null): SegmentOverlay | null {
  if (!bbox) return null;
  return {
    id: LIVE_ID,
    geometry: bboxToGeometry(bbox),
    fillColor: "#00ffff",
    fillOpacity: 0.05,
    strokeColor: "#00ffff",
    strokeOpacity: 0.9,
    strokeWidth: 2,
  };
}

/** The box whose request is in flight.
 *
 * Amber and unfilled, so it reads as "registered, working" rather than as an
 * object that already exists. The request is short but not instant, and without
 * this the box vanishes on release and the user cannot tell whether the drag
 * was seen.
 */
export function samPendingBoxOverlay(bbox: BBox | null): SegmentOverlay | null {
  if (!bbox) return null;
  return {
    id: PENDING_ID,
    geometry: bboxToGeometry(bbox),
    fillColor: "transparent",
    fillOpacity: 0,
    strokeColor: "#f8c848",
    strokeOpacity: 0.95,
    strokeWidth: 2,
  };
}
