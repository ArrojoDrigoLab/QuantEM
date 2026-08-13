import { buildSyntheticSegmentsFromGeometries } from "@/features/segmentation/screen/utils/optimisticSegments";
import type { SegmentObject } from "@/shared/types/segmentation";

import type { SamBoxResponse } from "./types";

/**
 * Turn the polygon returned by SAM into the vector object shown while the
 * revised raster overlay is being built and loaded.
 *
 * The endpoint has already persisted the object when it responds. Calling the
 * vector "optimistic" describes its display lifetime: it bridges the gap until
 * the authoritative raster reaches the same revision in the viewer.
 */
export function optimisticSegmentForSamResponse(
  segmentationId: string,
  response: SamBoxResponse
): SegmentObject | null {
  const objectId = response.confirmed_ids[0];
  const geometry = response.object.geometry_coords;
  if (!objectId || geometry.length < 3) return null;

  const [segment] = buildSyntheticSegmentsFromGeometries(
    segmentationId,
    [geometry],
    "CONFIRMED",
    [objectId]
  );
  return segment;
}
