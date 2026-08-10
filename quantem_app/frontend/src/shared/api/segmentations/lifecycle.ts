/**
 * Segmentation lifecycle: read one segmentation, delete one segmentation.
 *
 * Deletion follows the Mark-Done contract: the dialog reads `delete_preview`
 * fresh when it opens, and the DELETE carries the object count the user was
 * shown. A count that has moved since — usually a run that finished while the
 * dialog was open — comes back as a 409 with the current numbers, and nothing
 * is deleted.
 */

import { apiRequest } from "@/shared/api/core/http";
import type {
  DeleteSegmentationResponse,
  SegmentationDetailResponse,
} from "@/shared/types/segmentation";

/** One segmentation, with the live deletion-preview counts riding along. */
export function getSegmentationDetail(
  segmentationId: string
): Promise<SegmentationDetailResponse> {
  return apiRequest<SegmentationDetailResponse>(
    `/api/segmentations/${segmentationId}/`
  );
}

/**
 * Delete a segmentation and everything it owns.
 *
 * Refused (409) while a job is active on it, while it is locked by Mark Image
 * Done, and when `acknowledgedObjectCount` no longer matches a fresh read.
 * Analysis runs made from it are kept server-side, marked
 * `segmentation_deleted` in their payloads.
 */
export function deleteSegmentation(
  segmentationId: string,
  acknowledgedObjectCount?: number
): Promise<DeleteSegmentationResponse> {
  return apiRequest<DeleteSegmentationResponse>(
    `/api/segmentations/${segmentationId}/`,
    {
      method: "DELETE",
      ...(acknowledgedObjectCount === undefined
        ? {}
        : {
            body: JSON.stringify({
              acknowledged_object_count: acknowledgedObjectCount,
            }),
          }),
    }
  );
}
