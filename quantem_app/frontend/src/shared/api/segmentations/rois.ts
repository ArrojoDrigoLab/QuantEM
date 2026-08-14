import { apiRequest } from "@/shared/api/core/http";
import {
  withSmoothedSegmentGeometryBatch,
} from "@/utils/segmentGeometry";
import type {
  ActivateSegmentationRoiResponse,
  MarkRoiCompleteResponse,
  RerunSegmentationRoiResponse,
  RoiCompletionResponse,
  SegmentationConfigResponse,
  SegmentObject,
  SegmentationRoi,
} from "@/shared/types/segmentation";
import type { SegmentationInstanceParams } from "@/shared/types/images";

export function getSegmentationRois(segmentationId: string): Promise<SegmentationRoi[]> {
  return apiRequest<SegmentationRoi[]>(`/api/segmentations/${segmentationId}/roi/`);
}

export function createSegmentationRoi(
  segmentationId: string,
  payload: Partial<Pick<SegmentationRoi, "x" | "y" | "width" | "height">> & {
    source?: "AUTO" | "MANUAL" | "DEFAULT";
    seed?: number | null;
  }
): Promise<SegmentationRoi> {
  return apiRequest<SegmentationRoi>(`/api/segmentations/${segmentationId}/roi/`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function deleteSegmentationRoi(
  segmentationId: string,
  roiId: string
): Promise<void> {
  return apiRequest<void>(`/api/segmentations/${segmentationId}/roi/${roiId}/`, {
    method: "DELETE",
  });
}

export function activateSegmentationRoi(
  segmentationId: string,
  roiId: string
): Promise<ActivateSegmentationRoiResponse> {
  return apiRequest<ActivateSegmentationRoiResponse>(
    `/api/segmentations/${segmentationId}/roi/activate/`,
    {
      method: "POST",
      body: JSON.stringify({ roi_id: roiId }),
    }
  );
}

export function getRoiSegments(segmentationId: string): Promise<SegmentObject[]> {
  return apiRequest<SegmentObject[]>(`/api/segmentations/${segmentationId}/roi/segments`).then(
    withSmoothedSegmentGeometryBatch
  );
}

export function markRoiComplete(segmentationId: string): Promise<MarkRoiCompleteResponse> {
  return apiRequest<MarkRoiCompleteResponse>(`/api/segmentations/${segmentationId}/roi/complete`, {
    method: "POST",
  });
}

/**
 * Mark a specific ROI as done (or not) for THIS segmentation (organelle).
 * Per-organelle analogue of `markRoiComplete` (which marks the active ROI for
 * the flat per-ROI flag). Returns the updated ROI with
 * `completed_for_segmentation` reflecting the new state.
 */
export function setRoiCompleteForSegmentation(
  segmentationId: string,
  roiId: string,
  isComplete: boolean
): Promise<RoiCompletionResponse> {
  return apiRequest<RoiCompletionResponse>(
    `/api/segmentations/${segmentationId}/roi/${roiId}/complete`,
    { method: isComplete ? "POST" : "DELETE" }
  );
}

export function rerunSegmentationRoi(
  segmentationId: string,
  roiId?: string | null,
  sourceModel?: string | null,
  adapterId?: string | null
): Promise<RerunSegmentationRoiResponse> {
  return apiRequest<RerunSegmentationRoiResponse>(
    `/api/segmentations/${segmentationId}/rerun-roi/`,
    {
      method: "POST",
      body: JSON.stringify({
        ...(roiId ? { roi_id: roiId } : {}),
        ...(sourceModel ? { source_model: sourceModel } : {}),
        ...(adapterId ? { adapter_id: adapterId } : {}),
      }),
    }
  );
}

export function getSegmentationConfig(
  segmentationId: string
): Promise<SegmentationConfigResponse> {
  return apiRequest<SegmentationConfigResponse>(`/api/segmentations/${segmentationId}/config/`);
}

export function updateSegmentationConfig(
  segmentationId: string,
  payload:
    | { instance_params: Partial<SegmentationInstanceParams> }
    | Partial<SegmentationInstanceParams>
): Promise<SegmentationConfigResponse> {
  return apiRequest<SegmentationConfigResponse>(`/api/segmentations/${segmentationId}/config/`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}
