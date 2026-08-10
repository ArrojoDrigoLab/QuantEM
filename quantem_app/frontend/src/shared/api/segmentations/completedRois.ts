import { apiRequest } from "@/shared/api/core/http";
import type {
  CompletedRoi,
  SubtractCompletedRoiResponse,
} from "@/shared/types/segmentation";

export interface CreateCompletedRoiPayload {
  polygon_coords: Array<[number, number]>;
}

export function getCompletedRois(segmentationId: string): Promise<CompletedRoi[]> {
  return apiRequest<CompletedRoi[]>(`/api/segmentations/${segmentationId}/completed-rois/`);
}

export function createCompletedRoi(
  segmentationId: string,
  payload: CreateCompletedRoiPayload
): Promise<CompletedRoi> {
  return apiRequest<CompletedRoi>(`/api/segmentations/${segmentationId}/completed-rois/`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Subtract a freehand polygon from the confirmed-area layer. */
export function subtractCompletedRoi(
  segmentationId: string,
  payload: CreateCompletedRoiPayload
): Promise<SubtractCompletedRoiResponse> {
  return apiRequest<SubtractCompletedRoiResponse>(
    `/api/segmentations/${segmentationId}/completed-rois/subtract/`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}
