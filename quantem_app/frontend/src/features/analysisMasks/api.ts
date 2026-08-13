import { apiRequest } from "@/shared/api/core/http";
import type {
  AnalysisMaskObject,
  AnalysisMaskObjectDeleteResponse,
  AnalysisMaskObjectListResponse,
  AnalysisMaskObjectMutationResponse,
  AnalysisMaskObjectSaveResponse,
  AnalysisMaskOperation,
  AnalysisMaskShape,
} from "./types";

export function listAnalysisMaskObjects(
  segmentationId: string
): Promise<AnalysisMaskObjectListResponse> {
  return apiRequest<AnalysisMaskObjectListResponse>(
    `/api/segmentations/${segmentationId}/analysis-mask-objects/`
  );
}

export function patchAnalysisMaskObject(
  segmentationId: string,
  payload: {
    objectId: string | null;
    operation: AnalysisMaskOperation;
    shapes: AnalysisMaskShape[];
  }
): Promise<AnalysisMaskObjectMutationResponse> {
  return apiRequest<AnalysisMaskObjectMutationResponse>(
    `/api/segmentations/${segmentationId}/analysis-mask-objects/`,
    {
      method: "POST",
      body: JSON.stringify({
        object_id: payload.objectId,
        operation: payload.operation,
        shapes: payload.shapes,
      }),
    }
  );
}

export function saveAnalysisMaskObjects(
  segmentationId: string
): Promise<AnalysisMaskObjectSaveResponse> {
  return apiRequest<AnalysisMaskObjectSaveResponse>(
    `/api/segmentations/${segmentationId}/analysis-mask-objects/save/`,
    { method: "POST" }
  );
}

export function renameAnalysisMaskObject(
  segmentationId: string,
  objectId: string,
  name: string
): Promise<AnalysisMaskObject> {
  return apiRequest<AnalysisMaskObject>(
    `/api/segmentations/${segmentationId}/analysis-mask-objects/${objectId}/`,
    {
      method: "PATCH",
      body: JSON.stringify({ name }),
    }
  );
}

export function deleteAnalysisMaskObject(
  segmentationId: string,
  objectId: string
): Promise<AnalysisMaskObjectDeleteResponse> {
  return apiRequest<AnalysisMaskObjectDeleteResponse>(
    `/api/segmentations/${segmentationId}/analysis-mask-objects/${objectId}/`,
    { method: "DELETE" }
  );
}
