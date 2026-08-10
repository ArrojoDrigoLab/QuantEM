import { apiRequest } from "@/shared/api/core/http";
import {
  withSmoothedSegmentGeometry,
  withSmoothedSegmentGeometryBatch,
} from "@/utils/segmentGeometry";
import type {
  BatchLabelUpdatePayload,
  BatchLabelUpdateResponse,
  ClearSegmentationManualLabelsResponse,
  ConfirmBatchRequestPayload,
  ConfirmBatchResponse,
  ConfirmSegmentResponse,
  CreateUserFeedbackResponse,
  MarkSegmentationCompleteResponse,
  PointParams,
  QuerySegmentsInRegionResponse,
  RemoveAreaRequestPayload,
  RemoveAreaResponse,
  SegmentationOverlayMutationState,
  SegmentLabelUpdatePayload,
  SegmentObject,
  SegmentationCompletionPreview,
  SegmentRegionQueryPayload,
  UnlockSegmentationResponse,
  UserFeedback,
  UserFeedbackCreatePayload,
  UserFeedbackListParams,
} from "@/shared/types/segmentation";

export function getSegmentsAtPoint(
  segmentationId: string,
  params: PointParams,
  options?: {
    signal?: AbortSignal;
    geometryMode?: "full" | "hover";
  }
): Promise<SegmentObject[]> {
  const query = new URLSearchParams();
  query.set("x", params.x.toString());
  query.set("y", params.y.toString());
  if (params.states && params.states.length > 0) {
    query.set("states", params.states.join(","));
  }
  if (params.statuses && params.statuses.length > 0) {
    query.set("statuses", params.statuses.join(","));
  }
  if (params.source_model) {
    query.set("source_model", params.source_model);
  }
  if (options?.geometryMode === "hover") {
    query.set("geometry_detail", "hover");
  }
  return apiRequest<SegmentObject[]>(
    `/api/segmentations/${segmentationId}/segments/at-point?${query.toString()}`,
    options
  ).then(withSmoothedSegmentGeometryBatch);
}

export function querySegmentsInRegion(
  segmentationId: string,
  payload: SegmentRegionQueryPayload
): Promise<QuerySegmentsInRegionResponse> {
  return apiRequest<QuerySegmentsInRegionResponse>(
    `/api/segmentations/${segmentationId}/segments/query-region`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

export function updateSegmentLabel(
  segmentId: string,
  payload: SegmentLabelUpdatePayload
): Promise<ConfirmSegmentResponse> {
  return apiRequest<ConfirmSegmentResponse>(`/api/segments/${segmentId}/label/`, {
    method: "POST",
    body: JSON.stringify(payload),
  }).then(withSmoothedSegmentGeometry);
}

export function updateSegmentLabelsBatch(
  payload: BatchLabelUpdatePayload
): Promise<BatchLabelUpdateResponse> {
  return apiRequest<BatchLabelUpdateResponse>("/api/segments/labels/batch/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface DeleteSegmentsBatchResponse {
  deleted: number;
  overlay: SegmentationOverlayMutationState | null;
}

/** Hard-delete segments by id (e.g. ER "reject group" removes candidates entirely). */
export function deleteSegmentsBatch(
  segmentationId: string,
  payload: { ids: string[]; source_model?: string | null }
): Promise<DeleteSegmentsBatchResponse> {
  return apiRequest<DeleteSegmentsBatchResponse>(
    `/api/segmentations/${segmentationId}/segments/delete-batch/`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

export function clearSegmentationManualLabels(
  segmentationId: string
): Promise<ClearSegmentationManualLabelsResponse> {
  return apiRequest<ClearSegmentationManualLabelsResponse>(
    `/api/segmentations/${segmentationId}/labels/clear`,
    {
      method: "POST",
    }
  );
}

export function getUncertainSegments(
  segmentationId: string,
  limit: number = 50,
  sourceModel?: string | null
): Promise<SegmentObject[]> {
  const query = new URLSearchParams();
  query.set("limit", String(limit));
  if (sourceModel) query.set("source_model", sourceModel);
  return apiRequest<SegmentObject[]>(
    `/api/segmentations/${segmentationId}/segments/uncertain?${query.toString()}`
  ).then(withSmoothedSegmentGeometryBatch);
}

export function createUserFeedback(
  segmentationId: string,
  payload: UserFeedbackCreatePayload
): Promise<CreateUserFeedbackResponse> {
  return apiRequest<CreateUserFeedbackResponse>(
    `/api/segmentations/${segmentationId}/user-feedback/`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

export function listUserFeedback(
  segmentationId: string,
  params: UserFeedbackListParams = {}
): Promise<UserFeedback[]> {
  const query = new URLSearchParams();
  if (params.ids && params.ids.length > 0) {
    query.set("ids", params.ids.join(","));
  }
  if (params.utilized_statuses && params.utilized_statuses.length > 0) {
    query.set("utilized_statuses", params.utilized_statuses.join(","));
  }
  const qs = query.toString();
  return apiRequest<UserFeedback[]>(
    `/api/segmentations/${segmentationId}/user-feedback/${qs ? `?${qs}` : ""}`
  );
}

/**
 * Every drawn outline goes through `confirm-batch`, never through
 * `POST /api/segmentations/<id>/segments/`.
 *
 * A `createSegment` wrapper for the single-segment endpoint sat here exported
 * and uncalled: no drawing tool, no test, nothing. It looked like the obvious
 * thing to reach for the next time someone added a drawing mode, and it is the
 * wrong one. The two endpoints answer a self-crossing stroke -- which the
 * freehand tools produce routinely -- in genuinely different ways, and only one
 * of them is right for a drawing tool:
 *
 *   - The single-segment endpoint **refuses** it: `400` with "This outline
 *     crosses itself and separates into N pieces, which cannot be stored as one
 *     object", naming this endpoint as where to send them. (It used to be a
 *     `500` with a Django traceback; that is fixed, and it is still a refusal.)
 *   - `confirm-batch` **stores every piece**, one object per enclosed area, and
 *     reports it in `outlines` so the caller can say "that stroke made two
 *     objects". It did *not* always do this: until `parse_outline_pieces` it
 *     kept the largest lobe and dropped the rest under a plain
 *     `200 {"created": 1}` -- measured, a figure-of-eight of two 2500 px lobes
 *     stored 2500 px, and nothing anywhere said so. "Repairs the same geometry"
 *     described that behaviour and was too kind to it.
 *
 * So the reason to route drawing here is not that this endpoint is more
 * forgiving; it is that a stroke enclosing several areas is a real drawing and
 * this is the endpoint that keeps all of it.
 */
export function confirmSegmentsBatch(
  segmentationId: string,
  payload: ConfirmBatchRequestPayload
): Promise<ConfirmBatchResponse> {
  return apiRequest<ConfirmBatchResponse>(
    `/api/segmentations/${segmentationId}/segments/confirm-batch/`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

/**
 * What marking this segmentation done would destroy. Changes nothing.
 *
 * Ask this immediately before showing the confirmation, not once when the
 * screen loaded: `POST` refuses an `acknowledged_discard_count` that does not
 * match this exactly, precisely so a dialog opened while an inference run was
 * finishing cannot delete objects the user was never shown.
 */
export function getSegmentationCompletionPreview(
  segmentationId: string
): Promise<SegmentationCompletionPreview> {
  return apiRequest<SegmentationCompletionPreview>(
    `/api/segmentations/${segmentationId}/complete`
  );
}

/**
 * Lock a segmentation, and optionally delete everything nobody confirmed.
 *
 * The discard is opt-in on the wire as well as in the UI: omitting
 * `discardUnconfirmed` locks the segmentation and keeps every object, so no
 * call path can throw away a run's output by accident. Asking for it requires
 * the count the user was shown; a mismatch comes back `409` with a fresh
 * preview attached rather than deleting the difference.
 */
export function markSegmentationComplete(
  segmentationId: string,
  options: { discardUnconfirmed?: boolean; acknowledgedDiscardCount?: number } = {}
): Promise<MarkSegmentationCompleteResponse> {
  const body = options.discardUnconfirmed
    ? {
        discard_unconfirmed: true,
        acknowledged_discard_count: options.acknowledgedDiscardCount ?? 0,
      }
    : {};
  return apiRequest<MarkSegmentationCompleteResponse>(
    `/api/segmentations/${segmentationId}/complete`,
    {
      method: "POST",
      body: JSON.stringify(body),
    }
  );
}

/**
 * Unlock, and put back whatever the last completion discarded.
 *
 * `restored.restorable === false` with a non-zero `archived_count` means the
 * discard was too large to archive and those objects are gone; never report a
 * successful undo without reading it.
 */
export function unlockSegmentation(
  segmentationId: string
): Promise<UnlockSegmentationResponse> {
  return apiRequest<UnlockSegmentationResponse>(
    `/api/segmentations/${segmentationId}/complete`,
    {
      method: "DELETE",
    }
  );
}

export function removeSegmentationArea(
  segmentationId: string,
  payload: RemoveAreaRequestPayload
): Promise<RemoveAreaResponse> {
  return apiRequest<RemoveAreaResponse>(
    `/api/segmentations/${segmentationId}/segments/remove-area/`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}
