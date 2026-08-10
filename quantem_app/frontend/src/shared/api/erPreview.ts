import { apiRequest } from "@/shared/api/core/http";

export interface ErPreviewResult {
  ok: boolean;
  source_model: string;
  bbox: { x: number; y: number; width: number; height: number };
  prob_image: string; // grayscale probability map (data URL); colorized client-side
  color: [number, number, number];
  default_threshold: number;
  stats: { elapsed_s: number; frac: number; work_shape: number[] };
}

/** Run a transient ER segmentation model on an ROI; returns a colored overlay PNG.
 * Passing `roi_id` caches the float probability map server-side so the result can
 * later be "pinned" into candidates at a chosen threshold without re-running. */
export function runErModelPreview(
  assetId: string,
  body: {
    source_model: string;
    x: number;
    y: number;
    width: number;
    height: number;
    roi_id?: string;
  }
): Promise<ErPreviewResult> {
  return apiRequest<ErPreviewResult>(`/api/assets/${assetId}/er-model-preview/`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export interface ErPinResult {
  ok: boolean;
  count: number;
  threshold: number;
  source_model: string;
}

/** Threshold the cached ER preview prob map for an ROI and persist the connected
 * components as CANDIDATE segments (no model re-run). */
export function pinErCandidates(
  segmentationId: string,
  body: { roi_id: string; source_model: string; threshold: number; min_area?: number }
): Promise<ErPinResult> {
  return apiRequest<ErPinResult>(
    `/api/segmentations/${segmentationId}/er/pin-candidates/`,
    {
      method: "POST",
      body: JSON.stringify(body),
    }
  );
}
