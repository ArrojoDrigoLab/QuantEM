// Quantitative analysis runs. See API_CONTRACT.md §Analysis.
import { apiRequest, resolveApiUrl } from "@/shared/api/core/http";
import type {
  AnalysisRun,
  AnalysisRunCreatePayload,
  AnalysisRunCreateResponse,
  AnalysisRunSummary,
} from "@/shared/types/analysis";

/**
 * Queue an analysis of one segmentation. Returns immediately with the job to
 * poll and the run the job will fill in; a full-resolution analysis rasterises
 * several masks and runs a Monte-Carlo null, so nothing here is synchronous.
 */
export function startAnalysisRun(
  segmentationId: string,
  payload: AnalysisRunCreatePayload
): Promise<AnalysisRunCreateResponse> {
  return apiRequest<AnalysisRunCreateResponse>(
    `/api/segmentations/${segmentationId}/analysis/`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

/** Runs already started for a segmentation, newest first. */
export function getAnalysisRuns(
  segmentationId: string
): Promise<AnalysisRunSummary[]> {
  return apiRequest<AnalysisRunSummary[]>(
    `/api/segmentations/${segmentationId}/analysis/`
  );
}

export function getAnalysisRun(runId: string): Promise<AnalysisRun> {
  return apiRequest<AnalysisRun>(`/api/analysis/${runId}/`);
}

/**
 * Download URL for one file of a run's export bundle.
 *
 * The endpoint sets `Content-Disposition: attachment`, so this is safe to use
 * as a plain `href`: the browser saves it rather than navigating the SPA away.
 */
export function getAnalysisExportUrl(runId: string, name: string): string {
  return resolveApiUrl(
    `/api/analysis/${runId}/export/${encodeURIComponent(name)}`
  );
}
