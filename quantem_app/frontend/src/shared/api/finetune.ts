// Model catalogue and guided fine-tuning. See API_CONTRACT.md §Models and
// §"Guided fine-tuning".
import { apiRequest } from "@/shared/api/core/http";
import type {
  Adapter,
  AdaptCropsResponse,
  AdaptStartPayload,
  AdaptStartResponse,
  ModelCatalogue,
} from "@/shared/types/finetune";

/** The eight released packs plus anything the user has adapted. */
export function getModelCatalogue(): Promise<ModelCatalogue> {
  return apiRequest<ModelCatalogue>("/api/models/");
}

/**
 * `POST /api/models/<pack_id>/install/`.
 *
 * Two modes behind one endpoint, decided by `sourcePath`:
 *
 * - **With `sourcePath`** — install from a copy already on this machine (an
 *   unzipped release bundle, or raw training outputs). Runs inline; the `202`
 *   arrives with the job already terminal, so one poll reads SUCCESS.
 * - **Without** — download the pack from the model registry: `202` with a
 *   live `job_id` to poll for progress (`Job.progress` is the fraction,
 *   `Job.message` says what is happening), or an error naming exactly why
 *   this build cannot download (older backends refuse with a 501 that names
 *   the release-bundle route instead).
 *
 * `job_id` is null on the "already installed" short-circuit, which returns
 * `200` with nothing to poll.
 */
export interface ModelInstallResponse {
  job_id: string | null;
  pack_id?: string;
  status?: string;
  detail?: string;
}

export function installModelPack(
  packId: string,
  sourcePath?: string
): Promise<ModelInstallResponse> {
  return apiRequest<ModelInstallResponse>(
    `/api/models/${encodeURIComponent(packId)}/install/`,
    {
      method: "POST",
      body: JSON.stringify(sourcePath ? { source_path: sourcePath } : {}),
    }
  );
}

/**
 * What the user has annotated for this segmentation, and whether it is enough.
 *
 * `blockers` is the readiness verdict, not advice: a completed ROI is what
 * makes "not an object" mean background rather than unlabelled, and without one
 * every Dice on this page would be meaningless.
 */
export function getAdaptCrops(segmentationId: string): Promise<AdaptCropsResponse> {
  return apiRequest<AdaptCropsResponse>(
    `/api/segmentations/${segmentationId}/adapt/crops/`
  );
}

export function startAdaptation(
  segmentationId: string,
  payload: AdaptStartPayload
): Promise<AdaptStartResponse> {
  return apiRequest<AdaptStartResponse>(
    `/api/segmentations/${segmentationId}/adapt/`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );
}

export function getAdapter(adapterId: string): Promise<Adapter> {
  return apiRequest<Adapter>(`/api/adapters/${adapterId}/`);
}

/** Use this adapter for subsequent runs on its segmentation. */
export function applyAdapter(adapterId: string): Promise<Adapter> {
  return apiRequest<Adapter>(`/api/adapters/${adapterId}/apply/`, {
    method: "POST",
  });
}
