// Model catalogue and guided fine-tuning. See API_CONTRACT.md §Models and
// §"Guided fine-tuning".
import { ApiRequestError, apiRequest } from "@/shared/api/core/http";
import type {
  Adapter,
  AdaptCropsResponse,
  AdaptLatestResponse,
  AdaptStartPayload,
  AdaptStartResponse,
  FineTuneAdapterSummary,
  FineTuneApplyProgress,
  FineTuneApplyResponse,
  FineTunePreviewResponse,
  FineTuneProgress,
  FineTuneRunDetail,
  FineTuneRunPayload,
  FineTuneRunResponse,
  FineTuneScopeResponse,
  FineTuneScopeSelectionPayload,
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

/** Remove one downloaded model pack from this machine. */
export function removeModelPack(packId: string): Promise<void> {
  return apiRequest<void>(`/api/models/${encodeURIComponent(packId)}/`, {
    method: "DELETE",
  });
}

/**
 * What the user has annotated for this segmentation, and whether it is enough.
 *
 * `blockers` is the readiness verdict, not advice: a completed ROI is what
 * makes "not an object" mean background rather than unlabelled, and without one
 * every Dice on this page would be meaningless.
 */
export function getAdaptCrops(
  segmentationId: string,
  baseModel?: string | null
): Promise<AdaptCropsResponse> {
  // `base_model` is what makes `mode_blockers.head` answerable: the
  // training-window size rule is a property of the pack, so without one the
  // response can only report the reasons that hold for every pack.
  const query = baseModel
    ? `?base_model=${encodeURIComponent(baseModel)}`
    : "";
  return apiRequest<AdaptCropsResponse>(
    `/api/segmentations/${segmentationId}/adapt/crops/${query}`
  );
}

/**
 * The most recent run for this segmentation, from the server.
 *
 * Asked on mount instead of reading a browser-local pointer, so a reload, a
 * different machine and a cleared store all find the same run — and a finished
 * run never blocks the next one.
 */
export function getLatestAdaptRun(
  segmentationId: string
): Promise<AdaptLatestResponse> {
  return apiRequest<AdaptLatestResponse>(
    `/api/segmentations/${segmentationId}/adapt/latest/`
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

// ---------------------------------------------------------------------------
// Fine-tuning over a scope of images — `/api/finetune/`, contract §4.
//
// A second family of endpoints rather than a rewrite of the ones above: the
// "Improve" panel trains one segmentation from the screen it is open on and
// still uses them. These train one organelle across a chosen set of datasets
// and images, which is a different question and a different scope object.
// ---------------------------------------------------------------------------

/**
 * The whole selectable tree for one organelle, in one call.
 *
 * Not paginated on purpose: this is a desktop library of hundreds of images,
 * and a dialog that has to page to total its own count cannot show a live
 * count.
 */
export function getFineTuneScope(
  segmentationTypeId: string
): Promise<FineTuneScopeResponse> {
  return apiRequest<FineTuneScopeResponse>(
    `/api/finetune/scope/?segmentation_type=${encodeURIComponent(segmentationTypeId)}`
  );
}

/**
 * What a selection amounts to: its count, its tiles, its experiment, and
 * whether it can be trained on at all.
 *
 * A POST because the selection can be long, and because the server resolves
 * `dataset_ids` into assets — the client never expands them itself, so the two
 * cannot disagree about what "the scope" is.
 */
export function previewFineTuneScope(
  payload: FineTuneScopeSelectionPayload
): Promise<FineTunePreviewResponse> {
  return apiRequest<FineTunePreviewResponse>("/api/finetune/preview/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Existing fine-tunes for this organelle, for the overwrite dropdown. */
export function listFineTuneAdapters(
  segmentationTypeId: string
): Promise<FineTuneAdapterSummary[]> {
  return apiRequest<FineTuneAdapterSummary[]>(
    `/api/finetune/adapters/?segmentation_type=${encodeURIComponent(segmentationTypeId)}`
  );
}

/**
 * Start one. `202` with the adapter row to poll and the job behind it.
 *
 * A name that already exists for this organelle and no `overwrite_adapter_id`
 * is a `409`, not a silent second row: see {@link isFineTuneNameConflict}.
 */
export function startFineTuneRun(
  payload: FineTuneRunPayload
): Promise<FineTuneRunResponse> {
  return apiRequest<FineTuneRunResponse>("/api/finetune/runs/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getFineTuneProgress(adapterId: string): Promise<FineTuneProgress> {
  return apiRequest<FineTuneProgress>(
    `/api/finetune/runs/${encodeURIComponent(adapterId)}/progress/`
  );
}

export function getFineTuneRun(adapterId: string): Promise<FineTuneRunDetail> {
  return apiRequest<FineTuneRunDetail>(
    `/api/finetune/runs/${encodeURIComponent(adapterId)}/`
  );
}

/**
 * Run the finished model on selected images or Datasets in its Experiment.
 *
 * Never called on success. The dialog offers it; nothing is queued until the
 * user picks images and clicks (owner R13).
 */
export function applyFineTuneRun(
  adapterId: string,
  assetIds: string[],
  datasetIds: string[] = []
): Promise<FineTuneApplyResponse> {
  return apiRequest<FineTuneApplyResponse>(
    `/api/finetune/runs/${encodeURIComponent(adapterId)}/apply/`,
    {
      method: "POST",
      body: JSON.stringify({ asset_ids: assetIds, dataset_ids: datasetIds }),
    }
  );
}

/** Per-image progress for one opt-in Dataset/image application batch. */
export function getFineTuneApplyProgress(
  adapterId: string,
  batchId: string
): Promise<FineTuneApplyProgress> {
  const query = new URLSearchParams({ batch_id: batchId });
  return apiRequest<FineTuneApplyProgress>(
    `/api/finetune/runs/${encodeURIComponent(adapterId)}/apply/?${query}`
  );
}

/**
 * The sentence out of a `{"detail": "..."}` error body.
 *
 * `apiRequest` throws with the raw response text as its message, because most
 * of the app's errors are already sentences. These are JSON, so a dialog that
 * printed `err.message` would put a brace and a key in front of the user.
 */
export function fineTuneErrorMessage(error: unknown, fallback: string): string {
  const raw =
    error instanceof ApiRequestError || error instanceof Error ? error.message : "";
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown } | null;
    const detail = parsed && typeof parsed.detail === "string" ? parsed.detail : "";
    if (detail) return detail;
  } catch {
    // Not JSON: an older route, or a proxy in the way. The text is the message.
  }
  return raw.trim().startsWith("{") ? fallback : raw;
}

/** A 409 from `POST /runs/`: this name is taken and overwrite was not asked for. */
export function isFineTuneNameConflict(error: unknown): boolean {
  return error instanceof ApiRequestError && error.status === 409;
}
