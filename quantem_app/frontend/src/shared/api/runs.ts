/**
 * Costing a multi-organelle run, and starting it as one job.
 *
 * `getRunPlan` is deliberately a GET with no side effects: the workspace prices
 * the run while the user is still deciding, and a preview that created
 * segmentations or queued anything would make the price itself a commitment.
 */

import { apiRequest } from "@/shared/api/core/http";
import type { RunPlan, StartRunResponse } from "@/shared/types/runs";

export function getRunPlan(
  assetId: string,
  organelles: string[]
): Promise<RunPlan> {
  const query = organelles.length
    ? `?organelles=${encodeURIComponent(organelles.join(","))}`
    : "";
  return apiRequest<RunPlan>(`/api/assets/${assetId}/runs/${query}`);
}

/**
 * Start every ticked organelle as one run.
 *
 * `sourceModels` names a model family per organelle and is normally omitted:
 * the server picks the default pack, and stating it only matters when the user
 * chose the other family by hand.
 */
export function startImageRun(
  assetId: string,
  organelles: string[],
  sourceModels?: Record<string, string>
): Promise<StartRunResponse> {
  return apiRequest<StartRunResponse>(`/api/assets/${assetId}/runs/`, {
    method: "POST",
    body: JSON.stringify({
      organelles,
      ...(sourceModels && Object.keys(sourceModels).length
        ? { source_models: sourceModels }
        : {}),
    }),
  });
}
