/**
 * The include-level dial's two calls.
 *
 * `GET` is cheap and answers "can this move, and if not why" as well as where
 * the dial currently is, so the control can be greyed out with the reason
 * beside it instead of failing under the user's hand. `POST` queues one
 * re-extract and returns its job id.
 *
 * There is deliberately no generic `submitJob` here. Every other feature in
 * this app queues through a purpose-built endpoint that validates first and
 * returns `202 { job_id }`, and this one does the same: posting the job type
 * straight to `/api/jobs/` would skip every refusal the endpoint makes and
 * write a task that is certain to fail.
 */

import { apiRequest } from "@/shared/api/core/http";

export interface IncludeLevelState {
  /**
   * The level the objects on screen were found at, or `null` when nobody has
   * moved the dial. `null` is not 0.5: the run's own threshold is a different
   * fact, and showing it here would claim a dial position the user never set.
   */
  include_level: number | null;
  /** The model's own level, for "back to default". `null` if unknown. */
  default_include_level: number | null;
  minimum: number;
  maximum: number;
  run_version: number;
  object_count: number;
  can_move: boolean;
  /** Saved grayscale result for the live overlay. Present only when ready. */
  preview_url?: string;
  /** [x, y, width, height] in source-image pixels. */
  preview_bounds?: [number, number, number, number];
  /** Why it cannot move. Empty when it can. */
  detail: string;
  error_code?: string;
}

export interface IncludeLevelQueued {
  job_id: string;
  include_level: number;
}

export function getIncludeLevel(
  segmentationId: string,
  sourceModel?: string | null
): Promise<IncludeLevelState> {
  const query = new URLSearchParams();
  if (sourceModel) query.set("source_model", sourceModel);
  const qs = query.toString();
  return apiRequest<IncludeLevelState>(
    `/api/segmentations/${segmentationId}/include-level${qs ? `?${qs}` : ""}`
  );
}

export function setIncludeLevel(
  segmentationId: string,
  includeLevel: number,
  sourceModel?: string | null
): Promise<IncludeLevelQueued> {
  return apiRequest<IncludeLevelQueued>(
    `/api/segmentations/${segmentationId}/include-level`,
    {
      method: "POST",
      body: JSON.stringify(
        sourceModel
          ? { include_level: includeLevel, source_model: sourceModel }
          : { include_level: includeLevel }
      ),
    }
  );
}
