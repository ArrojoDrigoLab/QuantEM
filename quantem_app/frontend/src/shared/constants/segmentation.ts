import type { StatusStage } from "@/shared/types/images";

export const RASTER_BORDER_OPACITY = 0.95;
export const CONFIRMED_FILL_COLOR = "#33cc66";
export const CONFIRMED_BORDER_COLOR = "#237a47";
export const LABELED_FILL_COLOR = "#38bdf8";
export const LABELED_BORDER_COLOR = "#0f6f94";
export const REFINED_FILL_COLOR = "#3b82f6";
export const REFINED_BORDER_COLOR = "#1d4ed8";
export const CANDIDATE_FILL_COLOR = "#ff0000";
export const CANDIDATE_BORDER_COLOR = "#991b1b";
export const CELLS_INTERNAL_NAME = "quantem_internal_cells";
export const TISSUE_INTERNAL_NAME = "quantem_internal_tissue";

/**
 * The stages at which a segmentation holds objects that can be drawn.
 *
 * A finished inference run leaves `CANDIDATES_READY`
 * (`segmentation/organelle_tasks.py`), **not** `COMPLETED`: `COMPLETED` is
 * written in exactly one place, `POST /api/segmentations/<id>/complete`, i.e.
 * "Mark Image Done" on the labeling screen. Gating the viewer's overlay on
 * `COMPLETED` therefore meant the result of an 11-27 minute run could not be
 * seen at all until the user left the viewer, went to another screen and
 * declared the image finished — before ever looking at it.
 *
 * `UPDATING` and `COMPUTING_FEATURES` are post-run bookkeeping over objects
 * that already exist, so they draw too. `FAILED` is excluded: the run wrote
 * nothing, and whatever is on the segmentation belongs to an earlier run the
 * failure card names rather than silently paints.
 */
export const SEGMENTATION_RESULT_STAGES: readonly StatusStage[] = [
  "CANDIDATES_READY",
  "UPDATING",
  "COMPUTING_FEATURES",
  "COMPLETED",
];

/** True when this stage has objects the viewer can draw. */
export function segmentationHasResults(stage: StatusStage): boolean {
  return SEGMENTATION_RESULT_STAGES.includes(stage);
}

/**
 * Object states the read-only viewer draws.
 *
 * Mirrors the server's own `STATE_DEFAULT_VISIBLE`
 * (`segmentation/overlay_ngff/constants.py`): everything except `excluded`.
 * The viewer used to draw only `{confirmed, refined, labeled}`, which gave
 * every `candidate` alpha 0 — so even a segmentation that reached `COMPLETED`
 * without discarding its unreviewed objects rendered as an empty overlay over
 * hundreds of real ones.
 *
 * `excluded` stays out on purpose. An object the user removed by hand is a
 * decision, and painting it back would misreport that decision as a kept
 * object.
 */
export const VIEWER_VISIBLE_OVERLAY_STATES: Set<string> = new Set([
  "confirmed",
  "candidate",
  "labeled",
  "refined",
]);
