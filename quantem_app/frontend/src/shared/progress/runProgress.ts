/**
 * Turning a job's progress fields into the three indicators the plan asks for.
 *
 * The owner asked for three things while a run is going, and they are three
 * because they answer three different questions:
 *
 * 1. **per-organelle tiling progress** — how far through *this* model's pass;
 * 2. **the aggregate across every organelle for this image** — how far through
 *    the whole thing the user pressed go on;
 * 3. **a separate model-download indicator** — bytes coming over the network,
 *    which is not segmentation progress and must never be mistaken for it.
 *
 * Every number here comes from a structured field. Nothing is parsed out of
 * `job.message`, which is how the drawer used to get a tile count and is why
 * `DINO: 57% (Tile 32/56)` — an internal codename, and a percentage on a
 * different divisor from the bar beside it — was the only tile count that ever
 * reached a user.
 *
 * One divisor
 * -----------
 * A run's `progress` covers the whole job: the model load, the tiles, finding
 * objects, saving them. Its denominator is therefore larger than the tiling
 * plan's tile count, and it is *not* the number to put next to "32 of 56
 * tiles". The per-organelle line here is tiles-primary — bar, percentage and
 * count all divide by `unit_progress.total`, the tiling plan — so the two
 * numbers on the row are the same number.
 *
 * ## Where the pieces live
 *
 * This module is now the public face of three:
 *
 * * `jobPredicates.ts` — what kind of job a row is, and whether it is still
 *   going;
 * * `progressCopy.ts` — the phrasing: stage names, counts, time left, bytes,
 *   and the run panel's own title;
 * * `progressRows.ts` — the row model, and therefore the denominators.
 *
 * Every consumer still imports from here, so which file a symbol lives in is
 * not part of the contract. Splitting it means a change to a sentence and a
 * change to a denominator are two files, not one.
 */

export {
  hasStructuredProgress,
  isDownloadJob,
  isLiveJob,
  isRunJob,
  isStoppedRunJob,
} from "@/shared/progress/jobPredicates";

export {
  STAGE_PHRASES,
  formatBytes,
  formatTimeLeft,
  formatUnits,
  joinClauses,
  runPanelTitle,
} from "@/shared/progress/progressCopy";

export {
  aggregateRow,
  buildAggregateRows,
  buildProgressRows,
  downloadRow,
  legRow,
  organelleRow,
} from "@/shared/progress/progressRows";

export type {
  ProgressRow,
  ProgressRowKind,
} from "@/shared/progress/progressRows";
