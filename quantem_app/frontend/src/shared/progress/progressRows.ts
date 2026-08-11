/**
 * The row model: a job's structured progress fields turned into one drawable
 * line.
 *
 * Split out of `runProgress.ts` unchanged. This is where the denominators live
 * — the one a per-organelle row divides by, and the one the aggregate divides
 * by — which is exactly what a change to how a run is counted has to touch, and
 * exactly what a change of wording (`progressCopy.ts`) must not.
 */

import {
  STAGE_PHRASES,
  formatBytes,
  formatTimeLeft,
  formatUnits,
  joinClauses,
} from "@/shared/progress/progressCopy";
import { isDownloadJob, isRunJob } from "@/shared/progress/jobPredicates";
import type {
  JobBatchProgress,
  JobQueueItem,
  JobUnitProgress,
} from "@/shared/types/jobs";
import type { RunLeg } from "@/shared/types/runs";

export type ProgressRowKind = "aggregate" | "organelle" | "download";

export interface ProgressRow {
  key: string;
  kind: ProgressRowKind;
  /** Row marker: filled for work under way, hollow for waiting, arrow for bytes. */
  glyph: string;
  /** "Everything", "Mitochondria", "QuantEM — Nucleus". */
  name: string;
  /** Bar width 0-100, or null when there is no honest bar to draw. */
  percent: number | null;
  /** Whether the percentage is also written out. False for byte rows. */
  showPercentText: boolean;
  /** The clauses after the bar, already ordered and glossed. */
  detail: string;
  tone: "normal" | "warning";
  /** One sentence for a screen reader, since the row is mostly geometry. */
  ariaLabel: string;
}

function organelleName(job: JobQueueItem): string {
  return job.segmentation?.name || job.image?.display_name || job.task_label;
}

function unitsOrNull(job: JobQueueItem): JobUnitProgress | null {
  const units = job.unit_progress;
  if (!units || !units.total) return null;
  return units;
}

/**
 * One organelle's line.
 *
 * Deliberately different in each phase, because a single percentage cannot say
 * all of it honestly:
 *
 * * **loading the model** — no percentage at all. This is the 4-20 s (minutes,
 *   on the rebuild fallback) that used to read as a frozen 5 %. The denominator
 *   is already known, so the line says what is happening and how much work is
 *   coming, and claims no fraction of it done.
 * * **the tiles** — tiles-primary, one divisor, with the run's own estimate of
 *   what is left.
 * * **after the tiles** — the bar stays full and the phase is named. Tiles
 *   really are finished; morphology and saving are not, and a bar sitting at
 *   100 % with nothing beside it is what makes a run look hung.
 * * **stopped** — the count it actually reached, never rounded up.
 *
 * The stopped branch was written in wave 0c and was unreachable from both
 * surfaces that call this function: the Tasks drawer built structured rows only
 * out of `running`, and the labeling panel dropped a run the moment it left the
 * wave. So a user who cancelled at tile 18 of 56 was told "cancelled" and
 * nothing else, and a failed run never quoted a tile count anywhere. Both
 * callers now pass concluded runs through here; this is the copy they get.
 */
export function organelleRow(job: JobQueueItem): ProgressRow {
  const units = unitsOrNull(job);
  const stage = String(job.progress_stage || "");
  const name = organelleName(job);
  const queued = job.status === "PENDING" || job.status === "RETRY";
  const stopped = job.status === "FAILED" || job.status === "CANCELLED";

  let percent: number | null = null;
  let showPercentText = false;
  let detail: string;
  let tone: "normal" | "warning" = "normal";
  let glyph = job.status === "RUNNING" ? "●" : "○";

  if (stopped) {
    // A square, not the hollow circle a waiting run gets: "not started" and
    // "started and stopped" are opposite facts and looked identical.
    glyph = "■";
    tone = "warning";
    // No tiling plan on the row means the run stopped before it had one --
    // typically the model failed to load. Saying "0 tiles" there would invent a
    // denominator the run never had.
    const reached = units
      ? `stopped at ${formatUnits(units.done, units.total, units.label)}`
      : "stopped before it counted any tiles";
    const why =
      job.status === "CANCELLED" ? "you stopped this one" : "this one did not finish";
    detail = joinClauses([reached, why]);
    // The bar is left at the fraction it actually reached, in the warning
    // colour. A full bar would say it finished; an empty one would say it never
    // started, and 18 of 56 tiles is neither.
    percent = units ? units.percent : null;
  } else if (job.status === "SUCCESS") {
    percent = 100;
    showPercentText = false;
    detail = joinClauses([
      units ? formatUnits(units.done, units.total, units.label) : null,
      "finished",
    ]);
  } else if (queued) {
    detail = joinClauses([
      "waiting to start",
      units ? formatUnits(0, units.total, units.label) : null,
    ]);
  } else if (stage === "loading_model") {
    // Name both pieces of setup before tile 1.
    detail = units
      ? `preparing the model and image — ${formatUnits(units.done, units.total, units.label)}`
      : "preparing the model and image";
  } else if (units && units.done < units.total) {
    percent = units.percent;
    showPercentText = true;
    detail = joinClauses([
      formatUnits(units.done, units.total, units.label),
      formatTimeLeft(units.eta_seconds),
    ]);
  } else if (units) {
    percent = 100;
    detail = joinClauses([
      formatUnits(units.done, units.total, units.label),
      STAGE_PHRASES[stage] ?? "finishing up",
    ]);
  } else {
    detail = STAGE_PHRASES[stage] ?? "starting";
  }

  return {
    key: `run:${job.id}`,
    kind: "organelle",
    glyph,
    name,
    percent,
    showPercentText,
    detail,
    tone,
    ariaLabel: `${name}: ${
      percent !== null && showPercentText ? `${Math.round(percent)}%, ` : ""
    }${detail}`,
  };
}

/**
 * One organelle's line inside a run that covers several.
 *
 * Same four phases as {@link organelleRow} and deliberately the same words: a
 * user cannot tell whether the app started one job or three, and must not be
 * able to. The difference is only where the numbers come from — a leg of a
 * shared job rather than a job of its own — and that a leg has no ETA, because
 * the estimate belongs to the run as a whole and is on the aggregate line
 * above.
 */
export function legRow(job: JobQueueItem, leg: RunLeg): ProgressRow {
  const total = leg.units_total;
  const done = Math.min(leg.units_done, total || leg.units_done);
  const label = leg.unit_label || "tile";
  const stopped = leg.status === "FAILED" || leg.status === "CANCELLED";
  const queued = leg.status === "PENDING";

  let percent: number | null = null;
  let showPercentText = false;
  let detail: string;
  let tone: "normal" | "warning" = "normal";
  let glyph = leg.status === "RUNNING" ? "●" : "○";

  if (stopped) {
    glyph = "■";
    tone = "warning";
    detail = joinClauses([
      total ? `stopped at ${formatUnits(done, total, label)}` : "stopped before it counted any tiles",
      leg.status === "CANCELLED" ? "you stopped this one" : "this one did not finish",
    ]);
    percent = total ? Math.round((100 * done) / total) : null;
  } else if (leg.status === "SUCCESS") {
    percent = 100;
    detail = joinClauses([total ? formatUnits(done, total, label) : null, "finished"]);
  } else if (queued) {
    detail = joinClauses([
      "waiting to start",
      total ? formatUnits(0, total, label) : null,
    ]);
  } else if (total && done < total) {
    percent = leg.percent;
    showPercentText = true;
    detail = formatUnits(done, total, label);
  } else if (total) {
    percent = 100;
    detail = joinClauses([formatUnits(done, total, label), "finishing up"]);
  } else {
    detail = STAGE_PHRASES[String(job.progress_stage || "")] ?? "starting";
  }

  return {
    key: `run:${job.id}:${leg.segmentation_id || leg.name}`,
    kind: "organelle",
    glyph,
    name: leg.name,
    percent,
    showPercentText,
    detail,
    tone,
    ariaLabel: `${leg.name}: ${
      percent !== null && showPercentText ? `${Math.round(percent)}%, ` : ""
    }${detail}`,
  };
}

/** The organelle lines this job contributes: its legs, or the job itself. */
function organelleRowsFor(job: JobQueueItem): ProgressRow[] {
  const legs = job.run_legs;
  if (legs && legs.length) return legs.map((leg) => legRow(job, leg));
  return [organelleRow(job)];
}

/**
 * How many organelles a wave is running, not how many job rows it has.
 *
 * These stopped being the same number the moment one job could cover four
 * organelles. The aggregate line exists to say something the line below it does
 * not, so it appears when there is more than one organelle — whether they
 * arrived as four jobs or as one.
 */
function organelleCount(jobs: JobQueueItem[], batchId: string): number {
  let count = 0;
  for (const job of jobs) {
    if ((job.batch_id || "") !== batchId) continue;
    if (!isRunJob(job)) continue;
    count += job.run_legs?.length || 1;
  }
  return count;
}

/**
 * Leg-aware "N of M did not finish".
 *
 * The rollup counts *jobs*, and one job carrying four organelles of which two
 * failed reports `runs_failed: 0` right up until the job itself concludes. The
 * clause is about organelles, so it is counted over organelles.
 */
function unfinishedFromLegs(
  jobs: JobQueueItem[],
  batchId: string
): { unfinished: number; total: number } | null {
  let unfinished = 0;
  let total = 0;
  let sawLegs = false;
  for (const job of jobs) {
    if ((job.batch_id || "") !== batchId) continue;
    if (!isRunJob(job)) continue;
    const legs = job.run_legs;
    if (!legs || !legs.length) {
      total += 1;
      if (job.status === "FAILED" || job.status === "CANCELLED") unfinished += 1;
      continue;
    }
    sawLegs = true;
    for (const leg of legs) {
      total += 1;
      if (leg.status === "FAILED" || leg.status === "CANCELLED") unfinished += 1;
    }
  }
  return sawLegs ? { unfinished, total } : null;
}

/**
 * The "Everything" line for one image.
 *
 * Time-primary with tiles secondary, and deliberately not a raw combined tile
 * bar: tile counts span 10:1 within one image (mitochondria 858, nucleus 88),
 * so a combined bar would read 96 % while the slower organelle still had
 * minutes to run.
 *
 * **The denominator is the whole wave, including the tiles a failed or
 * cancelled run will never walk.** It used to be `units_reachable`, which drops
 * them, and the result was measured on screen: *"Everything on montage16real
 * 100% · 25 of 25 tiles · 1 of 2 did not finish"* for a wave of 118 requested
 * tiles that walked 25. A bar that fills because two thirds of the work was
 * abandoned is answering a question nobody asked. It now reads *"21% · 25 of
 * 118 tiles · 2 of 3 did not finish"*, and the last clause is what says the bar
 * is not going to fill.
 */
export function aggregateRow(
  batch: JobBatchProgress,
  imageName?: string,
  /**
   * Organelle counts, when the wave's runs are legs of one job rather than
   * jobs. Without it the clause would count job rows, and a run of four
   * organelles with two of them failed would say "0 of 1 did not finish".
   */
  legCounts?: { unfinished: number; total: number } | null
): ProgressRow {
  const unfinished = legCounts
    ? legCounts.unfinished
    : batch.runs_failed + batch.runs_cancelled;
  const outOf = legCounts ? legCounts.total : batch.runs_total;
  const unfinishedClause =
    unfinished > 0 ? `${unfinished} of ${outOf} did not finish` : null;
  const detail =
    batch.percent === null
      ? joinClauses([
          // The wave has a run whose size is not known, so there is no
          // fraction to quote -- not "nothing happened".
          `${batch.units_done} ${batch.unit_label || "tile"}s so far, of a total this run cannot say yet`,
          unfinishedClause,
        ])
      : joinClauses([
          formatTimeLeft(batch.eta_seconds),
          formatUnits(batch.units_done, batch.units_total, batch.unit_label),
          unfinishedClause,
        ]);
  const name = imageName ? `Everything on ${imageName}` : "Everything";
  return {
    key: `batch:${batch.batch_id}`,
    kind: "aggregate",
    glyph: "",
    name,
    percent: batch.percent,
    showPercentText: batch.percent !== null,
    detail,
    tone: unfinished > 0 ? "warning" : "normal",
    ariaLabel: `${name}: ${
      batch.percent !== null ? `${Math.round(batch.percent)}%, ` : ""
    }${detail}`,
  };
}

/**
 * A model coming down the wire.
 *
 * Its own glyph, its own name, and bytes rather than a percentage, so that it
 * can never be read as "the segmentation is 32 % done". This is the owner's
 * "SEPARATE indicator", and it is separate all the way down: the bytes live in
 * their own fields on the job row, not in the tile fields.
 */
export function downloadRow(job: JobQueueItem): ProgressRow | null {
  const download = job.download;
  if (!download) return null;
  const name = job.model_pack?.title || job.task_label;
  const detail = `downloading the model — ${formatBytes(download)}`;
  return {
    key: `download:${job.id}`,
    kind: "download",
    glyph: "↓",
    name,
    percent: download.percent,
    showPercentText: false,
    detail,
    tone: "normal",
    ariaLabel: `${name}: ${detail}`,
  };
}

/**
 * One aggregate line per wave present in `jobs`, never two.
 *
 * Every job in a wave carries the same rollup, so building the list per job
 * would print the same "Everything" line once for each organelle running. It
 * is deduplicated by `batch_id`, and it only appears once the wave has more
 * than one run in it: with a single run the aggregate repeats the line below it
 * word for word, and a duplicated number is exactly the kind of thing a
 * sceptical reader treats as two disagreeing numbers.
 */
export function buildAggregateRows(
  jobs: JobQueueItem[],
  options: { alwaysShowAggregate?: boolean } = {}
): ProgressRow[] {
  const rows: ProgressRow[] = [];
  const seen = new Set<string>();
  for (const job of jobs) {
    const batch = job.batch_progress;
    if (!batch || seen.has(batch.batch_id)) continue;
    seen.add(batch.batch_id);
    const batchId = job.batch_id || "";
    if (
      organelleCount(jobs, batchId) > 1 ||
      batch.runs_total > 1 ||
      options.alwaysShowAggregate
    ) {
      rows.push(
        aggregateRow(
          batch,
          job.image?.display_name,
          unfinishedFromLegs(jobs, batchId)
        )
      );
    }
  }
  return rows;
}

/**
 * The whole list, in the plan's order: the aggregate first, then one line per
 * organelle, then the downloads.
 */
export function buildProgressRows(
  jobs: JobQueueItem[],
  options: {
    imageName?: string;
    alwaysShowAggregate?: boolean;
    includeAggregate?: boolean;
  } = {}
): ProgressRow[] {
  const rows: ProgressRow[] = [];

  if (options.includeAggregate !== false) {
    const seen = new Set<string>();
    for (const job of jobs) {
      const batch = job.batch_progress;
      if (!batch || seen.has(batch.batch_id)) continue;
      seen.add(batch.batch_id);
      const batchId = job.batch_id || "";
      if (
        organelleCount(jobs, batchId) > 1 ||
        batch.runs_total > 1 ||
        options.alwaysShowAggregate
      ) {
        rows.push(
          aggregateRow(
            batch,
            options.imageName ?? job.image?.display_name,
            unfinishedFromLegs(jobs, batchId)
          )
        );
      }
    }
  }

  for (const job of jobs) {
    if (isRunJob(job)) rows.push(...organelleRowsFor(job));
  }

  for (const job of jobs) {
    if (!isDownloadJob(job)) continue;
    const row = downloadRow(job);
    if (row) rows.push(row);
  }

  return rows;
}
