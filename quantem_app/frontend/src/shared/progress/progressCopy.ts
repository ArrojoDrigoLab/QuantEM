/**
 * The words and numbers a progress row is written out of.
 *
 * Split out of `runProgress.ts` unchanged. This is the copy: what each phase is
 * called, how a count reads, how long is left, how many megabytes have arrived.
 * The row model that assembles these clauses is `progressRows.ts`; separating
 * them means changing a sentence and changing a denominator are two files.
 */

import { isLiveJob } from "@/shared/progress/jobPredicates";
import type { JobDownloadProgress, JobQueueItem } from "@/shared/types/jobs";

/**
 * What each phase is called on screen.
 *
 * `inference` has no phrase: during the tiles, the tiles *are* the sentence,
 * and adding "segmenting" beside "32 of 56 tiles" only takes up room.
 */
export const STAGE_PHRASES: Record<string, string> = {
  queued: "waiting to start",
  loading_model: "preparing the model and image",
  preparing_threshold: "preparing the threshold preview",
  extracting: "finding objects",
  saving: "saving objects",
  downloading_model: "downloading the model",
};

export function formatUnits(done: number, total: number, label: string): string {
  const noun = label || "tile";
  return `${done} of ${total} ${total === 1 ? noun : `${noun}s`}`;
}

/** "about 4 min left". Null when the estimate is not worth putting on screen. */
export function formatTimeLeft(seconds: number | null | undefined): string | null {
  if (seconds === null || seconds === undefined) return null;
  if (!Number.isFinite(seconds) || seconds <= 0) return null;
  if (seconds < 10) return "a few seconds left";
  if (seconds < 90) return `about ${Math.round(seconds / 5) * 5} seconds left`;
  const minutes = Math.round(seconds / 60);
  return minutes > 1 ? `about ${minutes} min left` : "about a minute left";
}

/**
 * "118 of 365 MB". Megabytes, not a percentage: the plan makes the download
 * indicator a different kind of row precisely so it cannot be read as the run.
 */
export function formatBytes(download: JobDownloadProgress): string {
  const mb = (value: number) => Math.round(value / 1e6);
  if (download.total_bytes === null || download.total_bytes === undefined) {
    return `${mb(download.current_bytes)} MB so far`;
  }
  return `${mb(download.current_bytes)} of ${mb(download.total_bytes)} MB`;
}

export function joinClauses(parts: Array<string | null | undefined>): string {
  return parts.filter((part): part is string => Boolean(part)).join(" · ");
}

/**
 * What to call the run panel, given what is on it.
 *
 * "Running" over a run that was cancelled forty seconds ago is the kind of
 * small lie that makes a reader stop trusting the rest of the panel. The panel
 * outlives the run on purpose — a row that vanishes the instant you press
 * Cancel takes the tile count you were watching with it — so the heading has to
 * follow the content.
 */
export function runPanelTitle(jobs: JobQueueItem[]): string {
  return jobs.some(isLiveJob) ? "Running" : "Last run";
}
