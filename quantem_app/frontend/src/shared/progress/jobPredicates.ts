/**
 * What kind of job a queue row is, and whether it is still going.
 *
 * Split out of `runProgress.ts` unchanged. These are the questions every
 * progress surface asks before it decides what to draw, and they are the
 * questions a package adding a new job type has to answer — so they are their
 * own file rather than a header on top of the row model.
 */

import type { JobQueueItem } from "@/shared/types/jobs";

/**
 * Job types that walk countable units and are drawn as a bar with a count.
 *
 * Originally "a segmentation run over tiles", and for the first three that is
 * still what it means. The set is what `buildProgressRows` filters on, so it is
 * really the answer to "does this job get a structured row?", and a job that
 * reports `unit_progress` and is not in here draws nothing at all.
 */
const RUN_JOB_TYPES = new Set([
  "run_segmentation_roi_task",
  "run_segmentation_full_task",
  // One run over one image covering several organelles. A run job like any
  // other from every surface's point of view; it simply draws one line per
  // organelle instead of one line for itself. Omitting it here made a
  // multi-organelle run invisible in the Tasks drawer and the run panel alike.
  "run_segmentation_for_image",
  // Fine-tuning. Not a segmentation pass -- it walks training steps, and under
  // cross-validation it walks rounds of them -- but it reports the same
  // `unit_progress` shape and is drawn by the same rows, so leaving it out
  // would have made a run that takes minutes show a bare percentage in the
  // Tasks drawer while the Fine-Tune dialog beside it showed steps and an ETA.
  // It reaches no other surface: the labeling screen's run panel filters on its
  // own `PROCESSING_BANNER_JOB_TYPES`, which does not include this.
  "train_organelle_adapter",
]);

const DOWNLOAD_JOB_TYPE = "install_model_pack";

/** Fine-tuning, as opposed to a segmentation pass, among the run jobs. */
export function isFineTuneJob(job: JobQueueItem): boolean {
  return job.type === "train_organelle_adapter";
}

export function isRunJob(job: JobQueueItem): boolean {
  return RUN_JOB_TYPES.has(job.type);
}

export function isDownloadJob(job: JobQueueItem): boolean {
  return job.type === DOWNLOAD_JOB_TYPE;
}

/** Still going, or waiting its turn — as opposed to concluded. */
export function isLiveJob(job: JobQueueItem): boolean {
  return (
    job.status === "RUNNING" || job.status === "PENDING" || job.status === "RETRY"
  );
}

/**
 * A segmentation run that stopped: the user cancelled it, or it failed.
 *
 * The queue reports both in one list, because from the queue's point of view
 * they are the same thing — a job that will not finish. They are not the same
 * thing to a person, which is why `organelleRow` says which one happened, and
 * why the surfaces that render these rows have to ask this question rather than
 * treating everything in that list as a failure.
 */
export function isStoppedRunJob(job: JobQueueItem): boolean {
  return isRunJob(job) && (job.status === "FAILED" || job.status === "CANCELLED");
}

/** True when this job has something structured to draw, rather than only text. */
export function hasStructuredProgress(job: JobQueueItem): boolean {
  return Boolean(job.unit_progress || job.download);
}
