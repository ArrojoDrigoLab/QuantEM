/**
 * One state for an adaptation run, from a job and an adapter that can disagree.
 *
 * Cancel is the case this exists for. The Adapt wizard invites you to walk away
 * from work it describes as "minutes to tens of minutes", so cancelling is a
 * likely path, not a rare one — and it used to be a dead end. The job row went
 * CANCELLED, the `Adapter` stayed RUNNING forever, and step 4 rendered a
 * progress bar and a CANCELLED badge with no button to start again, while steps
 * 5 and 6 stayed disabled because they gate on `status === "SUCCESS"`. The only
 * way back was to pick a different segmentation.
 *
 * The backend half is fixed: `reconcile_domain_objects_for_cancelled_job` now
 * concludes the adapter. But `Adapter` has no CANCELLED state — adding one is a
 * migration plus every screen that renders a status — so a cancelled run
 * arrives as `FAILED` carrying the sentence in `CANCELLED_DETAIL`. That
 * sentence is the only surviving evidence after a reload, because the wizard
 * drops the job id the moment the adapter reaches a terminal status.
 *
 * So: read the job when it is still in hand, fall back to the adapter's own
 * message, and default to "failed" when neither says cancelled. Calling a crash
 * a cancellation would be the worse error of the two, so the match has to fail
 * closed.
 */

import type { Adapter } from "@/shared/types/finetune";
import type { Job } from "@/shared/types/jobs";

/**
 * How a cancelled run identifies itself once the job row is gone.
 *
 * Anchored to the start of `quantem.jobs.failure_reconcile.CANCELLED_DETAIL`
 * ("Cancelled before it finished, so it produced no result. …"). A real
 * training failure's message is a Python exception, which does not begin this
 * way; if the constant is ever reworded the run reads as a plain failure, which
 * is the safe direction to be wrong in.
 */
const CANCELLED_DETAIL_PREFIX = /^\s*cancelled\b/i;

export interface AdaptRunOutcome {
  /** The status to render, or null when nothing has been started here. */
  status: Job["status"] | Adapter["status"] | null;
  /** True while the run may still change. */
  running: boolean;
  /** True once the run stopped without producing an adapter. */
  concluded: boolean;
  /** True when that stop was a cancellation rather than a failure. */
  cancelled: boolean;
  /** The one sentence to print about it. Null unless something went wrong. */
  message: string | null;
}

const NOT_STARTED: AdaptRunOutcome = {
  status: null,
  running: false,
  concluded: false,
  cancelled: false,
  message: null,
};

/** The last non-empty line of a traceback: the exception, not the frames. */
function lastTracebackLine(traceback: string | undefined): string | null {
  if (!traceback) return null;
  const lines = traceback
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length > 0 ? lines[lines.length - 1] : null;
}

export function resolveAdaptRunOutcome(
  job: Job | null | undefined,
  adapter: Adapter | null | undefined
): AdaptRunOutcome {
  if (!job && !adapter) return NOT_STARTED;

  const adapterError = adapter?.error?.trim() || null;
  const cancelled =
    job?.status === "CANCELLED" ||
    (adapter?.status === "FAILED" &&
      adapterError !== null &&
      CANCELLED_DETAIL_PREFIX.test(adapterError));

  if (cancelled) {
    return {
      status: "CANCELLED",
      running: false,
      concluded: true,
      cancelled: true,
      message:
        adapterError ||
        "Cancelled before it finished, so it produced no result. Nothing was saved.",
    };
  }

  // The adapter has settled: it is the durable record and the job is history.
  if (adapter?.status === "SUCCESS" || adapter?.status === "FAILED") {
    return {
      status: adapter.status,
      running: false,
      concluded: adapter.status === "FAILED",
      cancelled: false,
      message:
        adapter.status === "FAILED"
          ? adapterError ||
            lastTracebackLine(job?.error_traceback) ||
            "The run failed without reporting a reason."
          : null,
    };
  }

  if (job?.status === "FAILED") {
    return {
      status: "FAILED",
      running: false,
      concluded: true,
      cancelled: false,
      message:
        adapterError ||
        lastTracebackLine(job.error_traceback) ||
        "The run failed without reporting a reason.",
    };
  }

  const status = job?.status ?? adapter?.status ?? null;
  return {
    status,
    running:
      status === "PENDING" || status === "RUNNING" || status === "RETRY",
    concluded: false,
    cancelled: false,
    message: null,
  };
}
