/**
 * One state for a run, from two sources that can disagree.
 *
 * The analysis screen showed, at the same moment, a history row reading
 * PENDING, a panel reading FAILED with `worker subprocess exited with code
 * 3221225794` printed twice, and the sentence "This run is pending. Results
 * appear when it finishes." Three claims, two of them wrong, and the one the
 * reader is most likely to believe -- the permanent history row -- was the
 * least true: after the job scrolled out of the queue sidebar, PENDING was all
 * that was left.
 *
 * The disagreement is real and not purely cosmetic: an `AnalysisRun` row is
 * filled in by the worker, so a worker that dies never writes the terminal
 * status. The backend reconciles this (see the run/job reconciliation work),
 * but the client still has two objects in hand and must not render both.
 *
 * The rule, in one place so every panel agrees:
 *
 *  1. A run that has settled describes itself. It is the durable record, it is
 *     what the history row shows, and it outlives the job.
 *  2. Otherwise the job speaks for it. A job that has died is the only witness
 *     to a run that will never move off PENDING on its own.
 *  3. With neither, there is nothing to say.
 *
 * `error` is resolved the same way and is returned *once*, so a caller that
 * renders `state.error` cannot print it twice by also reaching for the job's
 * traceback.
 *
 * **Cancelling is not failing, and it has to be called the same thing on both
 * screens.** One cancel used to read `CANCELLED` / *"You cancelled this run."*
 * in the Adapt wizard and `FAILED` / *"This run failed."* on the Analysis
 * screen, because this function folded a cancelled job into FAILED while
 * `resolveAdaptRunOutcome` did not. The second wording sends someone looking
 * for a bug in a decision they made. `AnalysisRun` has no CANCELLED status --
 * adding one is a migration -- so a cancelled run arrives as FAILED carrying
 * `quantem.jobs.failure_reconcile.CANCELLED_DETAIL`, exactly as an `Adapter`
 * does, and is recognised the same way: by the job while it is in hand, by that
 * sentence afterwards. The match fails closed, because calling a crash a
 * cancellation is the worse of the two errors.
 */

import type { AnalysisRun, AnalysisRunStatus } from "@/shared/types/analysis";
import type { Job } from "@/shared/types/jobs";

/**
 * How a cancelled run identifies itself once the job row is gone.
 *
 * Anchored to the start of `CANCELLED_DETAIL` ("Cancelled before it finished,
 * so it produced no result. ..."), which the analysis reconciler writes into
 * `AnalysisRun.error`. A real failure's message is a Python exception, which
 * does not begin this way. Deliberately the same rule as
 * `adaptRunOutcome.CANCELLED_DETAIL_PREFIX`: two screens reading one backend
 * constant.
 */
const CANCELLED_DETAIL_PREFIX = /^\s*cancelled\b/i;

/** What a cancelled analysis run says for itself, if the server said nothing. */
const CANCELLED_FALLBACK =
  "Cancelled before it finished, so it produced no result. Nothing was saved; " +
  "start it again when you are ready.";

export interface AnalysisRunState {
  /**
   * The single status to render. Null when nothing has been selected.
   *
   * `"CANCELLED"` is not an `AnalysisRunStatus` the server can store; it is
   * this reconciler naming what happened, so the Analysis screen and the Adapt
   * wizard say the same word about the same click.
   */
  status: AnalysisRunStatus | "CANCELLED" | null;
  /** True while the run may still change: nothing terminal has been seen. */
  active: boolean;
  /** True when the run stopped because somebody cancelled it. */
  cancelled: boolean;
  /**
   * Why it stopped. Null unless something went wrong — and a cancellation
   * counts, because the sentence explaining that nothing was saved is the whole
   * of what the screen has to say about it.
   */
  error: string | null;
  /**
   * True when the job says the run stopped but the run row has not caught up.
   *
   * Worth surfacing rather than hiding: it means the durable record is behind,
   * and the run's own row will still read PENDING until the server reconciles.
   */
  reconciledFromJob: boolean;
}

const EMPTY: AnalysisRunState = {
  status: null,
  active: false,
  cancelled: false,
  error: null,
  reconciledFromJob: false,
};

function isTerminal(status: AnalysisRunStatus): boolean {
  return status === "SUCCESS" || status === "FAILED";
}

/**
 * The last non-empty line of a traceback: the exception, not the frames.
 *
 * A raw traceback in a panel is unreadable and the useful sentence is always
 * at the bottom.
 */
function lastTracebackLine(traceback: string | undefined): string | null {
  if (!traceback) return null;
  const lines = traceback
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length > 0 ? lines[lines.length - 1] : null;
}

export function resolveAnalysisRunState(
  run: AnalysisRun | null | undefined,
  job: Job | null | undefined
): AnalysisRunState {
  const runError = run?.error?.trim() || null;

  // Checked before either terminal branch, because both would call it FAILED.
  // A cancelled job is evidence on its own; once the job row is gone the
  // reconciler's sentence in `run.error` is all that is left of it.
  const cancelled =
    job?.status === "CANCELLED" ||
    (run?.status === "FAILED" &&
      runError !== null &&
      CANCELLED_DETAIL_PREFIX.test(runError));

  if (cancelled) {
    return {
      status: "CANCELLED",
      active: false,
      cancelled: true,
      error: runError || CANCELLED_FALLBACK,
      // The run row is only behind if it has not been failed yet. Once the
      // reconciler has written CANCELLED_DETAIL into it, the record agrees.
      reconciledFromJob: run?.status !== "FAILED",
    };
  }

  // The run has settled: it is the record, and the job is history.
  if (run && isTerminal(run.status)) {
    return {
      status: run.status,
      active: false,
      cancelled: false,
      error:
        run.status === "FAILED"
          ? runError || lastTracebackLine(job?.error_traceback) || "The run failed."
          : null,
      reconciledFromJob: false,
    };
  }

  // The job died and the run row has not caught up. Nothing else will ever say
  // so, which is exactly how a permanent PENDING row came to be the only thing
  // left on screen.
  if (job && job.status === "FAILED") {
    return {
      status: "FAILED",
      active: false,
      cancelled: false,
      error:
        runError ||
        lastTracebackLine(job.error_traceback) ||
        "The job failed without reporting a reason.",
      reconciledFromJob: true,
    };
  }

  if (run) {
    // Unsettled, so the *job* is the live witness and the run row is only
    // written when the handler moves on. A run mid-write reported `PENDING`
    // while the job reported `RUNNING` with "writing export bundle" as its
    // message -- and the screen rendered the status from one and the message
    // from the other, side by side.
    //
    // Only ever an upgrade to RUNNING. A job that has gone back to PENDING or
    // RETRY says nothing useful about a run already marked RUNNING, and
    // downgrading on it would make the badge flicker on every retry.
    const liveStatus: AnalysisRunStatus =
      job?.status === "RUNNING" && run.status === "PENDING" ? "RUNNING" : run.status;
    return {
      status: liveStatus,
      active: true,
      cancelled: false,
      error: null,
      reconciledFromJob: false,
    };
  }

  if (job) {
    // A job in flight before its run row has been read back.
    return {
      status: job.status === "RUNNING" ? "RUNNING" : "PENDING",
      active: true,
      cancelled: false,
      error: null,
      reconciledFromJob: false,
    };
  }

  return EMPTY;
}

/**
 * The history list, told what the reconciler knows.
 *
 * `AnalysisRunSummary.status` is the row as the server last wrote it, and the
 * worker only writes it when the handler moves on -- so a run *mid-write*
 * showed `PENDING` in the history beside a panel reading "writing export
 * bundle", two claims about one run, a hand's width apart. The panel is right:
 * it reads the live job. Nothing is invented here; the row that is currently
 * selected is given the status the panel above it is already showing, and every
 * other row is left exactly as the server sent it.
 */
export function reconcileRunHistory<T extends { id: string; status: AnalysisRunStatus }>(
  runs: T[],
  runId: string | null,
  state: AnalysisRunState
): Array<T & { displayStatus: AnalysisRunStatus | "CANCELLED" }> {
  return runs.map((run) => ({
    ...run,
    displayStatus:
      runId !== null && run.id === runId && state.status !== null
        ? state.status
        : run.status,
  }));
}
