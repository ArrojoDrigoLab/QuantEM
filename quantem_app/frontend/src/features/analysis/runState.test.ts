/**
 * One state, never two — and the failure printed once.
 *
 * The reported screen showed all of this at the same moment: a history row
 * saying PENDING, a panel saying FAILED / "worker subprocess exited with code
 * 3221225794" printed twice, and "This run is pending. Results appear when it
 * finishes." Then the job scrolled out of the queue sidebar and the permanent
 * PENDING row was all that survived.
 */

import { describe, expect, it } from "vitest";
import {
  reconcileRunHistory,
  resolveAnalysisRunState,
} from "@/features/analysis/runState";
import type {
  AnalysisRun,
  AnalysisRunStatus,
  AnalysisRunSummary,
} from "@/shared/types/analysis";
import type { Job } from "@/shared/types/jobs";

const WORKER_DIED =
  "RuntimeError: worker subprocess exited with code 3221225794";

function makeRun(
  status: AnalysisRunStatus,
  overrides: Partial<AnalysisRun> = {}
): AnalysisRun {
  return {
    id: "run-1",
    segmentation_id: "seg-1",
    status,
    group: "",
    created_at: "2026-01-01T00:00:00Z",
    started_at: null,
    finished_at: null,
    params: {},
    pixel_size_nm: null,
    calibrated: null,
    composition: null,
    objects: null,
    points: null,
    distances: null,
    monte_carlo: null,
    monte_carlo_self_check: null,
    caveats: [],
    export_dir: "",
    exports: [],
    error: "",
    ...overrides,
  };
}

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-1",
    type: "analysis",
    priority: "NORMAL" as Job["priority"],
    status: "PENDING",
    progress: 0,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    attempts: 1,
    max_attempts: 1,
    next_run_at: "2026-01-01T00:00:00Z",
    payload_json: {},
    cancel_requested: false,
    resource_class: "CPU" as Job["resource_class"],
    queue_name: "default",
    tags: [],
    ...overrides,
  };
}

describe("resolveAnalysisRunState", () => {
  it("says FAILED once when the job died and the run row is still PENDING", () => {
    // The exact reported contradiction.
    const state = resolveAnalysisRunState(
      makeRun("PENDING"),
      makeJob({ status: "FAILED", error_traceback: `Traceback...\n${WORKER_DIED}` })
    );

    expect(state.status).toBe("FAILED");
    expect(state.active).toBe(false);
    expect(state.error).toBe(WORKER_DIED);
    // So the screen can admit the history row is behind rather than letting
    // the two disagree in silence.
    expect(state.reconciledFromJob).toBe(true);
  });

  it("prefers the run's own error over the job traceback, and returns one string", () => {
    const state = resolveAnalysisRunState(
      makeRun("FAILED", { error: "The tissue mask is empty." }),
      makeJob({ status: "FAILED", error_traceback: `Traceback...\n${WORKER_DIED}` })
    );

    expect(state.status).toBe("FAILED");
    expect(state.error).toBe("The tissue mask is empty.");
    expect(state.error).not.toContain(WORKER_DIED);
    expect(state.reconciledFromJob).toBe(false);
  });

  it("takes only the last traceback line, not the frames", () => {
    const state = resolveAnalysisRunState(
      makeRun("PENDING"),
      makeJob({
        status: "FAILED",
        error_traceback: `Traceback (most recent call last):\n  File "x.py", line 1\n${WORKER_DIED}\n`,
      })
    );

    expect(state.error).toBe(WORKER_DIED);
  });

  it("never reports FAILED without something to say", () => {
    const state = resolveAnalysisRunState(
      makeRun("PENDING"),
      makeJob({ status: "FAILED" })
    );

    expect(state.status).toBe("FAILED");
    expect(state.error).toBe("The job failed without reporting a reason.");
  });

  /**
   * A cancellation is a decision, not a fault.
   *
   * This used to return `FAILED` / "The job was cancelled.", so one click read
   * `CANCELLED` / *"You cancelled this run."* in the Adapt wizard and `FAILED`
   * / *"This run failed."* on the Analysis screen. The second sends someone
   * looking for a bug in something they chose to do. These assertions were
   * updated with the behaviour they pin: the run still will not finish, and
   * that part was always right.
   */
  describe("a run the user cancelled", () => {
    const CANCELLED_DETAIL =
      "Cancelled before it finished, so it produced no result. Nothing was " +
      "saved; start it again when you are ready.";

    it("is named a cancellation while the job row is still in hand", () => {
      const state = resolveAnalysisRunState(
        makeRun("RUNNING"),
        makeJob({ status: "CANCELLED" })
      );

      expect(state.status).toBe("CANCELLED");
      expect(state.cancelled).toBe(true);
      expect(state.active).toBe(false);
      // Still the thing the old assertion was protecting: the run row will sit
      // on RUNNING for ever otherwise.
      expect(state.reconciledFromJob).toBe(true);
      expect(state.error).toContain("Nothing was saved");
    });

    it("is still a cancellation after the job row is gone", () => {
      // All that survives is the sentence `reconcile_domain_objects_for_
      // cancelled_job` wrote into `AnalysisRun.error`.
      const state = resolveAnalysisRunState(
        makeRun("FAILED", { error: CANCELLED_DETAIL }),
        null
      );

      expect(state.status).toBe("CANCELLED");
      expect(state.cancelled).toBe(true);
      expect(state.error).toBe(CANCELLED_DETAIL);
      // The record already agrees, so there is nothing for the screen to warn
      // the history list about.
      expect(state.reconciledFromJob).toBe(false);
    });

    it("does not read a real failure as a cancellation", () => {
      // Fail closed: calling a crash a cancellation is the worse error.
      const state = resolveAnalysisRunState(
        makeRun("FAILED", { error: WORKER_DIED }),
        null
      );

      expect(state.status).toBe("FAILED");
      expect(state.cancelled).toBe(false);
      expect(state.error).toBe(WORKER_DIED);
    });

    it("keeps the server's own sentence rather than inventing one", () => {
      const state = resolveAnalysisRunState(
        makeRun("FAILED", { error: CANCELLED_DETAIL }),
        makeJob({ status: "CANCELLED" })
      );

      expect(state.error).toBe(CANCELLED_DETAIL);
    });
  });

  it("lets a settled run outrank a job the queue has trimmed", () => {
    // The durable record wins once it has one: a SUCCESS run whose job row is
    // gone, or reduced to a stale PENDING, still reads SUCCESS.
    const state = resolveAnalysisRunState(
      makeRun("SUCCESS"),
      makeJob({ status: "PENDING" })
    );

    expect(state.status).toBe("SUCCESS");
    expect(state.active).toBe(false);
    expect(state.error).toBeNull();
  });

  /**
   * The badge lagged its own message.
   *
   * `AnalysisRun.status` is only written when the handler moves on, so a run
   * mid-write reported PENDING while its job reported RUNNING with "writing
   * export bundle" as the message -- and the panel rendered the status from the
   * first and the message from the second, one line apart.
   */
  it("takes RUNNING from the live job while the run row still says PENDING", () => {
    const state = resolveAnalysisRunState(
      makeRun("PENDING"),
      makeJob({ status: "RUNNING", progress: 80, message: "writing export bundle" })
    );

    expect(state.status).toBe("RUNNING");
    expect(state.active).toBe(true);
    expect(state.error).toBeNull();
  });

  it("does not downgrade a RUNNING run because its job was requeued", () => {
    // A retry puts the job back to PENDING. Flickering the badge on that would
    // be a second way of disagreeing with the message beside it.
    const state = resolveAnalysisRunState(
      makeRun("RUNNING"),
      makeJob({ status: "PENDING", message: "retry queued" })
    );

    expect(state.status).toBe("RUNNING");
  });

  it("carries no error on a run that has not failed", () => {
    for (const status of ["PENDING", "RUNNING", "SUCCESS"] as AnalysisRunStatus[]) {
      const state = resolveAnalysisRunState(makeRun(status), makeJob());
      expect(state.error).toBeNull();
    }
  });

  it("reports the job alone before the run row has been read back", () => {
    const state = resolveAnalysisRunState(null, makeJob({ status: "RUNNING" }));

    expect(state.status).toBe("RUNNING");
    expect(state.active).toBe(true);
  });

  it("says nothing when there is neither a run nor a job", () => {
    const state = resolveAnalysisRunState(null, null);

    expect(state.status).toBeNull();
    expect(state.active).toBe(false);
    expect(state.cancelled).toBe(false);
    expect(state.error).toBeNull();
  });
});

/**
 * The history badge and the panel above it are one run.
 *
 * `AnalysisRunSummary.status` is the row as the worker last wrote it, and the
 * worker only writes when the handler moves on -- so a run *mid-write* read
 * `PENDING` in the history beside a panel reading "writing export bundle". Two
 * claims about one run, a hand's width apart, and the one that looked permanent
 * was the wrong one.
 */
describe("reconcileRunHistory", () => {
  function summary(overrides: Partial<AnalysisRunSummary> = {}): AnalysisRunSummary {
    return {
      id: "run-1",
      status: "PENDING",
      group: "",
      created_at: "2026-01-01T00:00:00Z",
      started_at: null,
      finished_at: null,
      export_dir: "",
      error: "",
      n_objects: null,
      calibrated: null,
      n_caveats: 0,
      ...overrides,
    };
  }

  it("gives the selected row the status the panel is showing", () => {
    const rows = reconcileRunHistory(
      [summary()],
      "run-1",
      resolveAnalysisRunState(
        makeRun("PENDING"),
        makeJob({ status: "RUNNING", message: "writing export bundle" })
      )
    );

    expect(rows[0].displayStatus).toBe("RUNNING");
    // The stored value is left alone: this reconciles what is rendered, not
    // what the server said.
    expect(rows[0].status).toBe("PENDING");
  });

  it("names the selected row's cancellation the same way the panel does", () => {
    const rows = reconcileRunHistory(
      [summary({ status: "PENDING" })],
      "run-1",
      resolveAnalysisRunState(makeRun("PENDING"), makeJob({ status: "CANCELLED" }))
    );

    expect(rows[0].displayStatus).toBe("CANCELLED");
  });

  it("leaves every other row exactly as the server sent it", () => {
    const rows = reconcileRunHistory(
      [summary({ id: "run-1", status: "SUCCESS" }), summary({ id: "run-2" })],
      "run-2",
      resolveAnalysisRunState(makeRun("PENDING"), makeJob({ status: "RUNNING" }))
    );

    expect(rows[0].displayStatus).toBe("SUCCESS");
    expect(rows[1].displayStatus).toBe("RUNNING");
  });

  it("changes nothing when no run is selected, or nothing is known", () => {
    const nothing = resolveAnalysisRunState(null, null);

    expect(reconcileRunHistory([summary()], null, nothing)[0].displayStatus).toBe(
      "PENDING"
    );
    expect(reconcileRunHistory([summary()], "run-1", nothing)[0].displayStatus).toBe(
      "PENDING"
    );
  });
});
