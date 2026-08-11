/**
 * A cancelled adaptation has to be legible as a cancellation.
 *
 * `Adapter` has no CANCELLED state, so the backend concludes a cancelled run as
 * `FAILED` carrying `CANCELLED_DETAIL`. If the wizard reads only the status it
 * tells the user their deliberate cancel crashed, and sends them looking for a
 * bug that is not there.
 */

import { describe, expect, it } from "vitest";
import { resolveAdaptRunOutcome } from "@/features/improve/runOutcome";
import type { Adapter } from "@/shared/types/finetune";
import type { Job } from "@/shared/types/jobs";

/** The exact sentence `quantem.jobs.failure_reconcile.CANCELLED_DETAIL` writes. */
const CANCELLED_DETAIL =
  "Cancelled before it finished, so it produced no result. Nothing was saved; " +
  "start it again when you are ready.";

function adapter(overrides: Partial<Adapter> = {}): Adapter {
  return {
    id: "ad-1",
    base_model: "quantem:mito",
    name: "mito @ Liver",
    status: "RUNNING",
    mode: "head",
    steps: 300,
    trainable_params: null,
    segmentation_id: "seg-1",
    split_mode: "within-image",
    train_crop_names: [],
    heldout_crop_names: [],
    sweep: {},
    calibrated_threshold: null,
    heldout_dice: null,
    verified_reload: false,
    train_seconds: null,
    applied_at: null,
    created_at: "2026-02-01T00:00:00Z",
    error: "",
    caveats: [],
    ...overrides,
  };
}

function job(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-1",
    type: "train_organelle_adapter",
    status: "RUNNING",
    progress: 40,
    message: "training",
    created_at: "2026-02-01T00:00:00Z",
    ...overrides,
  } as Job;
}

describe("resolveAdaptRunOutcome", () => {
  it("says nothing when no run has been started", () => {
    expect(resolveAdaptRunOutcome(null, null)).toMatchObject({
      status: null,
      running: false,
      concluded: false,
      cancelled: false,
    });
  });

  it("reads a live job as running", () => {
    expect(resolveAdaptRunOutcome(job(), adapter())).toMatchObject({
      status: "RUNNING",
      running: true,
      concluded: false,
    });
  });

  it("calls a cancelled job cancelled while the job row is still in hand", () => {
    const outcome = resolveAdaptRunOutcome(
      job({ status: "CANCELLED" }),
      adapter({ status: "RUNNING" })
    );
    expect(outcome.cancelled).toBe(true);
    expect(outcome.concluded).toBe(true);
    expect(outcome.status).toBe("CANCELLED");
  });

  /**
   * After a reload the wizard has dropped the job id, so the reconciled
   * adapter's own sentence is the only evidence left that this was a cancel.
   */
  it("recognises a cancel from the reconciled adapter alone", () => {
    const outcome = resolveAdaptRunOutcome(
      null,
      adapter({ status: "FAILED", error: CANCELLED_DETAIL })
    );
    expect(outcome.cancelled).toBe(true);
    expect(outcome.status).toBe("CANCELLED");
    expect(outcome.message).toBe(CANCELLED_DETAIL);
  });

  it("does not mistake a real failure for a cancel", () => {
    const outcome = resolveAdaptRunOutcome(
      null,
      adapter({
        status: "FAILED",
        error: "ModelArchitectureUnavailable: timm is not installed.",
      })
    );
    expect(outcome.cancelled).toBe(false);
    expect(outcome.concluded).toBe(true);
    expect(outcome.status).toBe("FAILED");
    expect(outcome.message).toContain("ModelArchitectureUnavailable");
  });

  it("never leaves a failure without a sentence", () => {
    const outcome = resolveAdaptRunOutcome(
      job({ status: "FAILED", error_traceback: undefined }),
      adapter({ status: "FAILED", error: "" })
    );
    expect(outcome.message).toBe("The run failed without reporting a reason.");
  });

  it("prefers the last traceback line over the frames above it", () => {
    const outcome = resolveAdaptRunOutcome(
      job({
        status: "FAILED",
        error_traceback: "Traceback:\n  File x, line 1\nRuntimeError: out of memory",
      }),
      null
    );
    expect(outcome.message).toBe("RuntimeError: out of memory");
  });

  it("treats a finished adapter as the record, not a conclusion to recover from", () => {
    const outcome = resolveAdaptRunOutcome(
      job({ status: "SUCCESS" }),
      adapter({ status: "SUCCESS" })
    );
    expect(outcome).toMatchObject({
      status: "SUCCESS",
      running: false,
      concluded: false,
      cancelled: false,
      message: null,
    });
  });
});
