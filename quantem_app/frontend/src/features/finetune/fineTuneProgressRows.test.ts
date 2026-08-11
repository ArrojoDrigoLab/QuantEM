/**
 * The bar's words, including the one the owner called out: an ETA that does not
 * exist yet must read as an estimate pending, never as no time left.
 */

import { describe, expect, it } from "vitest";
import {
  ESTIMATING_TIME_LEFT,
  fineTuneProgressRow,
  fineTuneProgressRows,
} from "@/features/finetune/fineTuneProgressRows";
import { isFineTuneJob, isRunJob } from "@/shared/progress/jobPredicates";
import { buildProgressRows } from "@/shared/progress/progressRows";
import type { FineTuneProgress } from "@/shared/types/finetune";
import type { JobQueueItem } from "@/shared/types/jobs";

function progress(overrides: Partial<FineTuneProgress> = {}): FineTuneProgress {
  return {
    status: "RUNNING",
    stage: "training",
    step: 240,
    total_steps: 600,
    round: 1,
    total_rounds: 1,
    percent: 40,
    eta_seconds: 145,
    message: "Training on 12 tiles",
    error: "",
    ...overrides,
  };
}

describe("features/finetune/fineTuneProgressRows", () => {
  it("names the stage, the steps and the time left", () => {
    const row = fineTuneProgressRow(progress(), "Fasted liver mitochondria");
    expect(row.percent).toBe(40);
    expect(row.showPercentText).toBe(true);
    expect(row.detail).toBe("training · 240 of 600 steps · about 2 min left");
  });

  it("counts rounds when cross-validation is running, and not when it is not", () => {
    expect(
      fineTuneProgressRow(progress({ round: 2, total_rounds: 5 }), "n").detail
    ).toContain("round 2 of 5");
    // One round is not a dimension: "round 1 of 1" would invent one.
    expect(fineTuneProgressRow(progress(), "n").detail).not.toContain("round");
  });

  it("says it is still estimating rather than claiming no time left", () => {
    const row = fineTuneProgressRow(progress({ eta_seconds: null }), "n");
    expect(row.detail).toContain(ESTIMATING_TIME_LEFT);
    expect(row.detail).not.toContain("0 seconds");
    expect(row.detail).not.toContain("0s");
  });

  it("divides by the server's percent rather than recomputing one", () => {
    // Rounds and steps together: 38% is not 240/600, and the row must not
    // "correct" it to 40.
    const row = fineTuneProgressRow(
      progress({ percent: 38, round: 2, total_rounds: 5 }),
      "n"
    );
    expect(row.percent).toBe(38);
  });

  it("draws a full bar on success and a warning bar where it stopped", () => {
    const done = fineTuneProgressRow(
      progress({ status: "SUCCESS", step: 600, percent: 100 }),
      "n"
    );
    expect(done.percent).toBe(100);
    expect(done.detail).toBe("600 of 600 steps · finished");

    const failed = fineTuneProgressRow(
      progress({ status: "FAILED", step: 130, percent: 21 }),
      "n"
    );
    expect(failed.tone).toBe("warning");
    expect(failed.glyph).toBe("■");
    expect(failed.detail).toBe("stopped at 130 of 600 steps · this one did not finish");
  });

  it("states the size of the work before a worker has touched it", () => {
    const row = fineTuneProgressRow(
      progress({ status: "PENDING", step: 0, percent: null }),
      "n"
    );
    expect(row.percent).toBeNull();
    expect(row.detail).toBe("waiting to start · 0 of 600 steps");
  });

  it("has no rows before the first poll lands", () => {
    expect(fineTuneProgressRows(null, "n")).toEqual([]);
  });
});

describe("the fine-tune job in the shared queue surfaces", () => {
  const job = {
    id: "job-1",
    type: "train_organelle_adapter",
    task_label: "Adapt model to your data",
    status: "RUNNING",
    progress: 0.4,
    cancel_requested: false,
    queue_name: "p4_full",
    resource_class: "gpu",
    created_at: "2026-01-01T00:00:00Z",
    unit_progress: {
      done: 240,
      total: 600,
      label: "step",
      percent: 40,
      stage: "training",
      eta_seconds: 145,
    },
  } as unknown as JobQueueItem;

  it("is a run job, so the Tasks drawer draws it a real row", () => {
    expect(isRunJob(job)).toBe(true);
    expect(isFineTuneJob(job)).toBe(true);
    const rows = buildProgressRows([job], { includeAggregate: false });
    expect(rows).toHaveLength(1);
    expect(rows[0].detail).toContain("240 of 600 steps");
  });

  it("is still distinguishable from a segmentation pass", () => {
    expect(isFineTuneJob({ ...job, type: "run_segmentation_full_task" })).toBe(false);
  });
});
