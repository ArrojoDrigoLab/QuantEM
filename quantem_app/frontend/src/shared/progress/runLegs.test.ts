/**
 * One job, several organelles: the row model (package P4).
 *
 * The progress plumbing already drew three kinds of line -- per-organelle,
 * aggregate, download -- and it built the per-organelle ones from one job each.
 * The moment a run over four organelles became one job, that collapsed four
 * lines into one and made the aggregate row disappear (it only appeared with
 * more than one *job* in the wave). These pin the repair.
 */

import { describe, expect, it } from "vitest";
import {
  buildProgressRows,
  legRow,
} from "@/shared/progress/runProgress";
import type { JobBatchProgress, JobQueueItem } from "@/shared/types/jobs";
import type { RunLeg } from "@/shared/types/runs";

function leg(partial: Partial<RunLeg> = {}): RunLeg {
  const units_done = partial.units_done ?? 19;
  const units_total = partial.units_total ?? 88;
  return {
    segmentation_id: "seg-nucleus",
    name: "Nucleus",
    status: "RUNNING",
    units_done,
    units_total,
    unit_label: "tile",
    percent: Math.round((1000 * units_done) / units_total) / 10,
    ...partial,
  };
}

function batch(partial: Partial<JobBatchProgress> = {}): JobBatchProgress {
  return {
    batch_id: "asset:1:abc",
    unit_label: "tile",
    units_done: 877,
    units_total: 946,
    units_abandoned: 0,
    units_reachable: 946,
    percent: 92.7,
    runs_total: 1,
    runs_unplanned: 0,
    runs_pending: 0,
    runs_running: 1,
    runs_succeeded: 0,
    runs_failed: 0,
    runs_cancelled: 0,
    complete: false,
    eta_seconds: 120,
    runs: [],
    ...partial,
  };
}

function imageRunJob(legs: RunLeg[], partial: Partial<JobQueueItem> = {}): JobQueueItem {
  return {
    id: "job-1",
    type: "run_segmentation_for_image",
    task_label: "Segment this image",
    status: "RUNNING",
    progress: 92.7,
    cancel_requested: false,
    queue_name: "p4_full",
    resource_class: "gpu",
    created_at: "2026-08-10T00:00:00Z",
    image: { id: "asset-1", display_name: "Grid2_Cell04" },
    segmentation: null,
    batch_id: "asset:1:abc",
    batch_progress: batch(),
    run_legs: legs,
    progress_stage: "inference",
    ...partial,
  };
}

describe("one job, several organelles", () => {
  it("draws one line per organelle, not one line per job", () => {
    const job = imageRunJob([
      leg({
        segmentation_id: "seg-mito",
        name: "Mitochondria",
        status: "SUCCESS",
        units_done: 858,
        units_total: 858,
        percent: 100,
      }),
      leg(),
    ]);
    const rows = buildProgressRows([job]);
    const organelles = rows.filter((row) => row.kind === "organelle");
    expect(organelles.map((row) => row.name)).toEqual([
      "Mitochondria",
      "Nucleus",
    ]);
  });

  it("shows the aggregate even though the wave is a single job row", () => {
    const job = imageRunJob([
      leg({ segmentation_id: "seg-mito", name: "Mitochondria" }),
      leg(),
    ]);
    const rows = buildProgressRows([job]);
    expect(rows[0].kind).toBe("aggregate");
    expect(rows[0].name).toBe("Everything on Grid2_Cell04");
  });

  it("does not show an aggregate for a run of one organelle", () => {
    const job = imageRunJob([leg()]);
    const rows = buildProgressRows([job]);
    expect(rows.some((row) => row.kind === "aggregate")).toBe(false);
  });

  it("counts unfinished organelles, not unfinished job rows", () => {
    const job = imageRunJob([
      leg({ segmentation_id: "a", name: "Mitochondria", status: "SUCCESS" }),
      leg({ segmentation_id: "b", name: "Nucleus", status: "FAILED" }),
      leg({ segmentation_id: "c", name: "Lipid Droplets", status: "CANCELLED" }),
    ]);
    const rows = buildProgressRows([job]);
    // The rollup says runs_failed 0 -- the job itself is still running -- and
    // the honest clause is about organelles.
    expect(rows[0].detail).toContain("2 of 3 did not finish");
  });

  it("leaves a single-organelle job exactly as it was", () => {
    const job = imageRunJob([], {
      type: "run_segmentation_full_task",
      run_legs: null,
      segmentation: { id: "seg-1", name: "Mitochondria" },
      unit_progress: {
        done: 32,
        total: 56,
        label: "tile",
        percent: 57.1,
        stage: "inference",
        eta_seconds: null,
      },
      batch_progress: batch({ runs_total: 1 }),
    });
    const rows = buildProgressRows([job]);
    expect(rows.map((row) => row.kind)).toEqual(["organelle"]);
    expect(rows[0].name).toBe("Mitochondria");
    expect(rows[0].detail).toContain("32 of 56 tiles");
  });
});

describe("one organelle's line", () => {
  const job = imageRunJob([]);

  it("is tiles-primary while it runs", () => {
    const row = legRow(job, leg());
    expect(row.showPercentText).toBe(true);
    expect(row.detail).toBe("19 of 88 tiles");
    expect(row.glyph).toBe("●");
  });

  it("quotes the plan, not a guess, while it waits", () => {
    const row = legRow(job, leg({ status: "PENDING", units_done: 0 }));
    expect(row.detail).toBe("waiting to start · 0 of 88 tiles");
    expect(row.percent).toBeNull();
    expect(row.glyph).toBe("○");
  });

  it("keeps the count it reached when it stops, and says why", () => {
    const row = legRow(job, leg({ status: "FAILED", units_done: 18 }));
    expect(row.detail).toBe("stopped at 18 of 88 tiles · this one did not finish");
    expect(row.tone).toBe("warning");
    // A square, not a hollow circle: "never started" and "started and stopped"
    // are opposite facts.
    expect(row.glyph).toBe("■");
  });

  it("says who stopped it when the user did", () => {
    const row = legRow(job, leg({ status: "CANCELLED", units_done: 4 }));
    expect(row.detail).toContain("you stopped this one");
  });

  it("fills the bar only when the organelle has finished", () => {
    const row = legRow(
      job,
      leg({ status: "SUCCESS", units_done: 88, percent: 100 })
    );
    expect(row.percent).toBe(100);
    expect(row.detail).toBe("88 of 88 tiles · finished");
  });

  it("cannot report more tiles than the organelle planned", () => {
    const row = legRow(job, leg({ units_done: 999, units_total: 88 }));
    expect(row.detail).toBe("88 of 88 tiles · finishing up");
  });
});
