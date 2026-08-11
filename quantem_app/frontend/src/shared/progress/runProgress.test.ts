import { describe, expect, it } from "vitest";
import {
  aggregateRow,
  buildAggregateRows,
  buildProgressRows,
  downloadRow,
  formatBytes,
  formatTimeLeft,
  isStoppedRunJob,
  organelleRow,
  runPanelTitle,
} from "@/shared/progress/runProgress";
import type {
  JobBatchProgress,
  JobQueueItem,
  JobUnitProgress,
} from "@/shared/types/jobs";

function units(partial: Partial<JobUnitProgress> = {}): JobUnitProgress {
  const done = partial.done ?? 32;
  const total = partial.total ?? 56;
  return {
    done,
    total,
    label: "tile",
    percent: Math.round((1000 * done) / total) / 10,
    stage: "inference",
    eta_seconds: null,
    ...partial,
  };
}

function runJob(partial: Partial<JobQueueItem> = {}): JobQueueItem {
  return {
    id: "job-1",
    type: "run_segmentation_full_task",
    task_label: "Run full-image segmentation",
    status: "RUNNING",
    progress: 29.8,
    cancel_requested: false,
    queue_name: "p4_full",
    resource_class: "gpu",
    created_at: "2026-08-10T00:00:00Z",
    image: { id: "asset-1", display_name: "Grid2_Cell04" },
    segmentation: { id: "seg-1", name: "Mitochondria" },
    progress_stage: "inference",
    unit_progress: units(),
    ...partial,
  } as JobQueueItem;
}

function batch(partial: Partial<JobBatchProgress> = {}): JobBatchProgress {
  return {
    batch_id: "asset:a:1",
    unit_label: "tile",
    units_done: 34,
    units_total: 62,
    units_abandoned: 0,
    units_reachable: 62,
    percent: 54.8,
    runs_total: 2,
    runs_unplanned: 0,
    runs_pending: 0,
    runs_running: 2,
    runs_succeeded: 0,
    runs_failed: 0,
    runs_cancelled: 0,
    complete: false,
    eta_seconds: 260,
    runs: [],
    ...partial,
  };
}

describe("the per-organelle line", () => {
  it("is tiles-primary and never repeats the model's name for itself", () => {
    const row = organelleRow(runJob());
    expect(row.name).toBe("Mitochondria");
    expect(row.detail).toBe("32 of 56 tiles");
    expect(row.percent).toBe(57.1);
    expect(JSON.stringify(row).toLowerCase()).not.toContain("dino");
  });

  it("puts the percentage and the count on the same divisor", () => {
    // The job's own `progress` is 29.8 (17 of 57: the tiles plus the work
    // either side of them). The line must not quote that beside "17 of 56".
    const row = organelleRow(
      runJob({ progress: 29.8, unit_progress: units({ done: 17, total: 56 }) })
    );
    expect(row.percent).toBe(30.4);
    expect(Math.round(row.percent!)).toBe(Math.round((100 * 17) / 56));
    expect(row.detail).toContain("17 of 56 tiles");
  });

  it("says loading the model with the denominator, and claims no fraction done", () => {
    const row = organelleRow(
      runJob({
        progress_stage: "loading_model",
        unit_progress: units({ done: 0, total: 357, percent: 0 }),
      })
    );
    expect(row.detail).toBe("loading the model — 0 of 357 tiles");
    // No bar at all: this is the window that used to read as a frozen 5%.
    expect(row.percent).toBeNull();
  });

  it("names the phase after the tiles instead of sitting full and silent", () => {
    const row = organelleRow(
      runJob({
        progress_stage: "extracting",
        unit_progress: units({ done: 56, total: 56, percent: 100 }),
      })
    );
    expect(row.percent).toBe(100);
    expect(row.detail).toBe("56 of 56 tiles · finding objects");
  });

  it("reports where a cancelled run actually stopped, never rounded up", () => {
    const row = organelleRow(
      runJob({
        status: "CANCELLED",
        unit_progress: units({ done: 41, total: 56 }),
      })
    );
    expect(row.detail).toBe("stopped at 41 of 56 tiles · you stopped this one");
    expect(row.tone).toBe("warning");
  });

  it("reports where a failed run stopped, with the same count a cancelled one gets", () => {
    // The verifier's finding: a cancelled run reappeared under "Failed" as the
    // bare word "cancelled" and a failed run never quoted a tile count at all,
    // so neither of the two ways a run can stop said how much work was done.
    const row = organelleRow(
      runJob({ status: "FAILED", unit_progress: units({ done: 18, total: 56 }) })
    );
    expect(row.detail).toBe("stopped at 18 of 56 tiles · this one did not finish");
    expect(row.tone).toBe("warning");
    // The bar sits where the run actually got to: not full (it did not finish)
    // and not empty (it did 18 tiles).
    expect(row.percent).toBe(32.1);
  });

  it("invents no denominator for a run that stopped before it had a tiling plan", () => {
    // A run that dies while loading the model has no plan to quote. "0 of 0
    // tiles" would be a number the run never produced.
    const row = organelleRow(
      runJob({ status: "FAILED", unit_progress: null, progress_stage: "loading_model" })
    );
    expect(row.detail).toBe(
      "stopped before it counted any tiles · this one did not finish"
    );
    expect(row.percent).toBeNull();
  });

  it("marks a stopped run differently from one that has not started", () => {
    const stopped = organelleRow(runJob({ status: "CANCELLED" }));
    const waiting = organelleRow(runJob({ status: "PENDING" }));
    expect(stopped.glyph).not.toBe(waiting.glyph);
  });

  it("builds a row for a concluded run, which is how it reaches a screen", () => {
    // `buildProgressRows` is the only entry point both surfaces use. While it
    // was fed nothing but `running`, the stopped copy above was unreachable.
    const rows = buildProgressRows(
      [runJob({ status: "CANCELLED", unit_progress: units({ done: 18, total: 56 }) })],
      { includeAggregate: false }
    );
    expect(rows).toHaveLength(1);
    expect(rows[0].detail).toContain("stopped at 18 of 56 tiles");
  });

  it("carries the run's own estimate of what is left", () => {
    const row = organelleRow(
      runJob({ unit_progress: units({ done: 20, total: 56, eta_seconds: 260 }) })
    );
    expect(row.detail).toBe("20 of 56 tiles · about 4 min left");
  });
});

describe("the aggregate line", () => {
  it("is time-primary with tiles secondary", () => {
    const row = aggregateRow(batch(), "Grid2_Cell04");
    expect(row.name).toBe("Everything on Grid2_Cell04");
    expect(row.percent).toBe(54.8);
    expect(row.detail).toBe("about 4 min left · 34 of 62 tiles");
  });

  it("says out loud when the wave did not finish whole, and stops the bar short", () => {
    // The abandoned tiles stay in the denominator. They used to be taken out of
    // it, which put "100% · 59 of 59 tiles" on a wave that abandoned three and
    // "100% · 25 of 25 tiles" on one that abandoned ninety-three.
    const row = aggregateRow(
      batch({
        units_done: 59,
        units_total: 62,
        units_abandoned: 3,
        units_reachable: 59,
        percent: 95.2,
        runs_cancelled: 1,
        runs_succeeded: 1,
        runs_running: 0,
        complete: true,
        eta_seconds: null,
      })
    );
    expect(row.percent).toBe(95.2);
    expect(row.detail).toBe("59 of 62 tiles · 1 of 2 did not finish");
    expect(row.tone).toBe("warning");
  });

  it("reads the measured three-run wave honestly", () => {
    // mito cancelled at 19 of 56, nucleus 6 of 6, ER failed at 0 of 56.
    const row = aggregateRow(
      batch({
        units_done: 25,
        units_total: 118,
        units_abandoned: 93,
        units_reachable: 25,
        percent: 21.2,
        runs_total: 3,
        runs_running: 0,
        runs_succeeded: 1,
        runs_failed: 1,
        runs_cancelled: 1,
        complete: true,
        eta_seconds: null,
      }),
      "montage16real"
    );
    expect(row.name).toBe("Everything on montage16real");
    expect(row.percent).toBe(21.2);
    expect(row.detail).toBe("25 of 118 tiles · 2 of 3 did not finish");
  });

  it("draws no bar only when a run in the wave cannot say how big it is", () => {
    const row = aggregateRow(
      batch({ percent: null, units_done: 34, runs_unplanned: 1, runs_total: 3 })
    );
    expect(row.percent).toBeNull();
    expect(row.detail).toContain("34 tiles so far");
  });

  it("appears once per wave, not once per organelle in it", () => {
    const rollup = batch();
    const rows = buildAggregateRows([
      runJob({ id: "a", batch_progress: rollup }),
      runJob({ id: "b", batch_progress: rollup }),
    ]);
    expect(rows).toHaveLength(1);
  });

  it("stays away when the wave is one run, which it would only repeat", () => {
    const rows = buildAggregateRows([
      runJob({ batch_progress: batch({ runs_total: 1, runs_running: 1 }) }),
    ]);
    expect(rows).toHaveLength(0);
  });
});

describe("the model-download line", () => {
  const downloadJob = runJob({
    id: "dl-1",
    type: "install_model_pack",
    task_label: "Download model pack",
    segmentation: null,
    unit_progress: null,
    progress_stage: "downloading_model",
    model_pack: { id: "quantem:nucleus", title: "QuantEM — Nucleus" },
    download: { current_bytes: 118_000_000, total_bytes: 365_000_000, percent: 32.3 },
  });

  it("is bytes, its own glyph, and cannot be read as segmentation progress", () => {
    const row = downloadRow(downloadJob)!;
    expect(row.kind).toBe("download");
    expect(row.glyph).toBe("↓");
    expect(row.name).toBe("QuantEM — Nucleus");
    expect(row.detail).toBe("downloading the model — 118 of 365 MB");
    // No percentage text at all: the plan's separation is bytes, not percent.
    expect(row.showPercentText).toBe(false);
    expect(row.detail).not.toContain("%");
  });

  it("falls back to the job's own label for a pack this build cannot name", () => {
    const row = downloadRow({ ...downloadJob, model_pack: { id: "x:y", title: "" } })!;
    expect(row.name).toBe("Download model pack");
  });

  it("reports what has arrived when the size is not known", () => {
    expect(
      formatBytes({ current_bytes: 118_000_000, total_bytes: null, percent: null })
    ).toBe("118 MB so far");
  });
});

describe("the list", () => {
  it("is the aggregate, then each organelle, then the downloads", () => {
    const rollup = batch();
    const rows = buildProgressRows([
      runJob({ id: "mito", batch_progress: rollup }),
      runJob({
        id: "nucleus",
        segmentation: { id: "seg-2", name: "Nucleus" },
        batch_progress: rollup,
        unit_progress: units({ done: 2, total: 6 }),
      }),
      runJob({
        id: "dl",
        type: "install_model_pack",
        unit_progress: null,
        segmentation: null,
        model_pack: { id: "quantem:nucleus", title: "QuantEM — Nucleus" },
        download: { current_bytes: 1e6, total_bytes: 2e6, percent: 50 },
      }),
    ]);
    expect(rows.map((row) => row.kind)).toEqual([
      "aggregate",
      "organelle",
      "organelle",
      "download",
    ]);
  });
});

describe("what the panel is called", () => {
  it("says Running only while something is", () => {
    expect(runPanelTitle([runJob()])).toBe("Running");
    expect(runPanelTitle([runJob({ status: "PENDING" })])).toBe("Running");
  });

  it("stops saying Running over a run that has stopped", () => {
    // The panel outlives the run on purpose, so the heading has to follow the
    // content; "Running" over a cancelled run is a small lie that costs the
    // rest of the panel its credibility.
    expect(runPanelTitle([runJob({ status: "CANCELLED" })])).toBe("Last run");
    expect(runPanelTitle([runJob({ status: "FAILED" })])).toBe("Last run");
    expect(
      runPanelTitle([runJob({ status: "SUCCESS" }), runJob({ status: "RUNNING" })])
    ).toBe("Running");
  });
});

describe("telling the two ways a run stops apart", () => {
  it("counts a cancelled and a failed run, and nothing else", () => {
    expect(isStoppedRunJob(runJob({ status: "CANCELLED" }))).toBe(true);
    expect(isStoppedRunJob(runJob({ status: "FAILED" }))).toBe(true);
    expect(isStoppedRunJob(runJob({ status: "SUCCESS" }))).toBe(false);
    expect(isStoppedRunJob(runJob({ status: "RUNNING" }))).toBe(false);
    // A failed upload is not a run, and has no tiles to report.
    expect(
      isStoppedRunJob(runJob({ type: "upload_image_pipeline", status: "FAILED" }))
    ).toBe(false);
  });
});

describe("time phrasing", () => {
  it.each([
    [null, null],
    [0, null],
    [-4, null],
    [3, "a few seconds left"],
    [42, "about 40 seconds left"],
    [100, "about 2 min left"],
    [260, "about 4 min left"],
    [70, "about 70 seconds left"],
  ])("%s seconds reads as %s", (seconds, expected) => {
    expect(formatTimeLeft(seconds)).toBe(expected);
  });
});
