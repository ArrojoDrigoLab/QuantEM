/**
 * The run panel on the labeling screen.
 *
 * This is the screen a user is actually on while a run goes: they pressed the
 * button here and they are waiting here. What it used to show was one row per
 * job reading the job type and `Math.round(job.progress)%` -- "Run full-image
 * segmentation  5%" -- while the tile counts the backend had been writing since
 * wave 0b reached no screen at all. Thirty seconds of sampled page text during
 * a real 56-tile run contained the word "tile" zero times.
 */

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SegmentationJobBanner } from "@/features/segmentation/screen/components/SegmentationJobBanner";
import type { JobBatchProgress, JobQueueItem } from "@/shared/types/jobs";

const ROLLUP: JobBatchProgress = {
  batch_id: "asset:img-1:abc",
  unit_label: "tile",
  units_done: 533,
  units_total: 946,
  units_abandoned: 0,
  units_reachable: 946,
  percent: 56.3,
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
};

function runJob(overrides: Partial<JobQueueItem> = {}): JobQueueItem {
  return {
    id: "job-mito",
    type: "run_segmentation_full_task",
    task_label: "Run full-image segmentation",
    status: "RUNNING",
    progress: 61.2,
    message: "DINO: 62% (Tile 531/858)",
    cancel_requested: false,
    queue_name: "p4_full",
    resource_class: "gpu",
    created_at: "2026-08-10T11:50:00Z",
    image: { id: "img-1", display_name: "Grid2_Cell04" },
    segmentation: { id: "seg-1", name: "Mitochondria" },
    progress_stage: "inference",
    unit_progress: {
      done: 531,
      total: 858,
      label: "tile",
      percent: 61.9,
      stage: "inference",
      eta_seconds: 260,
    },
    batch_progress: ROLLUP,
    ...overrides,
  } as JobQueueItem;
}

describe("the run panel", () => {
  it("is silent when nothing is running", () => {
    const { container } = render(<SegmentationJobBanner jobs={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("draws the owner's three kinds of row at once", () => {
    render(
      <SegmentationJobBanner
        imageName="Grid2_Cell04"
        jobs={[
          runJob(),
          runJob({
            id: "job-nucleus",
            segmentation: { id: "seg-2", name: "Nucleus" },
            unit_progress: {
              done: 2,
              total: 88,
              label: "tile",
              percent: 2.3,
              stage: "inference",
              eta_seconds: 60,
            },
          }),
          runJob({
            id: "job-dl",
            type: "install_model_pack",
            task_label: "Download model pack",
            segmentation: null,
            batch_progress: null,
            unit_progress: null,
            progress_stage: "downloading_model",
            model_pack: { id: "quantem:nucleus", title: "QuantEM — Nucleus" },
            download: {
              current_bytes: 118_000_000,
              total_bytes: 365_000_000,
              percent: 32.3,
            },
          }),
        ]}
      />
    );

    const aggregate = screen.getByTestId("run-progress-row-aggregate");
    expect(within(aggregate).getByText(/Everything on Grid2_Cell04/)).toBeInTheDocument();
    expect(within(aggregate).getByText(/533 of 946 tiles/)).toBeInTheDocument();

    const organelles = screen.getAllByTestId("run-progress-row-organelle");
    expect(organelles).toHaveLength(2);
    expect(within(organelles[0]).getByText(/531 of 858 tiles/)).toBeInTheDocument();
    expect(within(organelles[1]).getByText(/2 of 88 tiles/)).toBeInTheDocument();

    const download = screen.getByTestId("run-progress-row-download");
    expect(
      within(download).getByText("downloading the model — 118 of 365 MB")
    ).toBeInTheDocument();
  });

  it("puts no internal model name and no free-text tile ratio on screen", () => {
    const { container } = render(<SegmentationJobBanner jobs={[runJob()]} />);
    expect(container.textContent?.toLowerCase()).not.toContain("dino");
    expect(container.textContent).not.toContain("Tile 531/858");
  });

  it("shows the tile fraction rather than the whole-job percentage", () => {
    // 61.9 is 531 of the plan's 858. 61.2 is the same run's `progress`, which
    // also carries the model load and the saving either side of the tiles.
    const { container } = render(<SegmentationJobBanner jobs={[runJob()]} />);
    expect(container.textContent).toContain("62%");
    expect(container.textContent).not.toContain("61%");
  });
});

/**
 * The panel after the run stops.
 *
 * Measured at 1 Hz by the wave-0c verifier: non-empty at t=19.87 s reading
 * "Mitochondria 20% 11 of 56 tiles", empty at t=20.90 s, one second after the
 * cancel. The count the user reached was simply gone, from the screen they were
 * standing on when they pressed the button.
 */
describe("the run panel after a run stops", () => {
  it("keeps the count a cancelled run reached", () => {
    render(
      <SegmentationJobBanner
        imageName="Grid2_Cell04"
        jobs={[
          runJob({
            status: "CANCELLED",
            batch_progress: null,
            unit_progress: {
              done: 11,
              total: 56,
              label: "tile",
              percent: 19.6,
              stage: "inference",
              eta_seconds: null,
            },
          }),
        ]}
      />
    );

    const row = screen.getByTestId("run-progress-row-organelle");
    expect(
      within(row).getByText("stopped at 11 of 56 tiles · you stopped this one")
    ).toBeInTheDocument();
  });

  it("keeps the count a failed run reached", () => {
    render(
      <SegmentationJobBanner
        jobs={[
          runJob({
            status: "FAILED",
            batch_progress: null,
            unit_progress: {
              done: 4,
              total: 6,
              label: "tile",
              percent: 66.7,
              stage: "inference",
              eta_seconds: null,
            },
          }),
        ]}
      />
    );

    expect(
      within(screen.getByTestId("run-progress-row-organelle")).getByText(
        "stopped at 4 of 6 tiles · this one did not finish"
      )
    ).toBeInTheDocument();
  });

  it("stops calling itself Running once nothing is", () => {
    const panel = render(
      <SegmentationJobBanner jobs={[runJob({ status: "CANCELLED" })]} />
    );
    expect(panel.getByText("Last run")).toBeInTheDocument();
    expect(panel.queryByText("Running")).toBeNull();
  });
});
