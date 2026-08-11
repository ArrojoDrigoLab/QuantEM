/**
 * What the Tasks drawer shows while a segmentation run is in flight.
 *
 * Kept apart from `JobQueueSidebar.test.tsx` because it is about one thing: the
 * three indicators, and the defects they replace. During a real 56-tile run the
 * drawer showed a bar at `Math.round(job.progress)` and, under it,
 * `job.message` verbatim:
 *
 *     56%
 *     DINO: 57% (Tile 32/56)
 *
 * Three faults in one row. "DINO" is the foundation encoder's internal name.
 * The bar divides by 57 (the tiles plus the work either side of them) and the
 * text divides by 56 (the tiling plan), so they disagree by a point on the same
 * run. And the count reaches the screen by being written into free text and
 * read back out, rather than from the columns that hold it.
 */

import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { JobQueueSidebar } from "@/features/library/components/JobQueueSidebar";
import type {
  JobBatchProgress,
  JobQueueItem,
  JobQueueStatus,
} from "@/shared/types/jobs";
import { clearDoneJobs, getJob, getJobQueueStatus } from "@/shared/api/jobs";

vi.mock("@/shared/api/jobs", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/jobs")>(
    "@/shared/api/jobs"
  );
  return {
    ...actual,
    cancelJob: vi.fn(),
    clearDoneJobs: vi.fn(),
    deleteJob: vi.fn(),
    getJob: vi.fn(),
    getJobQueueStatus: vi.fn(),
    retryJob: vi.fn(),
  };
});

function makeStatus(overrides: Partial<JobQueueStatus> = {}): JobQueueStatus {
  return {
    running: [],
    queues: [],
    failed: [],
    completed: [],
    worker: { scheduler_in_process: true },
    generated_at: "2026-08-10T12:00:00Z",
    ...overrides,
  };
}

/**
 * A run in flight, with the numbers the verifier sampled off a real run:
 * `progress` 29.82 is 17 of 57, `unit_progress.percent` 30.4 is 17 of the
 * plan's 56. The stale free-text message is kept on the fixture on purpose --
 * the backend still writes a message, and the point is that the drawer no
 * longer renders it for a job that can count its own work.
 */
function makeRunningRunJob(overrides: Partial<JobQueueItem> = {}): JobQueueItem {
  return {
    id: "job-run-1",
    type: "run_segmentation_full_task",
    task_label: "Run full-image segmentation",
    status: "RUNNING",
    progress: 29.82456140350877,
    message: "DINO: 30% (Tile 17/56)",
    cancel_requested: false,
    queue_name: "p4_full",
    resource_class: "gpu",
    created_at: "2026-08-10T11:50:00Z",
    started_at: "2026-08-10T11:51:00Z",
    finished_at: null,
    image: { id: "img-1", display_name: "Grid2_Cell04" },
    segmentation: {
      id: "seg-1",
      name: "Mitochondria",
      internal_name: "quantem_internal_mito",
      short_name: "Mito",
      long_name: "Mitochondria",
    },
    progress_stage: "inference",
    unit_progress: {
      done: 17,
      total: 56,
      label: "tile",
      percent: 30.4,
      stage: "inference",
      eta_seconds: 260,
    },
    download: null,
    batch_id: "asset:img-1:abc",
    batch_progress: null,
    ...overrides,
  };
}

function makeRollup(overrides: Partial<JobBatchProgress> = {}): JobBatchProgress {
  return {
    batch_id: "asset:img-1:abc",
    unit_label: "tile",
    units_done: 19,
    units_total: 62,
    units_abandoned: 0,
    units_reachable: 62,
    percent: 30.6,
    runs_total: 2,
    runs_unplanned: 0,
    runs_pending: 0,
    runs_running: 2,
    runs_succeeded: 0,
    runs_failed: 0,
    runs_cancelled: 0,
    complete: false,
    eta_seconds: 300,
    runs: [],
    ...overrides,
  };
}

describe("the Tasks drawer during a run", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(clearDoneJobs).mockResolvedValue({
      deleted: 0,
      cleared_statuses: [],
    });
    vi.mocked(getJob).mockRejectedValue(new Error("not asked for in these tests"));
  });

  it("shows the tile count, taken from the structured field", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({ running: [makeRunningRunJob()] })
    );

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    const row = await screen.findByTestId("run-progress-row-organelle");

    expect(within(row).getByText(/17 of 56 tiles/)).toBeInTheDocument();
    expect(within(row).getByText("Mitochondria")).toBeInTheDocument();
    expect(within(row).getByText(/about 4 min left/)).toBeInTheDocument();
  });

  it("never puts the model's own name for itself on screen", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({ running: [makeRunningRunJob()] })
    );

    const { container } = render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    await screen.findByTestId("run-progress-row-organelle");

    expect(container.textContent?.toLowerCase()).not.toContain("dino");
    expect(container.textContent).not.toContain("Tile 17/56");
  });

  it("shows one percentage for the run, on the tiling plan's divisor", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({ running: [makeRunningRunJob()] })
    );

    const { container } = render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    await screen.findByTestId("run-progress-row-organelle");

    const percentages = (container.textContent ?? "").match(/\d+%/g) ?? [];
    // 30 is 17 of 56. The whole-job 29.8 must not be standing next to it.
    expect(percentages).toEqual(["30%"]);
  });

  it("says loading the model, with the denominator already known", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({
        running: [
          makeRunningRunJob({
            progress: 5,
            progress_stage: "loading_model",
            message: "Preparing inference workload",
            unit_progress: {
              done: 0,
              total: 56,
              label: "tile",
              percent: 0,
              stage: "loading_model",
              eta_seconds: null,
            },
          }),
        ],
      })
    );

    const { container } = render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    const row = await screen.findByTestId("run-progress-row-organelle");

    expect(
      within(row).getByText("loading the model — 0 of 56 tiles")
    ).toBeInTheDocument();
    // The frozen 5% is gone: there is no fraction to claim yet.
    expect(container.textContent).not.toContain("5%");
  });

  it("rolls every organelle on the image into one line, drawn once", async () => {
    const rollup = makeRollup();
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({
        running: [
          makeRunningRunJob({ id: "a", batch_progress: rollup }),
          makeRunningRunJob({
            id: "b",
            batch_progress: rollup,
            segmentation: { id: "seg-2", name: "Nucleus" },
            unit_progress: {
              done: 2,
              total: 6,
              label: "tile",
              percent: 33.3,
              stage: "inference",
              eta_seconds: null,
            },
          }),
        ],
      })
    );

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    const aggregates = await screen.findAllByTestId("run-progress-row-aggregate");

    expect(aggregates).toHaveLength(1);
    expect(within(aggregates[0]).getByText(/19 of 62 tiles/)).toBeInTheDocument();
    expect(
      within(aggregates[0]).getByText(/Everything on Grid2_Cell04/)
    ).toBeInTheDocument();
    expect(screen.getAllByTestId("run-progress-row-organelle")).toHaveLength(2);
  });

  it("keeps a model download visibly apart from the run", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({
        running: [
          makeRunningRunJob(),
          makeRunningRunJob({
            id: "job-dl",
            type: "install_model_pack",
            task_label: "Download model pack",
            segmentation: null,
            progress_stage: "downloading_model",
            unit_progress: null,
            model_pack: { id: "quantem:nucleus", title: "QuantEM — Nucleus" },
            download: {
              current_bytes: 118_000_000,
              total_bytes: 365_000_000,
              percent: 32.3,
            },
          }),
        ],
      })
    );

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    const download = await screen.findByTestId("run-progress-row-download");

    expect(
      within(download).getByText("downloading the model — 118 of 365 MB")
    ).toBeInTheDocument();
    // Bytes, never a percentage: that is what stops it reading as the run.
    expect(within(download).queryByText(/%/)).toBeNull();
    expect(screen.getByTestId("run-progress-row-organelle")).toBeInTheDocument();
  });

  it("leaves a job that counts nothing with its percentage and its sentence", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({
        running: [
          {
            ...makeRunningRunJob({
              id: "job-upload",
              type: "upload_image_pipeline",
              task_label: "Process upload",
              progress: 42,
              message: "Reading the file",
              segmentation: null,
              unit_progress: null,
              batch_progress: null,
              progress_stage: "",
            }),
          },
        ],
      })
    );

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);

    expect(await screen.findByText("Reading the file")).toBeInTheDocument();
    expect(screen.getByText("42%")).toBeInTheDocument();
    expect(screen.queryByTestId("run-progress-row-organelle")).toBeNull();
  });
});

/**
 * A run that stopped, in the drawer.
 *
 * The queue reports FAILED and CANCELLED in one list, and this drawer built
 * structured rows only out of `running`. So the moment a user pressed Cancel
 * their run left the top of the drawer, reappeared at the bottom under the
 * heading "Failed", and said, in full:
 *
 *     Run full-image segmentation   CANCELLED   Retry
 *     montage16real   Mitochondria   Just now
 *     cancelled
 *
 * The tile count it reached was in `unit_progress` on that very payload.
 */
describe("the Tasks drawer after a run stops", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(clearDoneJobs).mockResolvedValue({
      deleted: 0,
      cleared_statuses: [],
    });
    vi.mocked(getJob).mockRejectedValue(new Error("not asked for in these tests"));
  });

  function makeStoppedRunJob(overrides: Partial<JobQueueItem> = {}): JobQueueItem {
    return makeRunningRunJob({
      status: "CANCELLED",
      progress: 32.1,
      message: "cancelled",
      finished_at: "2026-08-10T11:53:00Z",
      unit_progress: {
        done: 18,
        total: 56,
        label: "tile",
        percent: 32.1,
        stage: "inference",
        eta_seconds: null,
      },
      ...overrides,
    });
  }

  it("says how far a cancelled run got, and who stopped it", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({ failed: [makeStoppedRunJob()] })
    );

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    const row = await screen.findByTestId("run-progress-row-organelle");

    expect(
      within(row).getByText("stopped at 18 of 56 tiles · you stopped this one")
    ).toBeInTheDocument();
    expect(within(row).getByText("Mitochondria")).toBeInTheDocument();
  });

  it("drops the one-word message a cancellation used to be reduced to", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({ failed: [makeStoppedRunJob()] })
    );

    const { container } = render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    await screen.findByTestId("run-progress-row-organelle");

    expect(container.querySelector(".job-queue-message")).toBeNull();
  });

  it("says how far a failed run got, and keeps the reason it failed", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({
        failed: [
          makeStoppedRunJob({
            status: "FAILED",
            message: "The model could not be loaded from disk.",
          }),
        ],
      })
    );

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    const row = await screen.findByTestId("run-progress-row-organelle");

    expect(
      within(row).getByText("stopped at 18 of 56 tiles · this one did not finish")
    ).toBeInTheDocument();
    // The reason is the only text that says what went wrong; the count does not
    // replace it.
    expect(
      screen.getByText("The model could not be loaded from disk.")
    ).toBeInTheDocument();
  });

  it("does not file a cancellation under a heading that calls it a failure", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({ failed: [makeStoppedRunJob()] })
    );

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    await screen.findByTestId("run-progress-row-organelle");

    expect(screen.getByRole("heading", { name: "Stopped" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Failed" })).toBeNull();
  });
});

describe("the Tasks drawer before a run starts", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getJob).mockResolvedValue(null as never);
    vi.mocked(clearDoneJobs).mockResolvedValue({
      deleted: 0,
      cleared_statuses: [],
    } as never);
  });

  /**
   * A queued run's tiling plan is written when it is enqueued, so the row can
   * say how much work is waiting. It could not before: `progress_units_total`
   * was first written by the run itself, so a queued organelle was a row with a
   * status and nothing else, and the wave rollup could not count it either.
   */
  it("says how many tiles a queued run is waiting to walk", async () => {
    const queued = makeRunningRunJob({
      id: "job-queued-1",
      status: "PENDING",
      started_at: null,
      progress: 0,
      progress_stage: "queued",
      message: "",
      unit_progress: {
        done: 0,
        total: 88,
        label: "tile",
        percent: 0,
        stage: "queued",
        eta_seconds: null,
      },
    });
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({
        queues: [
          { queue_name: "p4_full", display_name: "P4 Background", pending: [queued] },
        ],
      })
    );

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    const row = await screen.findByTestId("run-progress-row-organelle");

    expect(
      within(row).getByText("waiting to start · 0 of 88 tiles")
    ).toBeInTheDocument();
  });
});
