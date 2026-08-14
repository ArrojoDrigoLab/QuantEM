import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { JobQueueSidebar } from "@/features/library/components/JobQueueSidebar";
import { ApiRequestError } from "@/shared/api/core/http";
import type { Job, JobQueueItem, JobQueueStatus } from "@/shared/types/jobs";
import {
  cancelJob,
  clearDoneJobs,
  deleteJob,
  getJob,
  getJobQueueStatus,
  retryJob,
} from "@/shared/api/jobs";

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
    worker: {
      scheduler_in_process: true,
    },
    generated_at: "2026-03-11T12:00:00Z",
    ...overrides,
  };
}

function makeFailedJob(overrides: Partial<JobQueueItem> = {}): JobQueueItem {
  return {
    id: "job-1",
    type: "upload_image_pipeline",
    task_label: "Process upload",
    status: "FAILED",
    progress: 100,
    message: "failed: RuntimeError: boom",
    cancel_requested: false,
    queue_name: "p2_upload",
    resource_class: "cpu",
    created_at: "2026-03-11T11:50:00Z",
    started_at: "2026-03-11T11:51:00Z",
    finished_at: "2026-03-11T11:52:00Z",
    image: { id: "img-1", display_name: "Image 1" },
    segmentation: null,
    ...overrides,
  };
}

function makeQueueJob(overrides: Partial<JobQueueItem> = {}): JobQueueItem {
  return {
    id: "job-q-1",
    type: "run_segmentation_roi_task",
    task_label: "Run ROI",
    status: "PENDING",
    progress: 0,
    message: "",
    cancel_requested: false,
    queue_name: "p1_feedback",
    resource_class: "cpu",
    created_at: "2026-03-11T11:50:00Z",
    started_at: null,
    finished_at: null,
    image: { id: "img-1", display_name: "Image 1" },
    segmentation: {
      id: "seg-1",
      name: "Mitochondria",
      internal_name: "quantem_internal_mito",
      short_name: "Mito",
      long_name: "Mitochondria",
    },
    ...overrides,
  };
}

function makeCompletedJob(overrides: Partial<JobQueueItem> = {}): JobQueueItem {
  return {
    ...makeFailedJob({
      id: "job-success-1",
      task_label: "Completed task",
      status: "SUCCESS",
      message: "done",
    }),
    ...overrides,
  };
}

/** What `GET /api/jobs/<id>/` returns — the only place `result_json` lives. */
function makeJobDetail(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-success-1",
    type: "run_segmentation_full_task",
    priority: "default",
    status: "SUCCESS",
    progress: 100,
    message: "done",
    created_at: "2026-03-11T11:50:00Z",
    updated_at: "2026-03-11T11:52:00Z",
    started_at: "2026-03-11T11:51:00Z",
    finished_at: "2026-03-11T11:52:00Z",
    attempts: 1,
    max_attempts: 3,
    next_run_at: "2026-03-11T11:50:00Z",
    payload_json: {},
    result_json: null,
    error_traceback: "",
    cancel_requested: false,
    resource_class: "gpu",
    queue_name: "p3_batch",
    tags: [],
    ...overrides,
  };
}

describe("JobQueueSidebar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(cancelJob).mockResolvedValue({ status: "cancel_requested" });
    vi.mocked(clearDoneJobs).mockResolvedValue({
      deleted: 0,
      cleared_statuses: [],
    });
    vi.mocked(deleteJob).mockResolvedValue(undefined);
    vi.mocked(retryJob).mockResolvedValue({
      status: "queued",
      job_id: "job-1",
    });
    // The queue payload carries no result_json, so the panel asks for the
    // detail of each completed row it renders. Default: a job with no advice.
    vi.mocked(getJob).mockImplementation((jobId: string) =>
      Promise.resolve(makeJobDetail({ id: jobId }))
    );
  });

  it("offers no worker-restart control", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(makeStatus());

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
    await screen.findByRole("heading", { name: "Task Queues" });

    // `POST /api/jobs/worker/restart/` is not a route; a button that 404s
    // silently is worse than no button.
    expect(screen.queryByRole("button", { name: /restart/i })).toBeNull();
  });

  it("does not raise an alarm from the scheduler flag", async () => {
    // `scheduler_in_process` reads false in configurations where jobs run fine
    // (checked against a live server: jobs completed while it stayed false), so
    // it must not drive a "nothing will start" warning.
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({
        worker: { scheduler_in_process: false },
        completed: [makeCompletedJob()],
      })
    );

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);

    expect(await screen.findByText("Completed task")).toBeInTheDocument();
    expect(screen.getByText("Live queue status")).toBeInTheDocument();
    expect(screen.queryByText(/will not start/i)).toBeNull();
    expect(screen.queryByText(/scheduler/i)).toBeNull();
  });

  it("renders an HTML error document as a short message, not markup", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.mocked(getJobQueueStatus).mockRejectedValue(
      new ApiRequestError("<!DOCTYPE html><html><body><h1>Forbidden (403)</h1></body></html>", {
        status: 403,
      })
    );

    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);

    const message = await screen.findByText(/task queue could not be loaded/i);
    expect(message.textContent).toContain("HTTP 403");
    expect(message.textContent).not.toContain("<");
  });

  it("renders a retry button for failed jobs and calls the retry endpoint", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({
        failed: [makeFailedJob()],
      })
    );

    const user = userEvent.setup();
    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);

    const retryButton = await screen.findByRole("button", { name: "Retry" });
    await user.click(retryButton);

    await waitFor(() => {
      expect(retryJob).toHaveBeenCalledWith("job-1");
    });
  });

  it("shows a pending retry state while the retry request is in flight", async () => {
    const retryRequest = {
      resolve: null as ((value: { status: "queued"; job_id: string }) => void) | null,
    };
    vi.mocked(getJobQueueStatus)
      .mockResolvedValueOnce(
        makeStatus({
          failed: [makeFailedJob()],
        })
      )
      .mockResolvedValue(makeStatus());
    vi.mocked(retryJob).mockImplementation(
      () =>
        new Promise((resolve) => {
          retryRequest.resolve = resolve;
        })
    );

    const user = userEvent.setup();
    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);

    await user.click(await screen.findByRole("button", { name: "Retry" }));
    expect(screen.getByRole("button", { name: "Retrying..." })).toBeDisabled();

    if (!retryRequest.resolve) {
      throw new Error("Retry resolver was not registered.");
    }
    retryRequest.resolve({ status: "queued", job_id: "job-1" });

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Retrying..." })).toBeNull();
    });
  });

  it("limits queued tasks to six visible rows and expands in batches of six", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({
        queues: [
          {
            queue_name: "p2_upload",
            display_name: "P2 Upload",
            pending: Array.from({ length: 8 }, (_, index) =>
              makeQueueJob({
                id: `job-q-${index + 1}`,
                type: "upload_image_pipeline",
                task_label: `Queued task ${index + 1}`,
                segmentation: null,
              })
            ),
          },
        ],
      })
    );

    const user = userEvent.setup();
    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);

    expect(await screen.findByText("Queued task 1")).toBeInTheDocument();
    expect(screen.getByText("Queued task 6")).toBeInTheDocument();
    expect(screen.queryByText("Queued task 7")).toBeNull();
    expect(screen.queryByText("Queued task 8")).toBeNull();

    await user.click(
      screen.getByRole("button", { name: "Show 2 more queued tasks" })
    );

    expect(screen.getByText("Queued task 7")).toBeInTheDocument();
    expect(screen.getByText("Queued task 8")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Show 2 more queued tasks" })
    ).toBeNull();
  });

  /**
   * "Remove" is the only destructive control in this panel and the only exit a
   * queued job has -- `POST .../cancel/` refuses anything that is not RUNNING.
   * It deletes the job row outright, so whatever that job was carrying has no
   * queue entry left to explain it, and the endpoint now concludes the domain
   * object on the way out. The confirmation said only "This will not run the
   * task", which is a fact about the job and silent about the record.
   */
  describe("removing a queued task", () => {
    async function openRemoveDialog(job: JobQueueItem) {
      vi.mocked(getJobQueueStatus).mockResolvedValue(
        makeStatus({
          queues: [
            { queue_name: "p4_full", display_name: "P4 Background", pending: [job] },
          ],
        })
      );
      const user = userEvent.setup();
      render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
      await user.click(await screen.findByRole("button", { name: "Remove" }));
      return { user, dialog: await screen.findByRole("dialog") };
    }

    it("says what happens to the segmentation the run belonged to", async () => {
      const { dialog } = await openRemoveDialog(
        makeQueueJob({
          type: "run_segmentation_full_task",
          task_label: "Run full-image segmentation",
        })
      );

      expect(dialog).toHaveTextContent("marked Failed");
      expect(dialog).toHaveTextContent("Objects already in it are kept");
      // Three queued rows can carry the same task_label, so the dialog has to
      // say which image and which segmentation is about to be stranded.
      expect(dialog).toHaveTextContent("Image 1 · Mitochondria");
    });

    it("says the adapter is concluded so the wizard stops waiting on it", async () => {
      const { dialog } = await openRemoveDialog(
        makeQueueJob({
          type: "train_organelle_adapter",
          task_label: "Adapt model to your data",
        })
      );

      expect(dialog).toHaveTextContent("Adapt wizard stops waiting on it");
      expect(dialog).toHaveTextContent("No weights were written");
    });

    it("says an analysis run stops sitting at Pending", async () => {
      const { dialog } = await openRemoveDialog(
        makeQueueJob({ type: "run_analysis", task_label: "Run analysis" })
      );

      expect(dialog).toHaveTextContent("rather than left sitting at Pending");
    });

    it("says plainly when nothing is at stake", async () => {
      const { dialog } = await openRemoveDialog(
        makeQueueJob({
          type: "rebuild_segmentation_overlay",
          task_label: "Rebuild segmentation overlay",
        })
      );

      expect(dialog).toHaveTextContent("Nothing is lost");
    });

    it("invents no consequence for a job type it does not know", async () => {
      // A build newer than this screen must not have a promise made for it.
      const { dialog } = await openRemoveDialog(
        makeQueueJob({ type: "some_future_job", task_label: "Something new" })
      );

      expect(dialog).toHaveTextContent("Image 1 · Mitochondria");
      expect(dialog).not.toHaveTextContent("marked Failed");
      expect(dialog).not.toHaveTextContent("Nothing is lost");
    });

    it("removes nothing until the dialog is confirmed", async () => {
      const { user } = await openRemoveDialog(
        makeQueueJob({ type: "run_analysis", task_label: "Run analysis" })
      );

      expect(deleteJob).not.toHaveBeenCalled();
      await user.click(screen.getByRole("button", { name: "Keep" }));
      expect(deleteJob).not.toHaveBeenCalled();
    });

    it("removes the job when confirmed", async () => {
      const { user } = await openRemoveDialog(
        makeQueueJob({ id: "job-q-9", type: "run_analysis" })
      );

      await user.click(
        within(screen.getByRole("dialog")).getByRole("button", { name: "Remove" })
      );

      await waitFor(() => expect(deleteJob).toHaveBeenCalledWith("job-q-9"));
    });
  });

  it("keeps running tasks unbounded while failed and completed expand independently", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      makeStatus({
        running: Array.from({ length: 8 }, (_, index) =>
          makeQueueJob({
            id: `job-running-${index + 1}`,
            type: "upload_image_pipeline",
            task_label: `Running task ${index + 1}`,
            status: "RUNNING",
            progress: 20,
            started_at: "2026-03-11T11:51:00Z",
          })
        ),
        failed: Array.from({ length: 8 }, (_, index) =>
          makeFailedJob({
            id: `job-failed-${index + 1}`,
            task_label: `Failed task ${index + 1}`,
          })
        ),
        completed: Array.from({ length: 8 }, (_, index) =>
          makeCompletedJob({
            id: `job-completed-${index + 1}`,
            task_label: `Completed task ${index + 1}`,
          })
        ),
      })
    );

    const user = userEvent.setup();
    render(<JobQueueSidebar isOpen onClose={vi.fn()} />);

    expect(await screen.findByText("Running task 8")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /running tasks/i })
    ).toBeNull();

    // "Stopped", not "Failed": the queue puts CANCELLED in this list too.
    const failedSection = screen
      .getByRole("heading", { name: "Stopped" })
      .closest("section");
    const completedSection = screen
      .getByRole("heading", { name: "Completed" })
      .closest("section");

    expect(failedSection).not.toBeNull();
    expect(completedSection).not.toBeNull();

    expect(within(failedSection!).getByText("Failed task 6")).toBeInTheDocument();
    expect(within(failedSection!).queryByText("Failed task 7")).toBeNull();
    expect(
      within(completedSection!).getByText("Completed task 6")
    ).toBeInTheDocument();
    expect(within(completedSection!).queryByText("Completed task 7")).toBeNull();

    await user.click(
      within(failedSection!).getByRole("button", {
        name: "Show 2 more stopped tasks",
      })
    );

    expect(within(failedSection!).getByText("Failed task 7")).toBeInTheDocument();
    expect(
      within(completedSection!).queryByText("Completed task 7")
    ).toBeNull();

    await user.click(
      within(completedSection!).getByRole("button", {
        name: "Show 2 more completed tasks",
      })
    );

    expect(
      within(completedSection!).getByText("Completed task 7")
    ).toBeInTheDocument();
    expect(
      within(completedSection!).getByText("Completed task 8")
    ).toBeInTheDocument();
  });

  describe("a finished job's own advice", () => {
    /**
     * `Job.result_json.next_steps` has always been written and never rendered.
     *
     * A re-run over a proofread image completes in seconds having added
     * nothing, and the queue said `completed: no new objects` -- true, and
     * silent about the fact that this is the *expected* outcome (a candidate
     * confirmed work stays above model preview output, which is what protects
     * proofreading) and about what to do instead of lowering the threshold.
     * Those three sentences went into the job row and stopped there.
     */
    const ADVICE = [
      "Nothing changed: the 41 object(s) you have already labelled here are exactly as they were.",
      "Rejected model proposals are not added again. Confirmed outlines stay unchanged above any new model preview; accepting that preview later merges strong overlaps or removes the confirmed pixels from it.",
      "If you think objects were missed, run over an area you have not labelled yet rather than lowering the threshold over one you have.",
    ];

    it("renders the next steps a completed job recorded", async () => {
      vi.mocked(getJobQueueStatus).mockResolvedValue(
        makeStatus({
          completed: [
            makeCompletedJob({
              id: "job-empty-rerun",
              message: "completed: no new objects.",
            }),
          ],
        })
      );
      vi.mocked(getJob).mockResolvedValue(
        makeJobDetail({
          id: "job-empty-rerun",
          result_json: { segment_count: 0, found_objects: false, next_steps: ADVICE },
        })
      );

      render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
      await screen.findByText("completed: no new objects.");

      for (const step of ADVICE) {
        expect(await screen.findByText(step)).toBeInTheDocument();
      }
    });

    it("asks for each visible completed job once, not once per poll", async () => {
      vi.mocked(getJobQueueStatus).mockResolvedValue(
        makeStatus({ completed: [makeCompletedJob({ id: "job-a" })] })
      );
      vi.mocked(getJob).mockResolvedValue(
        makeJobDetail({ id: "job-a", result_json: { next_steps: ADVICE } })
      );

      render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
      await screen.findByText(ADVICE[0]);

      // The queue refetches every three seconds and hands back a new array of
      // the same ids; a naive effect would refetch the detail of every visible
      // job on every tick.
      const callsAfterFirstRender = vi.mocked(getJob).mock.calls.length;
      expect(callsAfterFirstRender).toBe(1);
      await waitFor(() => {
        expect(vi.mocked(getJobQueueStatus).mock.calls.length).toBeGreaterThan(0);
      });
      expect(vi.mocked(getJob).mock.calls.length).toBe(callsAfterFirstRender);
    });

    it("does not ask about completed jobs it is not showing", async () => {
      // Six rows are rendered; the endpoint returns up to a hundred.
      vi.mocked(getJobQueueStatus).mockResolvedValue(
        makeStatus({
          completed: Array.from({ length: 20 }, (_unused, i) =>
            makeCompletedJob({ id: `job-${i}`, task_label: `Completed task ${i}` })
          ),
        })
      );

      render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
      await screen.findByText("Completed task 0");

      await waitFor(() => {
        expect(vi.mocked(getJob).mock.calls.length).toBe(6);
      });
      const asked = vi.mocked(getJob).mock.calls.map(([id]) => id);
      expect(asked).toEqual(["job-0", "job-1", "job-2", "job-3", "job-4", "job-5"]);
    });

    it("renders nothing extra for a job that recorded no advice", async () => {
      vi.mocked(getJobQueueStatus).mockResolvedValue(
        makeStatus({ completed: [makeCompletedJob({ id: "job-quiet" })] })
      );

      const { container } = render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
      await screen.findByText("Completed task");

      await waitFor(() => {
        expect(vi.mocked(getJob)).toHaveBeenCalledWith("job-quiet");
      });
      expect(container.querySelector(".job-queue-next-steps")).toBeNull();
    });

    it("survives a detail fetch that fails", async () => {
      // The row's message is already on screen and is the load-bearing part.
      vi.mocked(getJobQueueStatus).mockResolvedValue(
        makeStatus({ completed: [makeCompletedJob({ id: "job-broken" })] })
      );
      vi.mocked(getJob).mockRejectedValue(new Error("gone"));

      render(<JobQueueSidebar isOpen onClose={vi.fn()} />);
      expect(await screen.findByText("Completed task")).toBeInTheDocument();
      expect(await screen.findByText("done")).toBeInTheDocument();
    });
  });
});
