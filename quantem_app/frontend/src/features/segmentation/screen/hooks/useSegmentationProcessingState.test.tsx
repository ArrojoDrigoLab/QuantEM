import { act, renderHook, waitFor } from "@testing-library/react";
import type { PropsWithChildren } from "react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  makeSegmentation,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useSegmentationProcessingState } from "@/features/segmentation/screen/hooks/useSegmentationProcessingState";
import {
  markSegmentationComplete,
  unlockSegmentation,
} from "@/shared/api/segmentations/annotations";
import { getJobQueueStatus } from "@/shared/api/jobs";
import { rerunSegmentationRoi } from "@/shared/api/segmentations/rois";
import { ensureModelInstalled } from "@/features/models/ensureModelInstalled";
import type { JobQueueItem, JobQueueStatus } from "@/shared/types/jobs";

vi.mock("@/features/models/ensureModelInstalled", () => ({
  ensureModelInstalled: vi.fn(),
}));

function RouterWrapper({ children }: PropsWithChildren) {
  return <MemoryRouter>{children}</MemoryRouter>;
}

describe("useSegmentationProcessingState", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
    vi.mocked(ensureModelInstalled).mockImplementation(
      async (packId) => ({ id: packId }) as Awaited<ReturnType<typeof ensureModelInstalled>>
    );
  });

  it("refreshes segment views after marking a segmentation complete", async () => {
    const refreshSegmentViews = vi.fn(async () => {});
    const refetchSegmentations = vi.fn(async () => {});
    vi.mocked(markSegmentationComplete).mockResolvedValue(
      makeSegmentation({
        status_stage: "COMPLETED",
      })
    );

    const { result } = renderHook(
      () =>
        useSegmentationProcessingState({
          currentSegmentation: makeSegmentation(),
          activeSourceModel: null,
          supportsPointFeedback: false,
          supportsInstanceParams: false,
          currentInstanceParams: null,
          refetchSegmentations,
          refreshSegmentViews,
        }),
      {
        wrapper: RouterWrapper,
      }
    );

    await act(async () => {
      await result.current.handleToggleSegmentationComplete();
    });

    // No discard options: the complete endpoint keeps every object unless it is
    // explicitly asked otherwise, and this call path must never ask on its own.
    expect(markSegmentationComplete).toHaveBeenCalledWith("seg-1", undefined);
    expect(refreshSegmentViews).toHaveBeenCalledWith({ deferOverlayRefresh: true });
    expect(refetchSegmentations).not.toHaveBeenCalled();
  });

  it("passes the acknowledged discard count straight through", async () => {
    // The count comes from the confirmation, which read it live; the endpoint
    // returns 409 if it no longer matches, so this must not be re-derived here.
    const refreshSegmentViews = vi.fn(async () => {});
    vi.mocked(markSegmentationComplete).mockResolvedValue(
      makeSegmentation({ status_stage: "COMPLETED" })
    );

    const { result } = renderHook(
      () =>
        useSegmentationProcessingState({
          currentSegmentation: makeSegmentation(),
          activeSourceModel: null,
          supportsPointFeedback: false,
          supportsInstanceParams: false,
          currentInstanceParams: null,
          refetchSegmentations: vi.fn(async () => {}),
          refreshSegmentViews,
        }),
      { wrapper: RouterWrapper }
    );

    await act(async () => {
      await result.current.handleToggleSegmentationComplete({
        discardUnconfirmed: true,
        acknowledgedDiscardCount: 32,
      });
    });

    expect(markSegmentationComplete).toHaveBeenCalledWith("seg-1", {
      discardUnconfirmed: true,
      acknowledgedDiscardCount: 32,
    });
  });

  it("lets a refusal reach the caller instead of swallowing it", async () => {
    // The 409 exists to protect the confirmation dialog, so the dialog has to
    // see it. This used to be caught and logged, leaving a user staring at a
    // confirmation that did nothing and said nothing.
    const refreshSegmentViews = vi.fn(async () => {});
    vi.mocked(markSegmentationComplete).mockRejectedValue(
      new Error("stale count")
    );

    const { result } = renderHook(
      () =>
        useSegmentationProcessingState({
          currentSegmentation: makeSegmentation(),
          activeSourceModel: null,
          supportsPointFeedback: false,
          supportsInstanceParams: false,
          currentInstanceParams: null,
          refetchSegmentations: vi.fn(async () => {}),
          refreshSegmentViews,
        }),
      { wrapper: RouterWrapper }
    );

    await expect(
      result.current.handleToggleSegmentationComplete({
        discardUnconfirmed: true,
        acknowledgedDiscardCount: 7,
      })
    ).rejects.toThrow("stale count");
    expect(refreshSegmentViews).not.toHaveBeenCalled();
  });

  it("refreshes segment views after unlocking a completed segmentation", async () => {
    const refreshSegmentViews = vi.fn(async () => {});
    const refetchSegmentations = vi.fn(async () => {});
    vi.mocked(unlockSegmentation).mockResolvedValue(
      makeSegmentation({
        status_stage: "CANDIDATES_READY",
      })
    );

    const { result } = renderHook(
      () =>
        useSegmentationProcessingState({
          currentSegmentation: makeSegmentation({
            status_stage: "COMPLETED",
            is_complete: true,
          }),
          activeSourceModel: null,
          supportsPointFeedback: false,
          supportsInstanceParams: false,
          currentInstanceParams: null,
          refetchSegmentations,
          refreshSegmentViews,
        }),
      {
        wrapper: RouterWrapper,
      }
    );

    await act(async () => {
      await result.current.handleToggleSegmentationComplete();
    });

    expect(unlockSegmentation).toHaveBeenCalledWith("seg-1");
    expect(refreshSegmentViews).toHaveBeenCalledWith({ deferOverlayRefresh: true });
    expect(refetchSegmentations).not.toHaveBeenCalled();
  });

  it("tests the requested ROI and keeps its preview selected", async () => {
    const { result } = renderHook(
      () =>
        useSegmentationProcessingState({
          currentSegmentation: makeSegmentation(),
          activeSourceModel: "quantem:mito",
          supportsPointFeedback: false,
          supportsInstanceParams: false,
          currentInstanceParams: null,
          refetchSegmentations: vi.fn(async () => {}),
          refreshSegmentViews: vi.fn(async () => {}),
        }),
      { wrapper: RouterWrapper }
    );

    await waitFor(() => expect(result.current.segmentationRois).toHaveLength(2));
    await act(async () => {
      await result.current.handleRerunRoi("roi-existing");
    });

    expect(ensureModelInstalled).toHaveBeenCalledWith(
      "quantem:mito",
      expect.any(Object)
    );
    expect(rerunSegmentationRoi).toHaveBeenCalledWith(
      "seg-1",
      "roi-existing",
      "quantem:mito"
    );
    expect(result.current.previewRoiId).toBe("roi-existing");
  });
});

/**
 * What the run panel is given after a run stops.
 *
 * Cancelling the only run in a wave closes the wave, and the panel's job list
 * kept a concluded run only while its wave was still open. So one poll after
 * the cancel the row left the panel and took the tile count with it -- on the
 * screen the user pressed Cancel from.
 */
describe("the jobs the run panel is given", () => {
  const CANCELLED_RUN: JobQueueItem = {
    id: "job-cancelled",
    type: "run_segmentation_full_task",
    task_label: "Run full-image segmentation",
    status: "CANCELLED",
    progress: 32.1,
    message: "cancelled",
    cancel_requested: true,
    queue_name: "p4_full",
    resource_class: "gpu",
    created_at: "2026-01-01T00:00:00Z",
    started_at: "2026-01-01T00:00:10Z",
    finished_at: new Date(Date.now() - 20_000).toISOString(),
    image: { id: "img-1", display_name: "Image 1" },
    segmentation: { id: "seg-1", name: "Mitochondria" },
    progress_stage: "inference",
    unit_progress: {
      done: 18,
      total: 56,
      label: "tile",
      percent: 32.1,
      stage: "inference",
      eta_seconds: null,
    },
    download: null,
    // The wave closed with the run: nothing else in it is open, so there is no
    // rollup and `batch_id` alone cannot keep the row alive.
    batch_id: "asset:img-1:abc",
    batch_progress: null,
  };

  function statusWithStopped(job: JobQueueItem): JobQueueStatus {
    return {
      running: [],
      queues: [],
      failed: [job],
      completed: [],
      worker: { scheduler_in_process: true },
      generated_at: "2026-01-01T00:00:30Z",
    };
  }

  function renderProcessingState() {
    return renderHook(
      () =>
        useSegmentationProcessingState({
          currentSegmentation: makeSegmentation(),
          activeSourceModel: null,
          supportsPointFeedback: false,
          supportsInstanceParams: false,
          currentInstanceParams: null,
          refetchSegmentations: vi.fn(async () => {}),
          refreshSegmentViews: vi.fn(async () => {}),
        }),
      { wrapper: RouterWrapper }
    );
  }

  beforeEach(() => {
    setupSegmentationScreenTest();
  });

  it("keeps a run that was just cancelled, wave open or not", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(statusWithStopped(CANCELLED_RUN));

    const { result } = renderProcessingState();

    await waitFor(() => {
      expect(result.current.processingJobs).toHaveLength(1);
    });
    expect(result.current.processingJobs[0].id).toBe("job-cancelled");
    expect(result.current.shouldShowProcessingStatus).toBe(true);
  });

  it("keeps a run that just failed", async () => {
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      statusWithStopped({
        ...CANCELLED_RUN,
        id: "job-failed",
        status: "FAILED",
        cancel_requested: false,
        message: "failed",
      })
    );

    const { result } = renderProcessingState();

    await waitFor(() => {
      expect(result.current.processingJobs.map((job) => job.id)).toEqual([
        "job-failed",
      ]);
    });
  });

  it("lets a stopped run stop being news", async () => {
    // Not forever: an hour-old cancellation on a screen with nothing running is
    // history, and history belongs in the Tasks drawer.
    vi.mocked(getJobQueueStatus).mockResolvedValue(
      statusWithStopped({
        ...CANCELLED_RUN,
        finished_at: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
      })
    );

    const { result } = renderProcessingState();

    await waitFor(() => {
      expect(result.current.segmentationRois).not.toBeUndefined();
    });
    expect(result.current.processingJobs).toHaveLength(0);
  });

  it("still drops a run that succeeded once its wave is done", async () => {
    // A finished run only stays while its wave is open -- that behaviour is
    // deliberate and the linger must not quietly extend to it.
    vi.mocked(getJobQueueStatus).mockResolvedValue({
      running: [],
      queues: [],
      failed: [],
      completed: [{ ...CANCELLED_RUN, id: "job-done", status: "SUCCESS" }],
      worker: { scheduler_in_process: true },
      generated_at: "2026-01-01T00:00:30Z",
    });

    const { result } = renderProcessingState();

    await waitFor(() => {
      expect(result.current.segmentationRois).not.toBeUndefined();
    });
    expect(result.current.processingJobs).toHaveLength(0);
  });
});
