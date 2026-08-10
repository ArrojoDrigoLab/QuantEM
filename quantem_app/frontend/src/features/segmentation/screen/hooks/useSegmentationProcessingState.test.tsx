import { act, renderHook } from "@testing-library/react";
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

function RouterWrapper({ children }: PropsWithChildren) {
  return <MemoryRouter>{children}</MemoryRouter>;
}

describe("useSegmentationProcessingState", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
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
});
