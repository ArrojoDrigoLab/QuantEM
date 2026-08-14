import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  setupSegmentationScreenTest,
  workflowModeState,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useReviewModeState } from "@/features/segmentation/screen/hooks/review/useReviewModeState";

describe("useReviewModeState", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
    workflowModeState.setLeftMode.mockClear();
  });

  it("falls back from unsupported group hover actions", async () => {
    const setHoverActionMode = vi.fn();
    renderHook(() =>
      useReviewModeState({
        currentSegmentationId: "seg-1",
        isErSegmentation: false,
        supportsPointFeedback: false,
        hoverActionMode: "group-confirm",
        setHoverActionMode,
      })
    );

    await waitFor(() => {
      expect(setHoverActionMode).toHaveBeenCalledWith("confirm");
    });
  });

  it("drives the left mode from the correction tool and resets on segmentation changes", async () => {
    const { result, rerender } = renderHook(
      (currentSegmentationId: string) =>
        useReviewModeState({
          currentSegmentationId,
          isErSegmentation: true,
          supportsPointFeedback: true,
          hoverActionMode: "confirm",
          setHoverActionMode: vi.fn(),
        }),
      { initialProps: "seg-1" }
    );

    act(() => {
      result.current.setCorrectionMode({
        reviewPhase: "correction",
        correctionTool: "completed_roi",
      });
    });

    await waitFor(() => {
      expect(result.current.isCorrectionReview).toBe(true);
      expect(workflowModeState.setLeftMode).toHaveBeenCalledWith("completed_roi");
    });

    rerender("seg-2");

    await waitFor(() => {
      expect(result.current.correctionMode.reviewPhase).toBe("correction");
      expect(result.current.correctionMode.correctionTool).toBe("draw");
    });
  });
});
