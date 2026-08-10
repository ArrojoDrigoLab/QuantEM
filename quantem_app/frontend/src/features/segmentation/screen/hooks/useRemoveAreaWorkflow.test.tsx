import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  drawingState,
  makeSegmentation,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useRemoveAreaWorkflow } from "@/features/segmentation/screen/hooks/useRemoveAreaWorkflow";

describe("useRemoveAreaWorkflow", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
    drawingState.clearDrawing.mockClear();
  });

  it("does not rerun the reset effect on a benign rerender", async () => {
    const args = {
      currentSegmentation: makeSegmentation(),
      currentSegmentationId: "seg-1",
      registerAnnotationActivity: vi.fn(),
      handleOverlayMutationRefresh: vi.fn(),
      clearHoverInteraction: vi.fn(),
      showErrorToast: vi.fn(),
      showNoticeToast: vi.fn(),
    };

    const { rerender } = renderHook((hookArgs) => useRemoveAreaWorkflow(hookArgs), {
      initialProps: args,
    });

    await waitFor(() => {
      expect(drawingState.clearDrawing).toHaveBeenCalledTimes(1);
    });

    rerender({ ...args });

    await waitFor(() => {
      expect(drawingState.clearDrawing).toHaveBeenCalledTimes(1);
    });
  });
});
