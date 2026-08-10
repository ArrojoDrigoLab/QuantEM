import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  makeSegmentation,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useReviewGroupSelection } from "@/features/segmentation/screen/hooks/useReviewGroupSelection";
import {
  querySegmentsInRegion,
  updateSegmentLabelsBatch,
} from "@/shared/api/segmentations/annotations";

describe("useReviewGroupSelection", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
  });

  it("queries bbox candidates and applies a grouped confirm action", async () => {
    vi.mocked(querySegmentsInRegion).mockResolvedValue({
      segments: [
        {
          id: "candidate-segment-2",
          label_state: "CANDIDATE",
          confidence_score: 0.73,
          geometry_coords: [
            [15, 15],
            [35, 15],
            [35, 35],
            [15, 35],
            [15, 15],
          ],
        },
      ],
    });

    const { result } = renderHook(() =>
      useReviewGroupSelection({
        currentSegmentation: makeSegmentation(),
        activeSourceModel: null,
        isErSegmentation: false,
        workflowMode: "review",
        correctionMode: {
          reviewPhase: "model",
          correctionTool: "draw",
        },
        leftMode: "hover",
        hoverActionMode: "group-confirm",
        registerAnnotationActivity: vi.fn(),
        applyOptimisticLabel: vi.fn(),
        rollbackOptimisticLabel: vi.fn(),
        clearHoverInteraction: vi.fn(),
        stageOptimisticRevisionTargets: vi.fn(),
        getOptimisticTargetRevision: () => null,
        handleOverlayMutationRefresh: vi.fn(),
        showErrorToast: vi.fn(),
      })
    );

    act(() => {
      result.current.handleGroupImagePress(
        { x: 10, y: 10 },
        { x: 10, y: 10 }
      );
      result.current.handleGroupImageDrag(
        { x: 40, y: 40 },
        { x: 60, y: 60 }
      );
      result.current.handleGroupImageRelease(
        { x: 40, y: 40 },
        { x: 60, y: 60 }
      );
    });

    await waitFor(() => {
      expect(result.current.groupSelectionBBox).toEqual({
        x0: 10,
        y0: 10,
        x1: 40,
        y1: 40,
      });
    });

    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 150));
    });

    await waitFor(() => {
      expect(querySegmentsInRegion).toHaveBeenCalledWith("seg-1", {
        bbox: { x0: 10, y0: 10, x1: 40, y1: 40 },
        states: ["CANDIDATE", "INFERRED"],
        source_model: null,
        include_geometry: true,
      });
    });

    await waitFor(() => {
      expect(result.current.groupBboxHighlightedSegmentIds).toEqual([
        "candidate-segment-2",
      ]);
    });

    await act(async () => {
      result.current.handleToolbarGroupAction("group-confirm");
    });

    await waitFor(() => {
      expect(updateSegmentLabelsBatch).toHaveBeenCalledWith({
        labels: [{ id: "candidate-segment-2", label_state: "CONFIRMED" }],
        source_model: null,
      });
    });
  });
});
