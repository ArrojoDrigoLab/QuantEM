import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  makeSegmentation,
  makeSegment,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useReviewPointActions } from "@/features/segmentation/screen/hooks/useReviewPointActions";
import { updateSegmentLabelsBatch } from "@/shared/api/segmentations/annotations";

describe("useReviewPointActions", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
  });

  it("uses the current hover target when confirming a candidate", async () => {
    const clearHoverInteraction = vi.fn();
    const registerAnnotationActivity = vi.fn();
    const applyOptimisticLabel = vi.fn();

    const { result } = renderHook(() =>
      useReviewPointActions({
        currentSegmentation: makeSegmentation(),
        activeSourceModel: null,
        hoverPoint: { x: 20, y: 20 },
        hoverSegments: [
          makeSegment({
            id: "candidate-segment-1",
            label_state: "CANDIDATE",
          }),
        ],
        highlightedSegmentId: "candidate-segment-1",
        applyLabelOverrides: (segments) => segments,
        applyOptimisticLabel,
        rollbackOptimisticLabel: vi.fn(),
        clearHoverInteraction,
        registerAnnotationActivity,
        stageOptimisticRevisionTargets: vi.fn(),
        getOptimisticTargetRevision: () => null,
        handleOverlayMutationRefresh: vi.fn(),
        showErrorToast: vi.fn(),
      })
    );

    await act(async () => {
      await result.current.handleApplyPointAction({ x: 20, y: 20 }, "confirm");
    });

    expect(registerAnnotationActivity).toHaveBeenCalled();
    expect(applyOptimisticLabel).toHaveBeenCalledWith(
      "candidate-segment-1",
      "CONFIRMED",
      expect.objectContaining({ id: "candidate-segment-1" }),
      { stageOverlay: true }
    );
    expect(clearHoverInteraction).toHaveBeenCalled();
    await waitFor(() => {
      expect(updateSegmentLabelsBatch).toHaveBeenCalledWith({
        labels: [{ id: "candidate-segment-1", label_state: "CONFIRMED" }],
        source_model: null,
      });
    });
  });
});
