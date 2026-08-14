import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  makeSegmentation,
  makeSegment,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useReviewPointActions } from "@/features/segmentation/screen/hooks/useReviewPointActions";
import {
  deleteSegmentsBatch,
  updateSegmentLabelsBatch,
} from "@/shared/api/segmentations/annotations";

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
        hideOptimisticallyDeletedSegment: vi.fn(() => true),
        rollbackOptimisticallyDeletedSegment: vi.fn(),
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

  it("hides a confirmed object before hard-deleting it without changing its label", async () => {
    let resolveDelete!: (
      value: Awaited<ReturnType<typeof deleteSegmentsBatch>>
    ) => void;
    vi.mocked(deleteSegmentsBatch).mockImplementation(
      () => new Promise((resolve) => {
        resolveDelete = resolve;
      })
    );
    const hideSegment = vi.fn(() => true);
    const rollbackSegment = vi.fn();
    const applyOptimisticLabel = vi.fn();
    const handleOverlayMutationRefresh = vi.fn();

    const { result } = renderHook(() =>
      useReviewPointActions({
        currentSegmentation: makeSegmentation(),
        activeSourceModel: "quantem:mito",
        hoverPoint: null,
        hoverSegments: [],
        highlightedSegmentId: null,
        applyLabelOverrides: (segments) => segments,
        applyOptimisticLabel,
        rollbackOptimisticLabel: vi.fn(),
        hideOptimisticallyDeletedSegment: hideSegment,
        rollbackOptimisticallyDeletedSegment: rollbackSegment,
        clearHoverInteraction: vi.fn(),
        registerAnnotationActivity: vi.fn(),
        stageOptimisticRevisionTargets: vi.fn(),
        getOptimisticTargetRevision: () => null,
        handleOverlayMutationRefresh,
        showErrorToast: vi.fn(),
      })
    );

    let request: Promise<void> | undefined;
    act(() => {
      request = result.current.handleDeleteConfirmedObject("confirmed-1");
    });

    expect(hideSegment).toHaveBeenCalledWith("confirmed-1");
    expect(deleteSegmentsBatch).toHaveBeenCalledWith("seg-1", {
      ids: ["confirmed-1"],
      source_model: "quantem:mito",
    });
    expect(applyOptimisticLabel).not.toHaveBeenCalled();

    resolveDelete({ deleted: 1, overlay: null });
    await act(async () => {
      await request;
    });
    expect(rollbackSegment).not.toHaveBeenCalled();
    expect(handleOverlayMutationRefresh).toHaveBeenCalledWith(null);
  });
});
