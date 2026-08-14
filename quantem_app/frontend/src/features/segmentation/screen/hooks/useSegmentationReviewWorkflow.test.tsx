import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  drawingState,
  makeSegmentation,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useSegmentationReviewWorkflow } from "@/features/segmentation/screen/hooks/useSegmentationReviewWorkflow";
import type { useDrawing } from "@/hooks/useDrawing";

function renderWorkflow(exitNavigateMode: () => void) {
  return renderHook(() =>
    useSegmentationReviewWorkflow({
      currentSegmentation: makeSegmentation(),
      activeSourceModel: "quantem:mito",
      isErSegmentation: false,
      supportsPointFeedback: true,
      hoverSegments: [],
      highlightedSegmentId: null,
      hoverPoint: null,
      hoverActionMode: "confirm",
      setHoverActionMode: vi.fn(),
      clearHoverInteraction: vi.fn(),
      applyLabelOverrides: (items) => items,
      applyOptimisticLabel: vi.fn(),
      rollbackOptimisticLabel: vi.fn(),
      hideOptimisticallyDeletedSegment: vi.fn(() => true),
      rollbackOptimisticallyDeletedSegment: vi.fn(),
      stageOptimisticRevisionTargets: vi.fn(),
      getOptimisticTargetRevision: () => null,
      handleOverlayMutationRefresh: vi.fn(),
      registerAnnotationActivity: vi.fn(),
      showErrorToast: vi.fn(),
      showNoticeToast: vi.fn(),
      exitNavigateMode,
      drawing: drawingState as unknown as ReturnType<typeof useDrawing>,
      submitConfirmedGeometriesOptimistically: vi.fn(),
    })
  );
}

describe("useSegmentationReviewWorkflow", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
  });

  it("leaves Navigate mode when a drawing tool is chosen", () => {
    // Navigate is on by default and the interaction router drops every labeling
    // click while it is, so without this the first stroke after picking a tool
    // silently did nothing.
    const exitNavigateMode = vi.fn();
    const { result } = renderWorkflow(exitNavigateMode);

    act(() => {
      result.current.mode.handleCorrectionToolChange("draw");
    });

    expect(exitNavigateMode).toHaveBeenCalledTimes(1);
    expect(result.current.mode.correctionMode.correctionTool).toBe("draw");
  });

  it("leaves Navigate mode for the confirmed-area tool too", () => {
    const exitNavigateMode = vi.fn();
    const { result } = renderWorkflow(exitNavigateMode);

    act(() => {
      result.current.mode.handleCorrectionToolChange("completed_roi");
    });

    expect(exitNavigateMode).toHaveBeenCalledTimes(1);
  });

  it("leaves Navigate mode when entering the correction phase", () => {
    const exitNavigateMode = vi.fn();
    const { result } = renderWorkflow(exitNavigateMode);

    act(() => {
      result.current.mode.handleReviewPhaseChange("correction");
    });

    expect(exitNavigateMode).toHaveBeenCalledTimes(1);
    expect(result.current.mode.correctionMode.reviewPhase).toBe("correction");
  });

  it("leaves Navigate mode when a group action mode is chosen", () => {
    // Switching to Confirm Group left Navigate on, so the first box-drag panned
    // the image while the panel said "Drag a box in labeling view" -- the
    // interaction router drops every press/drag/release while Navigate is on.
    const exitNavigateMode = vi.fn();
    const { result } = renderWorkflow(exitNavigateMode);

    act(() => {
      result.current.mode.handleHoverActionModeChange("group-confirm");
    });

    expect(exitNavigateMode).toHaveBeenCalledTimes(1);
  });

  it("leaves Navigate mode for the point action modes too", () => {
    // onLeftClick is dropped while Navigate is on, so Confirm Object was just
    // as dead as Confirm Group.
    const exitNavigateMode = vi.fn();
    const { result } = renderWorkflow(exitNavigateMode);

    act(() => {
      result.current.mode.handleHoverActionModeChange("reject");
    });

    expect(exitNavigateMode).toHaveBeenCalledTimes(1);
  });

  it("keeps Navigate mode when returning to the review phase", () => {
    // Going back to Review is not a request to label, and panning is the useful
    // default there.
    const exitNavigateMode = vi.fn();
    const { result } = renderWorkflow(exitNavigateMode);

    act(() => {
      result.current.mode.handleReviewPhaseChange("model");
    });

    expect(exitNavigateMode).not.toHaveBeenCalled();
    expect(drawingState.clearDrawing).toHaveBeenCalled();
  });
});
