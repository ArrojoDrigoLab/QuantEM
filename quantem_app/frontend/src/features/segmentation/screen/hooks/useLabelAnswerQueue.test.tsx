import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { setupSegmentationScreenTest } from "@/features/segmentation/SegmentationScreen.testUtils";
import {
  LABEL_ANSWER_COALESCE_WINDOW_MS,
  useLabelAnswerQueue,
} from "@/features/segmentation/screen/hooks/useLabelAnswerQueue";
import { updateSegmentLabelsBatch } from "@/shared/api/segmentations/annotations";
import type { SegmentationOverlayMutationState } from "@/shared/types/segmentation";

function makeArgs(overrides: Record<string, unknown> = {}) {
  return {
    segmentationId: "segmentation-1",
    activeSourceModel: null as string | null,
    rollbackOptimisticLabel: vi.fn(),
    stageOptimisticRevisionTargets: vi.fn(),
    getOptimisticTargetRevision: vi.fn(() => 7),
    handleOverlayMutationRefresh: vi.fn(),
    showErrorToast: vi.fn(),
    ...overrides,
  };
}

describe("useLabelAnswerQueue", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
  });

  it("sends a burst of answers as one request, in the order they were given", async () => {
    const args = makeArgs();
    const { result } = renderHook(() => useLabelAnswerQueue(args));

    act(() => {
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
      result.current.enqueueAnswer({
        segmentId: "b",
        labelState: "EXCLUDED",
        fallbackMessage: "Failed to reject the selected object.",
      });
      result.current.enqueueAnswer({
        segmentId: "c",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
    });

    await waitFor(() => {
      expect(updateSegmentLabelsBatch).toHaveBeenCalledTimes(1);
    });
    expect(updateSegmentLabelsBatch).toHaveBeenCalledWith({
      labels: [
        { id: "a", label_state: "CONFIRMED" },
        { id: "b", label_state: "EXCLUDED" },
        { id: "c", label_state: "CONFIRMED" },
      ],
      source_model: null,
    });
  });

  it("sends nothing before the window closes", async () => {
    const args = makeArgs();
    const { result } = renderHook(() => useLabelAnswerQueue(args));

    act(() => {
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
    });

    await new Promise((resolve) =>
      setTimeout(resolve, Math.floor(LABEL_ANSWER_COALESCE_WINDOW_MS / 2))
    );
    expect(updateSegmentLabelsBatch).not.toHaveBeenCalled();

    await waitFor(() => {
      expect(updateSegmentLabelsBatch).toHaveBeenCalledTimes(1);
    });
  });

  it("sends one answer per object, carrying the state the reviewer left it in", async () => {
    const args = makeArgs();
    const { result } = renderHook(() => useLabelAnswerQueue(args));

    act(() => {
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "CANDIDATE",
        fallbackMessage: "Failed to un-mark the selected object.",
      });
    });

    await waitFor(() => {
      expect(updateSegmentLabelsBatch).toHaveBeenCalledTimes(1);
    });
    expect(updateSegmentLabelsBatch).toHaveBeenCalledWith({
      labels: [{ id: "a", label_state: "CANDIDATE" }],
      source_model: null,
    });
  });

  it("stages the overlay revision for every answer in the batch", async () => {
    const stageOptimisticRevisionTargets = vi.fn();
    const handleOverlayMutationRefresh = vi.fn();
    // Annotated, not inferred: without this `rebuild_mode` widens to `string`
    // and no longer satisfies the union the API returns.
    const overlay: SegmentationOverlayMutationState = {
      desired_revision: 9,
      applied_revision: 8,
      sync_applied: false,
      rebuild_mode: "async_partial",
    };
    vi.mocked(updateSegmentLabelsBatch).mockResolvedValue({
      updated: 2,
      overlays: { "segmentation-1": overlay },
    });
    const args = makeArgs({
      stageOptimisticRevisionTargets,
      handleOverlayMutationRefresh,
      getOptimisticTargetRevision: vi.fn(() => 9),
    });
    const { result } = renderHook(() => useLabelAnswerQueue(args));

    act(() => {
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
      result.current.enqueueAnswer({
        segmentId: "b",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
    });

    await waitFor(() => {
      expect(stageOptimisticRevisionTargets).toHaveBeenCalledWith(["a", "b"], 9);
    });
    expect(handleOverlayMutationRefresh).toHaveBeenCalledWith(overlay);
  });

  it("puts every answer in a failed batch back, and says how many, once", async () => {
    const rollbackOptimisticLabel = vi.fn();
    const showErrorToast = vi.fn();
    vi.mocked(updateSegmentLabelsBatch).mockRejectedValue(
      new Error('{"error":"This image is finished, so it cannot be changed."}')
    );
    const args = makeArgs({ rollbackOptimisticLabel, showErrorToast });
    const { result } = renderHook(() => useLabelAnswerQueue(args));

    act(() => {
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
      result.current.enqueueAnswer({
        segmentId: "b",
        labelState: "EXCLUDED",
        fallbackMessage: "Failed to reject the selected object.",
      });
    });

    await waitFor(() => {
      expect(showErrorToast).toHaveBeenCalledTimes(1);
    });
    expect(rollbackOptimisticLabel).toHaveBeenCalledWith("a");
    expect(rollbackOptimisticLabel).toHaveBeenCalledWith("b");
    // The server's own reason survives, and the count the reviewer lost is
    // added to it rather than replacing it.
    expect(showErrorToast).toHaveBeenCalledWith(
      "This image is finished, so it cannot be changed. " +
        "All 2 have been put back the way they were."
    );
  });

  it("keeps the single-answer wording when only one answer failed", async () => {
    const showErrorToast = vi.fn();
    vi.mocked(updateSegmentLabelsBatch).mockRejectedValue(new Error(""));
    const args = makeArgs({ showErrorToast });
    const { result } = renderHook(() => useLabelAnswerQueue(args));

    act(() => {
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "EXCLUDED",
        fallbackMessage: "Failed to reject the selected object.",
      });
    });

    await waitFor(() => {
      expect(showErrorToast).toHaveBeenCalledWith(
        "Failed to reject the selected object."
      );
    });
  });

  it("flushes on demand without waiting for the window", async () => {
    const args = makeArgs();
    const { result } = renderHook(() => useLabelAnswerQueue(args));

    await act(async () => {
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
      await result.current.flushAnswers();
    });

    expect(updateSegmentLabelsBatch).toHaveBeenCalledTimes(1);
  });

  it("sends pending answers under the source model they were given in", async () => {
    const { result, rerender } = renderHook(
      (props: { activeSourceModel: string | null }) =>
        useLabelAnswerQueue(makeArgs({ activeSourceModel: props.activeSourceModel })),
      { initialProps: { activeSourceModel: "quantem_mito" as string | null } }
    );

    act(() => {
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
    });
    // Switching models before the window closes must not relabel the answer as
    // the other model's: the bundle it dirties depends on which one it was.
    rerender({ activeSourceModel: "quantem_nucleus" });

    await waitFor(() => {
      expect(updateSegmentLabelsBatch).toHaveBeenCalledTimes(1);
    });
    expect(updateSegmentLabelsBatch).toHaveBeenCalledWith({
      labels: [{ id: "a", label_state: "CONFIRMED" }],
      source_model: "quantem_mito",
    });
  });

  it("does not discard an answer when the screen goes away", async () => {
    const args = makeArgs();
    const { result, unmount } = renderHook(() => useLabelAnswerQueue(args));

    act(() => {
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
    });
    unmount();

    await waitFor(() => {
      expect(updateSegmentLabelsBatch).toHaveBeenCalledTimes(1);
    });
  });

  it("does not discard an answer when the window is hidden", async () => {
    const args = makeArgs();
    const { result } = renderHook(() => useLabelAnswerQueue(args));

    act(() => {
      result.current.enqueueAnswer({
        segmentId: "a",
        labelState: "CONFIRMED",
        fallbackMessage: "Failed to confirm the selected object.",
      });
    });

    act(() => {
      Object.defineProperty(document, "visibilityState", {
        configurable: true,
        get: () => "hidden",
      });
      document.dispatchEvent(new Event("visibilitychange"));
    });

    await waitFor(() => {
      expect(updateSegmentLabelsBatch).toHaveBeenCalledTimes(1);
    });
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: () => "visible",
    });
  });
});
