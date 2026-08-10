import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  hoverSelectionState,
  makeSegmentation,
  makeSegment,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useSegmentationHoverQuery } from "@/features/segmentation/screen/hooks/useSegmentationHoverQuery";
import { HOVER_QUERY_DEBOUNCE_MS } from "@/features/segmentation/screen/utils/constants";
import { getSegmentsAtPoint } from "@/shared/api/segmentations/annotations";

describe("useSegmentationHoverQuery", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
  });

  it("debounces point queries and forwards the resolved segments to hover selection", async () => {
    vi.mocked(getSegmentsAtPoint).mockResolvedValue([makeSegment()]);

    const { result } = renderHook(() =>
      useSegmentationHoverQuery({
        currentSegmentation: makeSegmentation(),
        activeSourceModel: null,
      })
    );

    vi.useFakeTimers();
    try {
      act(() => {
        result.current.scheduleHoverSegmentQuery(
          { x: 18, y: 22 },
          ["CANDIDATE"],
          (segments) => segments,
          "Failed hover query"
        );
        result.current.scheduleHoverSegmentQuery(
          { x: 24, y: 28 },
          ["CANDIDATE"],
          (segments) => segments,
          "Failed hover query"
        );
      });

      expect(getSegmentsAtPoint).not.toHaveBeenCalled();

      await act(async () => {
        vi.advanceTimersByTime(HOVER_QUERY_DEBOUNCE_MS);
        await Promise.resolve();
      });

      expect(getSegmentsAtPoint).toHaveBeenCalledWith("seg-1", {
        x: 24,
        y: 28,
        states: ["CANDIDATE"],
      }, expect.any(Object));
      expect(hoverSelectionState.findSegmentsAtPoint).toHaveBeenCalledWith(
        { x: 24, y: 28 },
        [expect.objectContaining({ id: "segment-1" })]
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it("clears hover state and cancels any scheduled query", () => {
    const { result } = renderHook(() =>
      useSegmentationHoverQuery({
        currentSegmentation: makeSegmentation(),
        activeSourceModel: null,
      })
    );

    vi.useFakeTimers();
    try {
      act(() => {
        result.current.scheduleHoverSegmentQuery(
          { x: 18, y: 22 },
          ["CANDIDATE"],
          (segments) => segments,
          "Failed hover query"
        );
        result.current.clearHoverInteraction();
        vi.runAllTimers();
      });

      expect(getSegmentsAtPoint).not.toHaveBeenCalled();
      expect(hoverSelectionState.clearHover).toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("aborts an in-flight hover query before starting the next one", async () => {
    let firstSignal: AbortSignal | undefined;
    const firstRequest = new Promise<never>(() => undefined);
    vi.mocked(getSegmentsAtPoint)
      .mockImplementationOnce((_segmentationId, _params, options) => {
        firstSignal = options?.signal;
        return firstRequest;
      })
      .mockResolvedValueOnce([makeSegment({ id: "segment-2" })]);

    const { result } = renderHook(() =>
      useSegmentationHoverQuery({
        currentSegmentation: makeSegmentation(),
        activeSourceModel: null,
      })
    );

    vi.useFakeTimers();
    try {
      act(() => {
        result.current.scheduleHoverSegmentQuery(
          { x: 18, y: 22 },
          ["CANDIDATE"],
          (segments) => segments,
          "Failed hover query"
        );
        vi.advanceTimersByTime(HOVER_QUERY_DEBOUNCE_MS);
      });

      await act(async () => {
        await Promise.resolve();
      });

      expect(firstSignal?.aborted).toBe(false);

      act(() => {
        result.current.scheduleHoverSegmentQuery(
          { x: 24, y: 28 },
          ["CANDIDATE"],
          (segments) => segments,
          "Failed hover query"
        );
        vi.advanceTimersByTime(HOVER_QUERY_DEBOUNCE_MS);
      });

      await act(async () => {
        await Promise.resolve();
      });

      expect(firstSignal?.aborted).toBe(true);
      expect(getSegmentsAtPoint).toHaveBeenNthCalledWith(
        2,
        "seg-1",
        {
          x: 24,
          y: 28,
          states: ["CANDIDATE"],
        },
        expect.any(Object)
      );
    } finally {
      vi.useRealTimers();
    }
  });
});
