import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useSegmentationScreenUiState } from "@/features/segmentation/screen/hooks/useSegmentationScreenUiState";
import { TOAST_AUTO_DISMISS_MS } from "@/shared/ui/toast";

describe("useSegmentationScreenUiState", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("auto-dismisses error toasts after 4 seconds and refreshes the timer for repeated messages", () => {
    vi.useFakeTimers();

    const { result } = renderHook(() =>
      useSegmentationScreenUiState({ currentSegmentationId: "seg-1" })
    );

    act(() => {
      result.current.showErrorToast("Failed to confirm the prompt shape.");
    });
    expect(result.current.toast?.message).toBe("Failed to confirm the prompt shape.");

    act(() => {
      vi.advanceTimersByTime(TOAST_AUTO_DISMISS_MS - 1);
    });
    expect(result.current.toast?.message).toBe("Failed to confirm the prompt shape.");

    act(() => {
      result.current.showErrorToast("Failed to confirm the prompt shape.");
    });
    expect(result.current.toast?.message).toBe("Failed to confirm the prompt shape.");

    act(() => {
      vi.advanceTimersByTime(TOAST_AUTO_DISMISS_MS - 1);
    });
    expect(result.current.toast?.message).toBe("Failed to confirm the prompt shape.");

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(result.current.toast).toBeNull();
  });
});
