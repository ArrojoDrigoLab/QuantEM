import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useSegmentationScreenUiState } from "@/features/segmentation/screen/hooks/useSegmentationScreenUiState";
import { TOAST_AUTO_DISMISS_MS } from "@/shared/ui/toast";

describe("useSegmentationScreenUiState", () => {
  it("shows confirmed overlays and enables navigation when a result becomes locked", () => {
    const { result, rerender } = renderHook(
      ({ readOnly }) => useSegmentationScreenUiState({ currentSegmentationId: "seg-1", readOnly }),
      { initialProps: { readOnly: false } },
    );
    act(() => result.current.setShowConfirmedPanel(false));
    rerender({ readOnly: true });
    expect(result.current.leftNavigateMode).toBe(true);
    expect(result.current.showConfirmedPanel).toBe(true);
    act(() => result.current.setShowConfirmedPanel(false));
    expect(result.current.showConfirmedPanel).toBe(false);
  });

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
