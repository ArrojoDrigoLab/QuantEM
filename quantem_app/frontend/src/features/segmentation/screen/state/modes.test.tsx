import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useSegmentationModes } from "@/features/segmentation/screen/state/modes";

describe("useSegmentationModes", () => {
  it("arrives ready to label, not ready to pan", () => {
    // Navigate used to start on, so the first click on every image did
    // nothing and the only explanation was a passive note in the sidebar.
    const { result } = renderHook(() =>
      useSegmentationModes({ currentSegmentationId: "seg-1" })
    );

    expect(result.current.leftNavigateMode).toBe(false);
  });

  it("does not re-arm Navigate when the user switches organelle", () => {
    const { result, rerender } = renderHook(
      ({ id }: { id: string }) => useSegmentationModes({ currentSegmentationId: id }),
      { initialProps: { id: "seg-mito" } }
    );

    act(() => {
      result.current.setLeftNavigateMode(true);
    });
    expect(result.current.leftNavigateMode).toBe(true);

    rerender({ id: "seg-nucleus" });
    expect(result.current.leftNavigateMode).toBe(true);

    act(() => {
      result.current.exitNavigateMode();
    });
    rerender({ id: "seg-mito" });
    expect(result.current.leftNavigateMode).toBe(false);
  });

  it("still lets the user turn Navigate on and off deliberately", () => {
    const { result } = renderHook(() =>
      useSegmentationModes({ currentSegmentationId: "seg-1" })
    );

    act(() => {
      result.current.toggleLeftNavigateMode();
    });
    expect(result.current.leftNavigateMode).toBe(true);

    act(() => {
      result.current.toggleLeftNavigateMode();
    });
    expect(result.current.leftNavigateMode).toBe(false);
  });
});
