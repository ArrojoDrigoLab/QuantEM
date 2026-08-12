import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useViewerCursorState } from "@/viewer/components/internal/useViewerCursorState";
import type { ViewMetrics } from "@/viewer/components/internal/viewerMath";

function cursorStateFor(metrics: ViewMetrics) {
  const hook = renderHook(() =>
    useViewerCursorState({ brushMode: true, brushSize: 10, metrics })
  );
  act(() => hook.result.current.setIsPointerInside(true));
  act(() =>
    hook.result.current.updateOverlayCursor({ x: 40, y: 50 }, { x: 100, y: 100 })
  );
  return hook.result.current.overlayCursorState;
}

describe("useViewerCursorState", () => {
  it("keeps the brush opening scaled to the image while the ring width stays fixed", () => {
    const zoomedOut = cursorStateFor({
      imageWidth: 1_000,
      imageHeight: 1_000,
      containerWidth: 100,
      containerHeight: 100,
      visibleWidth: 1_000,
      visibleHeight: 1_000,
      minX: 0,
      minY: 0,
    });
    const zoomedIn = cursorStateFor({
      imageWidth: 1_000,
      imageHeight: 1_000,
      containerWidth: 100,
      containerHeight: 100,
      visibleWidth: 100,
      visibleHeight: 100,
      minX: 0,
      minY: 0,
    });

    // 10 image pixels become 1 CSS pixel while zoomed out, and 10 CSS pixels
    // while zoomed in. Subtracting both 2 px borders recovers those diameters.
    expect(zoomedOut.outerSize - zoomedOut.borderWidth * 2).toBe(1);
    expect(zoomedIn.outerSize - zoomedIn.borderWidth * 2).toBe(10);
    expect(zoomedOut.borderWidth).toBe(2);
    expect(zoomedIn.borderWidth).toBe(2);
  });
});
