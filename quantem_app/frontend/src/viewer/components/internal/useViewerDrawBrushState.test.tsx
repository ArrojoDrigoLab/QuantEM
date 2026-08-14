import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useViewerDrawBrushState } from "@/viewer/components/internal/useViewerDrawBrushState";

describe("useViewerDrawBrushState", () => {
  it("marks the live brush preview width as image-space", () => {
    const { result } = renderHook(() =>
      useViewerDrawBrushState({
        drawMode: false,
        brushMode: true,
        brushSize: 24,
        brushColor: "#33cc66",
      })
    );

    act(() => {
      result.current.startBrushStroke({ x: 10, y: 10 });
    });

    expect(result.current.brushPreviewOverlay).toMatchObject({
      strokeWidth: 24,
      strokeWidthUnits: "image",
    });
  });
});
