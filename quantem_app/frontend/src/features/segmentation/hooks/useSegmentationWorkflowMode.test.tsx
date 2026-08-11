import { renderHook, act } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useSegmentationWorkflowMode } from "@/features/segmentation/hooks/useSegmentationWorkflowMode";

describe("useSegmentationWorkflowMode", () => {
  it("defaults to review and hover", () => {
    const { result } = renderHook(() => useSegmentationWorkflowMode());

    expect(result.current.workflowMode).toBe("review");
    expect(result.current.leftMode).toBe("hover");
  });

  it("returns the canvas to hover whenever the workflow mode changes", () => {
    const { result } = renderHook(() => useSegmentationWorkflowMode());

    act(() => {
      result.current.setLeftMode("draw");
    });
    expect(result.current.leftMode).toBe("draw");

    act(() => {
      result.current.setWorkflowMode("uncertain");
    });

    expect(result.current.workflowMode).toBe("uncertain");
    expect(result.current.leftMode).toBe("hover");
  });

  it("has no annotate mode to enter", () => {
    // `annotate` was a third workflow whose handlers were already empty, so
    // choosing it armed a canvas that could not do anything. It is gone from
    // the type, which is what stops it being reintroduced by a string literal.
    const modes: Array<ReturnType<typeof useSegmentationWorkflowMode>["workflowMode"]> = [
      "review",
      "uncertain",
    ];
    expect(modes).not.toContain("annotate");
  });
});
