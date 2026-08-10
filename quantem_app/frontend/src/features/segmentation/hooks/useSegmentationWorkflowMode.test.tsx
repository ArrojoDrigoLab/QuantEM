import { renderHook, act } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useSegmentationWorkflowMode } from "@/features/segmentation/hooks/useSegmentationWorkflowMode";

describe("useSegmentationWorkflowMode", () => {
  it("defaults to review and hover", () => {
    const { result } = renderHook(() => useSegmentationWorkflowMode());

    expect(result.current.workflowMode).toBe("review");
    expect(result.current.leftMode).toBe("hover");
  });

  it("switches left mode to annotate when annotate mode is selected", () => {
    const { result } = renderHook(() => useSegmentationWorkflowMode());

    act(() => {
      result.current.setWorkflowMode("annotate");
    });

    expect(result.current.workflowMode).toBe("annotate");
    expect(result.current.leftMode).toBe("annotate");
  });
});
