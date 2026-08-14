import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useFollowModelRunSelection } from "@/features/segmentation/screen/hooks/useFollowModelRunSelection";
import type { SegmentationModelRunSelection } from "@/features/segmentation/screen/hooks/useSegmentationProcessingState";

function run(
  status: SegmentationModelRunSelection["status"],
  overrides: Partial<SegmentationModelRunSelection> = {}
): SegmentationModelRunSelection {
  return {
    jobId: "job-1",
    status,
    sourceModel: "quantem:mito",
    adapterId: null,
    ...overrides,
  };
}

describe("useFollowModelRunSelection", () => {
  it("selects a queued model and selects it again when that run succeeds", () => {
    const onBaseModelSelected = vi.fn();
    const onAdaptedModelSelected = vi.fn();
    const { rerender } = renderHook(
      ({ currentRun }) =>
        useFollowModelRunSelection({
          segmentationId: "seg-1",
          run: currentRun,
          onBaseModelSelected,
          onAdaptedModelSelected,
        }),
      { initialProps: { currentRun: run("PENDING") } }
    );

    expect(onBaseModelSelected).toHaveBeenCalledTimes(1);
    expect(onBaseModelSelected).toHaveBeenLastCalledWith("quantem:mito");

    act(() => rerender({ currentRun: run("RUNNING") }));
    expect(onBaseModelSelected).toHaveBeenCalledTimes(1);

    act(() => rerender({ currentRun: run("SUCCESS") }));
    expect(onBaseModelSelected).toHaveBeenCalledTimes(2);
    expect(onBaseModelSelected).toHaveBeenLastCalledWith("quantem:mito");

    act(() => rerender({ currentRun: run("SUCCESS") }));
    expect(onBaseModelSelected).toHaveBeenCalledTimes(2);
  });

  it("restores the exact fine-tuned model rather than only its base", () => {
    const onBaseModelSelected = vi.fn();
    const onAdaptedModelSelected = vi.fn();
    renderHook(() =>
      useFollowModelRunSelection({
        segmentationId: "seg-1",
        run: run("RUNNING", {
          sourceModel: "omniem:mito",
          adapterId: "adapter-1",
        }),
        onBaseModelSelected,
        onAdaptedModelSelected,
      })
    );

    expect(onAdaptedModelSelected).toHaveBeenCalledWith(
      "adapter-1",
      "omniem:mito"
    );
    expect(onBaseModelSelected).not.toHaveBeenCalled();
  });

  it("does not override an explicit selection for an old completed run", () => {
    const onBaseModelSelected = vi.fn();
    renderHook(() =>
      useFollowModelRunSelection({
        segmentationId: "seg-1",
        run: run("SUCCESS"),
        onBaseModelSelected,
        onAdaptedModelSelected: vi.fn(),
      })
    );

    expect(onBaseModelSelected).not.toHaveBeenCalled();
  });
});
