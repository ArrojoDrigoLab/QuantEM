import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  LABELING_ROI_SIZE,
  useErRoiWorkflow,
} from "@/features/segmentation/screen/hooks/useErRoiWorkflow";
import type { SegmentationRoi } from "@/shared/types/segmentation";

function makeRoi(): SegmentationRoi {
  return {
    id: "roi-1",
    segmentation: "seg-1",
    x: 100,
    y: 200,
    width: 512,
    height: 512,
    source: "MANUAL",
    is_active: true,
    is_complete: false,
    completed_for_segmentation: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

describe("useErRoiWorkflow", () => {
  it("places new ROIs at 1024 square pixels by default", () => {
    const { result } = renderHook(() =>
      useErRoiWorkflow({
        currentSegmentationId: "seg-1",
        enabled: true,
        image: { width: 2048, height: 1536 },
        isPointInsideImageBounds: () => true,
        refetchSegmentationRois: vi.fn(),
        showErrorToast: vi.fn(),
      })
    );

    act(() => {
      result.current.startPlacement();
    });

    expect(LABELING_ROI_SIZE).toBe(1024);
    expect(result.current.resolvePendingRoi({ x: 1024, y: 768 })).toEqual({
      roiId: null,
      x: 512,
      y: 256,
      width: 1024,
      height: 1024,
    });
  });

  it("clamps the default ROI to images smaller than 1024 pixels", () => {
    const { result } = renderHook(() =>
      useErRoiWorkflow({
        currentSegmentationId: "seg-1",
        enabled: true,
        image: { width: 800, height: 600 },
        isPointInsideImageBounds: () => true,
        refetchSegmentationRois: vi.fn(),
        showErrorToast: vi.fn(),
      })
    );

    expect(result.current.resolvePendingRoi({ x: 400, y: 300 })).toEqual({
      roiId: null,
      x: 0,
      y: 0,
      width: 800,
      height: 600,
    });
  });

  it("opens an existing ROI with eight resize handles and resizes from a corner", () => {
    const { result } = renderHook(() =>
      useErRoiWorkflow({
        currentSegmentationId: "seg-1",
        enabled: true,
        image: { width: 2048, height: 1536 },
        isPointInsideImageBounds: () => true,
        refetchSegmentationRois: vi.fn(),
        showErrorToast: vi.fn(),
      })
    );

    act(() => {
      result.current.editRoi(makeRoi());
    });

    expect(result.current.pendingRoi).toEqual({
      roiId: "roi-1",
      x: 100,
      y: 200,
      width: 512,
      height: 512,
    });
    expect(result.current.pendingRoiOverlays).toHaveLength(9);

    act(() => {
      result.current.handleEditPress({ x: 612, y: 712 });
      result.current.handleEditDrag({ x: 900, y: 1000 });
      result.current.handleEditRelease({ x: 900, y: 1000 });
    });

    expect(result.current.pendingRoi).toEqual({
      roiId: "roi-1",
      x: 100,
      y: 200,
      width: 800,
      height: 800,
    });
  });

  it("moves an edited ROI by dragging its interior and clamps it to the image", () => {
    const { result } = renderHook(() =>
      useErRoiWorkflow({
        currentSegmentationId: "seg-1",
        enabled: true,
        image: { width: 1200, height: 1000 },
        isPointInsideImageBounds: () => true,
        refetchSegmentationRois: vi.fn(),
        showErrorToast: vi.fn(),
      })
    );

    act(() => {
      result.current.editRoi(makeRoi());
    });
    act(() => {
      result.current.handleEditPress({ x: 356, y: 456 });
      result.current.handleEditDrag({ x: 1600, y: 1400 });
    });

    expect(result.current.pendingRoi).toMatchObject({
      x: 688,
      y: 488,
      width: 512,
      height: 512,
    });
  });
});
