import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/shared/api/segmentations/rois", () => ({
  activateSegmentationRoi: vi.fn(),
  createSegmentationRoi: vi.fn(),
  deleteSegmentationRoi: vi.fn(),
  setRoiCompleteForSegmentation: vi.fn(),
}));

import {
  LABELING_ROI_SIZE,
  useErRoiWorkflow,
} from "@/features/segmentation/screen/hooks/useErRoiWorkflow";
import { setRoiCompleteForSegmentation } from "@/shared/api/segmentations/rois";
import type {
  RoiCompletionResponse,
  SegmentationRoi,
} from "@/shared/types/segmentation";

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
  beforeEach(() => {
    vi.clearAllMocks();
  });

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

  it("marks Done optimistically while candidate and overlay refreshes finish", async () => {
    let resolveRequest: ((value: RoiCompletionResponse) => void) | undefined;
    vi.mocked(setRoiCompleteForSegmentation).mockReturnValue(
      new Promise<RoiCompletionResponse>((resolve) => {
        resolveRequest = resolve;
      })
    );
    const refetchSegmentationRois = vi.fn(async () => undefined);
    const refreshSegmentViews = vi.fn(async () => undefined);
    const { result } = renderHook(() =>
      useErRoiWorkflow({
        currentSegmentationId: "seg-1",
        enabled: true,
        image: { width: 2048, height: 1536 },
        isPointInsideImageBounds: () => true,
        refetchSegmentationRois,
        refreshSegmentViews,
        showErrorToast: vi.fn(),
      })
    );

    let request: Promise<void> | undefined;
    act(() => {
      request = result.current.markRoiDone("roi-1", true);
    });

    expect(result.current.markingRoiId).toBe("roi-1");
    expect(result.current.markingRoiDone).toBe(true);

    await act(async () => {
      resolveRequest?.({
        ...makeRoi(),
        completed_for_segmentation: true,
        candidate_cleanup: { deleted: 2, updated: 0, created: 0 },
        overlay: {
          desired_revision: 2,
          applied_revision: 1,
          sync_applied: false,
          rebuild_mode: "async_partial",
        },
      });
      await request;
    });

    expect(refetchSegmentationRois).toHaveBeenCalledOnce();
    expect(refreshSegmentViews).toHaveBeenCalledWith({
      deferOverlayRefresh: true,
    });
    expect(result.current.markingRoiId).toBeNull();
    expect(result.current.markingRoiDone).toBeNull();
  });
});
