import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  createCompletedRoi,
  getCompletedRois,
} from "@/shared/api/segmentations/completedRois";
import { useCompletedRoiWorkflow } from "@/features/segmentation/screen/hooks/useCompletedRoiWorkflow";
import { makeSegmentation } from "@/features/segmentation/SegmentationScreen.testUtils";

vi.mock("@/shared/api/segmentations/completedRois", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/completedRois")
  >("@/shared/api/segmentations/completedRois");
  return {
    ...actual,
    getCompletedRois: vi.fn(),
    createCompletedRoi: vi.fn(),
  };
});

describe("useCompletedRoiWorkflow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not fetch completed ROIs while inactive", async () => {
    renderHook(() =>
      useCompletedRoiWorkflow({
        currentSegmentation: makeSegmentation(),
        active: false,
        isPointInsideImageBounds: () => true,
        registerAnnotationActivity: vi.fn(),
        showErrorToast: vi.fn(),
      })
    );

    await waitFor(() => {
      expect(getCompletedRois).not.toHaveBeenCalled();
    });
  });

  it("fetches only when active and saves a closed polygon", async () => {
    vi.mocked(getCompletedRois)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        {
          id: "completed-roi-1",
          segmentation: "seg-1",
          polygon_coords: [
            [10, 10],
            [30, 10],
            [10, 20],
            [10, 10],
          ],
          bbox: { x0: 10, y0: 10, x1: 30, y1: 20 },
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        },
      ]);
    vi.mocked(createCompletedRoi).mockResolvedValue({
      id: "completed-roi-1",
      segmentation: "seg-1",
      polygon_coords: [
        [10, 10],
        [30, 10],
        [10, 20],
        [10, 10],
      ],
      bbox: { x0: 10, y0: 10, x1: 30, y1: 20 },
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    });

    const registerAnnotationActivity = vi.fn();
    const { result } = renderHook(() =>
      useCompletedRoiWorkflow({
        currentSegmentation: makeSegmentation(),
        active: true,
        isPointInsideImageBounds: () => true,
        registerAnnotationActivity,
        showErrorToast: vi.fn(),
      })
    );

    await waitFor(() => {
      expect(getCompletedRois).toHaveBeenCalledWith("seg-1");
    });

    act(() => {
      result.current.handlePolygonClick({ x: 10, y: 10 });
    });
    act(() => {
      result.current.handlePolygonMouseMove({ x: 20, y: 20 });
    });
    act(() => {
      result.current.handlePolygonClick({ x: 30, y: 10 });
    });

    await act(async () => {
      await result.current.handleClosePolygon();
    });

    await waitFor(() => {
      expect(result.current.canSave).toBe(true);
    });

    act(() => {
      result.current.requestSave();
    });
    expect(result.current.saveDialogOpen).toBe(true);

    await act(async () => {
      await result.current.confirmSave();
    });

    expect(createCompletedRoi).toHaveBeenCalledTimes(1);
    const [, payload] = vi.mocked(createCompletedRoi).mock.calls[0];
    expect(payload.polygon_coords.length).toBeGreaterThanOrEqual(4);
    expect(payload.polygon_coords[0]).toEqual(payload.polygon_coords[payload.polygon_coords.length - 1]);
    expect(registerAnnotationActivity).toHaveBeenCalled();

    await waitFor(() => {
      expect(result.current.items).toHaveLength(1);
    });
    expect(result.current.hasDraft).toBe(false);
    expect(result.current.saveDialogOpen).toBe(false);
  });
});
