import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSegmentationInteractionRouter } from "@/features/segmentation/screen/hooks/useSegmentationInteractionRouter";

function makeArgs(
  overrides: Partial<Parameters<typeof useSegmentationInteractionRouter>[0]> = {}
): Parameters<typeof useSegmentationInteractionRouter>[0] {
  return {
    currentSegmentationId: "seg-1",
    leftNavigateMode: false,
    roiPlacementActive: false,
    isPointInsideImageBounds: () => true,
    applyLabelOverrides: (items) => items,
    scheduleHoverSegmentQuery: vi.fn(),
    clearHoverInteraction: vi.fn(),
    onRoiPlacementClick: vi.fn(),
    completedRoi: {
      isActive: false,
      handlePolygonClick: vi.fn(),
      handlePolygonMouseMove: vi.fn(),
    },
    erPolygon: {
      isActive: false,
      handlePolygonClick: vi.fn(),
      handlePolygonMouseMove: vi.fn(),
    },
    tissue: {
      enabled: false,
      polygon: {
        isActive: false,
        handlePolygonClick: vi.fn(),
        handlePolygonMouseMove: vi.fn(),
      },
    },
    review: {
      hoverActionMode: "confirm",
      leftMode: "hover",
      workflowMode: "review",
      isGroupActionMode: false,
      group: {
        handleImagePress: vi.fn(),
        handleImageDrag: vi.fn(),
        handleImageRelease: vi.fn(),
      },
      pointActions: {
        handleApply: vi.fn(async () => {}),
      },
    },
    ...overrides,
  };
}

describe("useSegmentationInteractionRouter", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("routes review hover movement through the shared hover query", () => {
    const args = makeArgs();
    const { result } = renderHook(() => useSegmentationInteractionRouter(args));

    result.current.onLeftMouseMove({ x: 18, y: 22 });

    expect(args.scheduleHoverSegmentQuery).toHaveBeenCalledWith(
      { x: 18, y: 22 },
      ["CONFIRMED", "CANDIDATE", "INFERRED"],
      expect.any(Function),
      "Failed to query hover segments:"
    );
  });

  it("routes an ROI placement click through the placement handler", () => {
    const args = makeArgs({ roiPlacementActive: true });
    const { result } = renderHook(() => useSegmentationInteractionRouter(args));

    result.current.onLeftClick({ x: 100, y: 120 });

    expect(args.onRoiPlacementClick).toHaveBeenCalledWith({ x: 100, y: 120 });
  });

  it("routes ROI placement clicks even while navigate mode is active", () => {
    const args = makeArgs({ roiPlacementActive: true, leftNavigateMode: true });
    const { result } = renderHook(() => useSegmentationInteractionRouter(args));

    result.current.onLeftClick({ x: 100, y: 120 });

    expect(args.onRoiPlacementClick).toHaveBeenCalledWith({ x: 100, y: 120 });
  });

  it("routes ER polygon clicks through the polygon handler", () => {
    const args = makeArgs({
      erPolygon: {
        isActive: true,
        handlePolygonClick: vi.fn(),
        handlePolygonMouseMove: vi.fn(),
      },
    });
    const { result } = renderHook(() => useSegmentationInteractionRouter(args));

    result.current.onLeftClick({ x: 33, y: 44 });

    expect(args.erPolygon.handlePolygonClick).toHaveBeenCalledWith({ x: 33, y: 44 });
  });

  it("routes tissue polygon clicks through the polygon handler", () => {
    const tissuePolygonClick = vi.fn();
    const pointActionApply = vi.fn(async () => {});
    const args = makeArgs({
      tissue: {
        enabled: true,
        polygon: {
          isActive: true,
          handlePolygonClick: tissuePolygonClick,
          handlePolygonMouseMove: vi.fn(),
        },
      },
      review: {
        ...makeArgs().review,
        hoverActionMode: "confirm",
        pointActions: { handleApply: pointActionApply },
      },
    });
    const { result } = renderHook(() => useSegmentationInteractionRouter(args));

    result.current.onLeftClick({ x: 33, y: 44 });

    expect(tissuePolygonClick).toHaveBeenCalledWith({ x: 33, y: 44 });
    // Tissue short-circuits the review confirm/reject point action.
    expect(pointActionApply).not.toHaveBeenCalled();
  });

  it("ignores tissue clicks when the brush (no polygon) tool is active", () => {
    const pointActionApply = vi.fn(async () => {});
    const args = makeArgs({
      tissue: {
        enabled: true,
        polygon: {
          isActive: false,
          handlePolygonClick: vi.fn(),
          handlePolygonMouseMove: vi.fn(),
        },
      },
      review: {
        ...makeArgs().review,
        hoverActionMode: "confirm",
        pointActions: { handleApply: pointActionApply },
      },
    });
    const { result } = renderHook(() => useSegmentationInteractionRouter(args));

    result.current.onLeftClick({ x: 5, y: 6 });

    expect(args.tissue.polygon.handlePolygonClick).not.toHaveBeenCalled();
    expect(pointActionApply).not.toHaveBeenCalled();
  });

  it("routes review point clicks through the point-action handler", () => {
    const args = makeArgs({
      review: {
        ...makeArgs().review,
        hoverActionMode: "reject",
      },
    });
    const { result } = renderHook(() => useSegmentationInteractionRouter(args));

    result.current.onLeftClick({ x: 12, y: 14 });

    expect(args.review.pointActions.handleApply).toHaveBeenCalledWith(
      { x: 12, y: 14 },
      "reject"
    );
  });

  it("routes completed ROI clicks before other left-panel interactions", () => {
    const args = makeArgs({
      completedRoi: {
        isActive: true,
        handlePolygonClick: vi.fn(),
        handlePolygonMouseMove: vi.fn(),
      },
    });
    const { result } = renderHook(() => useSegmentationInteractionRouter(args));

    result.current.onLeftClick({ x: 44, y: 55 });

    expect(args.completedRoi.handlePolygonClick).toHaveBeenCalledWith({ x: 44, y: 55 });
    expect(args.review.pointActions.handleApply).not.toHaveBeenCalled();
  });

  it("only starts bbox selection gestures in group-action mode", () => {
    const args = makeArgs();
    const { result } = renderHook(() => useSegmentationInteractionRouter(args));

    result.current.onLeftImagePress({ x: 10, y: 12 }, { x: 100, y: 120 });

    expect(args.review.group.handleImagePress).not.toHaveBeenCalled();

    const groupArgs = makeArgs({
      review: { ...makeArgs().review, isGroupActionMode: true },
    });
    const { result: groupResult } = renderHook(() =>
      useSegmentationInteractionRouter(groupArgs)
    );

    groupResult.current.onLeftImagePress({ x: 10, y: 12 }, { x: 100, y: 120 });

    expect(groupArgs.review.group.handleImagePress).toHaveBeenCalledWith(
      { x: 10, y: 12 },
      { x: 100, y: 120 }
    );
  });
});
