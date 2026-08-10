import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSegmentationKeyboardShortcuts } from "@/features/segmentation/screen/hooks/useSegmentationKeyboardShortcuts";
import {
  drawingState,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";

function makeArgs(
  overrides: Partial<Parameters<typeof useSegmentationKeyboardShortcuts>[0]> = {}
): Parameters<typeof useSegmentationKeyboardShortcuts>[0] {
  return {
    leftNavigateMode: false,
    toggleLeftNavigateMode: vi.fn(),
    cycleHoverIndex: vi.fn(),
    drawing: drawingState as never,
    removeArea: {
      mode: "none",
      clearDrawing: vi.fn(),
      canApply: false,
      handleApply: vi.fn(async () => {}),
    },
    completedRoi: {
      isActive: false,
      canClosePolygon: false,
      hasDraft: false,
      handleClosePolygon: vi.fn(async () => {}),
      clearDraft: vi.fn(),
    },
    erPolygon: {
      isActive: false,
      canClosePolygon: false,
      hasDraft: false,
      handleClosePolygon: vi.fn(async () => {}),
      clearDraft: vi.fn(),
    },
    tissue: {
      enabled: false,
      polygonCanClose: false,
      polygonHasDraft: false,
      canConfirmBrush: false,
      handleClosePolygon: vi.fn(async () => {}),
      clearPolygon: vi.fn(),
      handleConfirmBrush: vi.fn(async () => {}),
    },
    review: {
      isGroupActionMode: false,
      clearGroupSelection: vi.fn(),
      groupSelectionBBox: null,
      groupBboxHighlightedSegmentIds: [],
      handleBatchGroupAction: vi.fn(async () => {}),
      activeGroupActionLabelState: null,
      leftMode: "hover",
      correctionTool: "draw",
      handleAcceptPolygon: vi.fn(async () => {}),
    },
    ...overrides,
  };
}

describe("useSegmentationKeyboardShortcuts", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
  });

  it("toggles navigate mode with the A shortcut", () => {
    const args = makeArgs();
    renderHook(() => useSegmentationKeyboardShortcuts(args));

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "a" }));

    expect(args.toggleLeftNavigateMode).toHaveBeenCalled();
  });

  it("clears the completed ROI draft with Delete", () => {
    const args = makeArgs({
      completedRoi: {
        isActive: true,
        canClosePolygon: false,
        hasDraft: true,
        handleClosePolygon: vi.fn(async () => {}),
        clearDraft: vi.fn(),
      },
    });
    renderHook(() => useSegmentationKeyboardShortcuts(args));

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Delete" }));

    expect(args.completedRoi.clearDraft).toHaveBeenCalled();
  });

  it("closes the ER polygon draft with R (commit)", () => {
    const args = makeArgs({
      erPolygon: {
        isActive: true,
        canClosePolygon: true,
        hasDraft: true,
        handleClosePolygon: vi.fn(async () => {}),
        clearDraft: vi.fn(),
      },
    });
    renderHook(() => useSegmentationKeyboardShortcuts(args));

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "r" }));

    expect(args.erPolygon.handleClosePolygon).toHaveBeenCalled();
  });

  it("clears the ER polygon draft with Delete", () => {
    const args = makeArgs({
      erPolygon: {
        isActive: true,
        canClosePolygon: false,
        hasDraft: true,
        handleClosePolygon: vi.fn(async () => {}),
        clearDraft: vi.fn(),
      },
    });
    renderHook(() => useSegmentationKeyboardShortcuts(args));

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Delete" }));

    expect(args.erPolygon.clearDraft).toHaveBeenCalled();
  });

  it("closes the tissue polygon draft with R", () => {
    const args = makeArgs({
      tissue: {
        enabled: true,
        polygonCanClose: true,
        polygonHasDraft: true,
        canConfirmBrush: false,
        handleClosePolygon: vi.fn(async () => {}),
        clearPolygon: vi.fn(),
        handleConfirmBrush: vi.fn(async () => {}),
      },
    });
    renderHook(() => useSegmentationKeyboardShortcuts(args));

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "r" }));

    expect(args.tissue.handleClosePolygon).toHaveBeenCalled();
  });

  it("confirms the drawn area with Enter in draw mode", () => {
    drawingState.brushStrokes = [
      { id: "stroke-1", label: 1, size: 24, points: [{ x: 1, y: 1 }] },
    ] as never;
    const args = makeArgs({
      review: {
        ...makeArgs().review,
        leftMode: "draw",
        correctionTool: "draw",
      },
    });
    renderHook(() => useSegmentationKeyboardShortcuts(args));

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));

    expect(args.review.handleAcceptPolygon).toHaveBeenCalled();
    drawingState.brushStrokes = [];
  });

  it("confirms the tissue brush with Enter", () => {
    const args = makeArgs({
      tissue: {
        enabled: true,
        polygonCanClose: false,
        polygonHasDraft: false,
        canConfirmBrush: true,
        handleClosePolygon: vi.fn(async () => {}),
        clearPolygon: vi.fn(),
        handleConfirmBrush: vi.fn(async () => {}),
      },
    });
    renderHook(() => useSegmentationKeyboardShortcuts(args));

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));

    expect(args.tissue.handleConfirmBrush).toHaveBeenCalled();
  });

});
