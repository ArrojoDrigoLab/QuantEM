import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSegmentationKeyboardShortcuts } from "@/features/segmentation/screen/hooks/useSegmentationKeyboardShortcuts";
import {
  drawingState,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { panKeyState, resetPanKeyStateForTests } from "@/viewer/panKeyState";

type PointVerbsArg = NonNullable<
  Parameters<typeof useSegmentationKeyboardShortcuts>[0]["pointVerbs"]
>;

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

  /**
   * The verbs, which are the whole point of the reform: a decision costs a
   * keystroke over the object, not a trip to the sidebar and back.
   */
  describe("the point verbs", () => {
    function makeVerbs(overrides: Partial<PointVerbsArg> = {}): PointVerbsArg {
      return {
        hoverPoint: { x: 120, y: 240 },
        hasHoverTarget: true,
        keep: vi.fn(),
        remove: vi.fn(),
        unmark: vi.fn(),
        ...overrides,
      };
    }

    beforeEach(() => {
      resetPanKeyStateForTests();
    });

    it("keeps the hovered object when space is tapped, with no mouse travel", () => {
      const pointVerbs = makeVerbs();
      renderHook(() => useSegmentationKeyboardShortcuts(makeArgs({ pointVerbs })));

      window.dispatchEvent(new KeyboardEvent("keydown", { key: " ", code: "Space" }));
      window.dispatchEvent(new KeyboardEvent("keyup", { key: " ", code: "Space" }));

      expect(pointVerbs.keep).toHaveBeenCalledWith({ x: 120, y: 240 });
    });

    it("does not keep anything when that space press was a pan", () => {
      const pointVerbs = makeVerbs();
      renderHook(() => useSegmentationKeyboardShortcuts(makeArgs({ pointVerbs })));

      window.dispatchEvent(new KeyboardEvent("keydown", { key: " ", code: "Space" }));
      // The viewer reports that the held space actually moved the image.
      panKeyState.markSpacePan();
      window.dispatchEvent(new KeyboardEvent("keyup", { key: " ", code: "Space" }));

      expect(pointVerbs.keep).not.toHaveBeenCalled();
    });

    it("removes with x and un-marks with u", () => {
      const pointVerbs = makeVerbs();
      renderHook(() => useSegmentationKeyboardShortcuts(makeArgs({ pointVerbs })));

      window.dispatchEvent(new KeyboardEvent("keydown", { key: "x" }));
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "u" }));

      expect(pointVerbs.remove).toHaveBeenCalledWith({ x: 120, y: 240 });
      expect(pointVerbs.unmark).toHaveBeenCalledWith({ x: 120, y: 240 });
    });

    it("does nothing at all when the pointer is over no object", () => {
      const pointVerbs = makeVerbs({ hoverPoint: null, hasHoverTarget: false });
      renderHook(() => useSegmentationKeyboardShortcuts(makeArgs({ pointVerbs })));

      window.dispatchEvent(new KeyboardEvent("keydown", { key: "x" }));
      window.dispatchEvent(new KeyboardEvent("keydown", { key: " ", code: "Space" }));
      window.dispatchEvent(new KeyboardEvent("keyup", { key: " ", code: "Space" }));

      expect(pointVerbs.remove).not.toHaveBeenCalled();
      expect(pointVerbs.keep).not.toHaveBeenCalled();
    });

    it("stays out of the way while Navigate is on", () => {
      const pointVerbs = makeVerbs();
      renderHook(() =>
        useSegmentationKeyboardShortcuts(makeArgs({ pointVerbs, leftNavigateMode: true }))
      );

      window.dispatchEvent(new KeyboardEvent("keydown", { key: "x" }));
      window.dispatchEvent(new KeyboardEvent("keydown", { key: " ", code: "Space" }));
      window.dispatchEvent(new KeyboardEvent("keyup", { key: " ", code: "Space" }));

      expect(pointVerbs.remove).not.toHaveBeenCalled();
      expect(pointVerbs.keep).not.toHaveBeenCalled();
    });

    it("routes z, [ and ] to whoever owns them, and leaves them alone otherwise", () => {
      const undo = vi.fn();
      const next = vi.fn();
      const previous = vi.fn();
      const pointVerbs = makeVerbs({ undo, next, previous });
      renderHook(() => useSegmentationKeyboardShortcuts(makeArgs({ pointVerbs })));

      window.dispatchEvent(new KeyboardEvent("keydown", { key: "z" }));
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "]" }));
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "[" }));

      expect(undo).toHaveBeenCalledTimes(1);
      expect(next).toHaveBeenCalledTimes(1);
      expect(previous).toHaveBeenCalledTimes(1);
    });

    it("undoes even in Navigate mode, where the user is looking around", () => {
      const undo = vi.fn();
      const pointVerbs = makeVerbs({ undo });
      renderHook(() =>
        useSegmentationKeyboardShortcuts(makeArgs({ pointVerbs, leftNavigateMode: true }))
      );

      window.dispatchEvent(new KeyboardEvent("keydown", { key: "z" }));

      expect(undo).toHaveBeenCalledTimes(1);
    });

    it("does not swallow an unclaimed key", () => {
      const pointVerbs = makeVerbs();
      renderHook(() => useSegmentationKeyboardShortcuts(makeArgs({ pointVerbs })));

      const event = new KeyboardEvent("keydown", { key: "z", cancelable: true });
      window.dispatchEvent(event);

      expect(event.defaultPrevented).toBe(false);
    });
  });
});
