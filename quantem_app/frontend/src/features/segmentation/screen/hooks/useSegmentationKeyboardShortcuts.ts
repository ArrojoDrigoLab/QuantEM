import { useEffect } from "react";
import { useDrawing } from "@/hooks/useDrawing";
import type { LeftMode } from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import type { Point } from "@/utils/geometry";
import { panKeyState } from "@/viewer/panKeyState";
import type { CorrectionTool } from "@/shared/types";

/**
 * The verbs that act on whatever the pointer is over.
 *
 * Proofreading is one decision repeated hundreds of times, and every one of
 * them cost a round trip to a sidebar button: point at the object, drag the
 * mouse across the screen, click "Confirm Object", drag back, find your place
 * again. The keys put the decision where the eye already is.
 *
 * `undo`, `previous` and `next` are named here and left optional on purpose.
 * They belong to later packages, and reserving the names now means those
 * packages pass a function rather than reopening this file and re-deciding what
 * `z` should mean. A key with no handler is not consumed, so nothing is
 * swallowed before it does something.
 */
interface PointVerbs {
  /** Where the pointer is over the image, or null when it is over nothing. */
  hoverPoint: Point | null;
  /** There is an object under the pointer to act on. */
  hasHoverTarget: boolean;
  /** Space: this one is real, keep it. */
  keep: (point: Point) => void | Promise<void>;
  /** x: this one is not a real object -- a recorded negative, not a shrug. */
  remove: (point: Point) => void | Promise<void>;
  /** u: take back a mark, putting the object back to a guess. */
  unmark: (point: Point) => void | Promise<void>;
  /** z: undo the last label change. */
  undo?: (() => void | Promise<void>) | null;
  /** ] : move to the next object. */
  next?: (() => void) | null;
  /** [ : move to the previous object. */
  previous?: (() => void) | null;
}

interface UseSegmentationKeyboardShortcutsArgs {
  leftNavigateMode: boolean;
  toggleLeftNavigateMode: () => void;
  cycleHoverIndex: (direction: "next" | "prev") => void;
  pointVerbs?: PointVerbs;
  drawing: ReturnType<typeof useDrawing>;
  removeArea: {
    mode: "none" | "objects" | "area";
    clearDrawing: () => void;
    canApply: boolean;
    handleApply: () => Promise<void>;
  };
  completedRoi: {
    isActive: boolean;
    canClosePolygon: boolean;
    hasDraft: boolean;
    handleClosePolygon: () => Promise<void> | void;
    clearDraft: () => void;
  };
  erPolygon: {
    isActive: boolean;
    canClosePolygon: boolean;
    hasDraft: boolean;
    handleClosePolygon: () => Promise<void> | void;
    clearDraft: () => void;
  };
  tissue: {
    enabled: boolean;
    polygonCanClose: boolean;
    polygonHasDraft: boolean;
    canConfirmBrush: boolean;
    handleClosePolygon: () => Promise<void> | void;
    clearPolygon: () => void;
    handleConfirmBrush: () => Promise<void> | void;
  };
  review: {
    isGroupActionMode: boolean;
    clearGroupSelection: () => void;
    groupSelectionBBox: unknown;
    groupBboxHighlightedSegmentIds: string[];
    handleBatchGroupAction: (
      segmentIds: string[],
      nextLabelState: "CONFIRMED" | "EXCLUDED"
    ) => Promise<void>;
    activeGroupActionLabelState: "CONFIRMED" | "EXCLUDED" | null;
    leftMode: LeftMode;
    correctionTool: CorrectionTool;
    handleAcceptPolygon: () => Promise<void>;
  };
}

export function useSegmentationKeyboardShortcuts({
  leftNavigateMode,
  toggleLeftNavigateMode,
  cycleHoverIndex,
  pointVerbs,
  drawing,
  removeArea,
  completedRoi,
  erPolygon,
  tissue,
  review,
}: UseSegmentationKeyboardShortcutsArgs) {
  useEffect(() => {
    /**
     * Space came up. Keep the object under the pointer -- unless that press was
     * a pan.
     *
     * Space does two jobs on this canvas and they share a keystroke: hold and
     * drag moves the image, a tap keeps what is under the cursor. So keep fires
     * on release, and only when the viewer says no pan happened in between.
     * Firing on press instead would keep an object every time somebody reached
     * for the pan gesture.
     */
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key !== " " && event.code !== "Space") return;
      const panned = panKeyState.consumeSpacePan();
      if (panned || leftNavigateMode || !pointVerbs) return;
      if (review.leftMode !== "hover" || review.isGroupActionMode) return;
      if (completedRoi.isActive || erPolygon.isActive || tissue.enabled) return;
      if (removeArea.mode === "area") return;
      const target = event.target as HTMLElement | null;
      if (
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable
      ) {
        return;
      }
      const point = pointVerbs.hoverPoint;
      if (!point || !pointVerbs.hasHoverTarget) return;
      event.preventDefault();
      void pointVerbs.keep(point);
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      const consumeKeyEvent = () => {
        event.preventDefault();
        event.stopPropagation();
        if ("stopImmediatePropagation" in event) {
          event.stopImmediatePropagation();
        }
      };

      const target = event.target as HTMLElement | null;
      const isEditable =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable;
      if (isEditable) return;

      const isNavigateToggleKey =
        !event.repeat &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.altKey &&
        event.key.toLowerCase() === "a";
      if (isNavigateToggleKey) {
        consumeKeyEvent();
        toggleLeftNavigateMode();
        return;
      }

      // Undo and step-to-next-object are not tied to a tool, so they work in
      // every mode -- including Navigate, where the user is looking around
      // between decisions and is exactly the person who wants to take the last
      // one back.
      const isPlainKey =
        !event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey;
      if (isPlainKey && pointVerbs) {
        if (event.key.toLowerCase() === "z" && pointVerbs.undo) {
          consumeKeyEvent();
          void pointVerbs.undo();
          return;
        }
        if (event.key === "]" && pointVerbs.next) {
          consumeKeyEvent();
          pointVerbs.next();
          return;
        }
        if (event.key === "[" && pointVerbs.previous) {
          consumeKeyEvent();
          pointVerbs.previous();
          return;
        }
      }

      const isArrowKey =
        event.key === "ArrowUp" ||
        event.key === "ArrowRight" ||
        event.key === "ArrowDown" ||
        event.key === "ArrowLeft";
      if (isArrowKey) {
        consumeKeyEvent();
      }

      const isSpace = event.key === " " || event.code === "Space";
      const isEnter = event.key === "Enter";
      const isConfirmKey = isSpace || isEnter;
      const isDeleteKey = event.key === "Delete" || event.key === "Backspace";
      const isPlainShortcut =
        !event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey;
      const lowerKey = event.key.toLowerCase();

      if (removeArea.mode === "area") {
        if (isDeleteKey) {
          consumeKeyEvent();
          removeArea.clearDrawing();
          return;
        }
        if (isConfirmKey) {
          consumeKeyEvent();
          if (removeArea.canApply) {
            void removeArea.handleApply();
          }
          return;
        }
      }

      if (leftNavigateMode) {
        return;
      }

      if (completedRoi.isActive) {
        if (isPlainShortcut && lowerKey === "r" && completedRoi.canClosePolygon) {
          consumeKeyEvent();
          void completedRoi.handleClosePolygon();
          return;
        }
        if (isDeleteKey && completedRoi.hasDraft) {
          consumeKeyEvent();
          completedRoi.clearDraft();
        }
        return;
      }

      if (erPolygon.isActive) {
        if (isPlainShortcut && lowerKey === "r" && erPolygon.canClosePolygon) {
          consumeKeyEvent();
          void erPolygon.handleClosePolygon();
          return;
        }
        if (isDeleteKey && erPolygon.hasDraft) {
          consumeKeyEvent();
          erPolygon.clearDraft();
        }
        return;
      }

      if (tissue.enabled) {
        if (isPlainShortcut && lowerKey === "r" && tissue.polygonCanClose) {
          consumeKeyEvent();
          void tissue.handleClosePolygon();
          return;
        }
        if (isDeleteKey && tissue.polygonHasDraft) {
          consumeKeyEvent();
          tissue.clearPolygon();
          return;
        }
        if (isConfirmKey && tissue.canConfirmBrush) {
          consumeKeyEvent();
          void tissue.handleConfirmBrush();
          return;
        }
        return;
      }

      if (review.isGroupActionMode && isDeleteKey) {
        consumeKeyEvent();
        review.clearGroupSelection();
        return;
      }

      if (isConfirmKey && review.isGroupActionMode && review.groupSelectionBBox) {
        consumeKeyEvent();
        const groupActionLabelState =
          review.activeGroupActionLabelState === "EXCLUDED" ? "EXCLUDED" : "CONFIRMED";
        const groupSelectionIds = [...review.groupBboxHighlightedSegmentIds];
        if (groupSelectionIds.length > 0) {
          void review.handleBatchGroupAction(groupSelectionIds, groupActionLabelState);
        }
        return;
      }

      if (review.leftMode === "hover") {
        // The two verbs that carry the proofreading rhythm. `x` writes a real
        // negative and `u` puts a mark back to a guess -- deliberately
        // different keys for deliberately different acts, because the model
        // learns from the first and not the second.
        const hoverPoint = pointVerbs?.hoverPoint ?? null;
        if (isPlainShortcut && pointVerbs && hoverPoint && pointVerbs.hasHoverTarget) {
          if (lowerKey === "x") {
            consumeKeyEvent();
            void pointVerbs.remove(hoverPoint);
            return;
          }
          if (lowerKey === "u") {
            consumeKeyEvent();
            void pointVerbs.unmark(hoverPoint);
            return;
          }
        }
        if (event.key === "ArrowUp" || event.key === "ArrowRight") {
          cycleHoverIndex("next");
        } else if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
          cycleHoverIndex("prev");
        }
      } else if (review.leftMode === "draw" && review.correctionTool !== "erase") {
        if (isConfirmKey) {
          consumeKeyEvent();
          if (drawing.brushStrokes.length > 0) {
            void review.handleAcceptPolygon();
          }
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown, true);
    window.addEventListener("keyup", handleKeyUp, true);
    return () => {
      window.removeEventListener("keydown", handleKeyDown, true);
      window.removeEventListener("keyup", handleKeyUp, true);
    };
  }, [
    cycleHoverIndex,
    drawing,
    leftNavigateMode,
    completedRoi,
    erPolygon,
    pointVerbs,
    tissue,
    removeArea,
    review,
    toggleLeftNavigateMode,
  ]);
}
