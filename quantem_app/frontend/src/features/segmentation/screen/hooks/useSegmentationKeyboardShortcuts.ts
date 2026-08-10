import { useEffect } from "react";
import { useDrawing } from "@/hooks/useDrawing";
import type { LeftMode } from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import type { CorrectionTool } from "@/shared/types";

interface UseSegmentationKeyboardShortcutsArgs {
  leftNavigateMode: boolean;
  toggleLeftNavigateMode: () => void;
  cycleHoverIndex: (direction: "next" | "prev") => void;
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
  drawing,
  removeArea,
  completedRoi,
  erPolygon,
  tissue,
  review,
}: UseSegmentationKeyboardShortcutsArgs) {
  useEffect(() => {
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
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [
    cycleHoverIndex,
    drawing,
    leftNavigateMode,
    completedRoi,
    erPolygon,
    tissue,
    removeArea,
    review,
    toggleLeftNavigateMode,
  ]);
}
