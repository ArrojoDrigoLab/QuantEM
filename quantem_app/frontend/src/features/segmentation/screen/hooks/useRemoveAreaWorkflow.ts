import { useCallback, useEffect, useMemo, useState } from "react";
import { removeSegmentationArea } from "@/shared/api/segmentations/annotations";
import { useDrawing } from "@/hooks/useDrawing";
import { mutationNoticeMessage } from "@/features/segmentation/screen/utils/mutationNotices";
import { brushStrokesToConnectedPolygons } from "@/utils/brushMask";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { type Point } from "@/utils/geometry";
import type { ImageSegmentation } from "@/shared/types/images";
import type { SegmentationOverlayMutationState } from "@/shared/types/segmentation";
import type { RightPanelRemoveMode } from "@/features/segmentation/components/SegmentationRightPanel";

interface UseRemoveAreaWorkflowArgs {
  currentSegmentation: ImageSegmentation | null;
  currentSegmentationId: string | null;
  registerAnnotationActivity: () => void;
  handleOverlayMutationRefresh: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => void;
  clearHoverInteraction: () => void;
  showErrorToast: (message: string) => void;
  /** Something that worked but did not do what the gesture looked like. */
  showNoticeToast: (message: string) => void;
}

export function useRemoveAreaWorkflow({
  currentSegmentation,
  currentSegmentationId,
  registerAnnotationActivity,
  handleOverlayMutationRefresh,
  clearHoverInteraction,
  showErrorToast,
  showNoticeToast,
}: UseRemoveAreaWorkflowArgs) {
  const removeAreaDrawing = useDrawing();
  const removeAreaBrushStrokes = removeAreaDrawing.brushStrokes;
  const handleBrushStroke = removeAreaDrawing.handleBrushStroke;
  const clearDrawing = removeAreaDrawing.clearDrawing;
  const [rightPanelRemoveMode, setRightPanelRemoveMode] =
    useState<RightPanelRemoveMode>("none");
  const [isRemovingArea, setIsRemovingArea] = useState(false);

  const removeAreaPolygons = useMemo(
    () => brushStrokesToConnectedPolygons(removeAreaBrushStrokes),
    [removeAreaBrushStrokes]
  );

  const canApplyRemoveArea = removeAreaPolygons.length > 0 && !isRemovingArea;

  const handleRemoveAreaBrushStroke = useCallback(
    (points: Point[]) => {
      registerAnnotationActivity();
      handleBrushStroke(points);
    },
    [handleBrushStroke, registerAnnotationActivity]
  );

  const handleApplyRemoveArea = useCallback(async () => {
    if (!currentSegmentation || isRemovingArea || removeAreaPolygons.length === 0) {
      return;
    }
    registerAnnotationActivity();

    // Posted as drawn, for the same reason the confirm path is: this is the
    // area subtracted from a confirmed object, so simplifying it here would
    // change that object's stored area by however much the simplifier moved
    // the erase boundary. See `SEGMENT_SMOOTHING_TOLERANCE` in `@/config`.
    const areas = removeAreaPolygons
      .map((polygon) =>
        polygon.map((point) => [point.x, point.y] as [number, number])
      )
      .filter((geometryCoords) => geometryCoords.length >= 3)
      .map((geometryCoords) => ({ geometry_coords: geometryCoords }));
    if (areas.length === 0) return;

    setIsRemovingArea(true);
    try {
      const response = await removeSegmentationArea(currentSegmentation.id, { areas });
      handleOverlayMutationRefresh(response.overlay);
      clearDrawing();
      clearHoverInteraction();
      // The endpoint answers 207 with a `measurement` block when it reshaped an
      // object it could not re-measure: the new outline is committed and the
      // stored area still describes the shape before the cut. The strokes
      // vanish from the canvas either way, so without this the two outcomes
      // look identical.
      const notice = mutationNoticeMessage(response, {
        nothingStoredMessage:
          "Nothing was removed: the area you drew does not overlap any confirmed object.",
      });
      if (notice) showNoticeToast(notice);
    } catch (error) {
      // Was `console.error` only: an erase that failed outright cleared the
      // strokes and said nothing at all.
      showErrorToast(
        extractApiErrorMessage(error, "Failed to remove the area from confirmed objects.")
      );
    } finally {
      setIsRemovingArea(false);
    }
  }, [
    clearHoverInteraction,
    currentSegmentation,
    handleOverlayMutationRefresh,
    isRemovingArea,
    registerAnnotationActivity,
    clearDrawing,
    removeAreaPolygons,
    showErrorToast,
    showNoticeToast,
  ]);

  useEffect(() => {
    setRightPanelRemoveMode("none");
    setIsRemovingArea(false);
    clearDrawing();
  }, [clearDrawing, currentSegmentationId]);

  return {
    rightPanelRemoveMode,
    setRightPanelRemoveMode,
    isRemovingArea,
    canApplyRemoveArea,
    removeAreaDrawing,
    handleRemoveAreaBrushStroke,
    handleApplyRemoveArea,
  };
}
