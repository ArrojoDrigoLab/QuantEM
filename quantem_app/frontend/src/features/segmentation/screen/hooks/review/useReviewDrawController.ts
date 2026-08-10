import { useCallback } from "react";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { mutationNoticeMessage } from "@/features/segmentation/screen/utils/mutationNotices";
import { useDrawing } from "@/hooks/useDrawing";
import {
  deleteSegmentsBatch,
  querySegmentsInRegion,
} from "@/shared/api/segmentations/annotations";
import { brushStrokesToConnectedPolygons } from "@/utils/brushMask";
import type { Point } from "@/utils/geometry";
import type {
  ConfirmBatchResponse,
  ImageSegmentation,
  SegmentationOverlayMutationState,
} from "@/shared/types";

interface UseReviewDrawControllerArgs {
  currentSegmentation: ImageSegmentation | null;
  activeSourceModel: string | null;
  isErSegmentation: boolean;
  drawing: ReturnType<typeof useDrawing>;
  registerAnnotationActivity: () => void;
  handleOverlayMutationRefresh: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => void;
  showErrorToast: (message: string) => void;
  /** Something that worked but did not do what the gesture looked like. */
  showNoticeToast: (message: string) => void;
  submitConfirmedGeometriesOptimistically: (options: {
    geometries: Array<Array<[number, number]>>;
    samScores?: Array<number | null | undefined>;
    mergeOverlaps?: boolean;
    manualCreation?: boolean;
  }) => Promise<ConfirmBatchResponse | null>;
}

export function useReviewDrawController({
  currentSegmentation,
  activeSourceModel,
  isErSegmentation,
  drawing,
  registerAnnotationActivity,
  handleOverlayMutationRefresh,
  showErrorToast,
  showNoticeToast,
  submitConfirmedGeometriesOptimistically,
}: UseReviewDrawControllerArgs) {
  const handleAcceptPolygon = useCallback(async () => {
    if (!currentSegmentation) return;
    registerAnnotationActivity();
    // Posted as drawn. Every ring here used to be Douglas-Peucker-simplified at
    // 1.0 px on its way out, which moved the outline before the server measured
    // it -- see `SEGMENT_SMOOTHING_TOLERANCE` in `@/config` for the numbers
    // (-7.8% on a 20 px droplet, +4.4% on the same droplet brushed).
    const brushPolygons = drawing.getBrushPolygons();
    const payloadGeometries = brushPolygons.map((polygon) =>
      polygon.map((point) => [point.x, point.y] as [number, number])
    );
    if (payloadGeometries.length === 0 && drawing.pendingPolygon) {
      payloadGeometries.push(
        drawing.pendingPolygon.map(
          (point) => [point.x, point.y] as [number, number]
        )
      );
    }
    if (payloadGeometries.length === 0) return;

    try {
      const response = await submitConfirmedGeometriesOptimistically({
        geometries: payloadGeometries,
        // ER merges a drawn area into overlapping confirmed objects (combining
        // them into one); other organelles keep the split behavior.
        mergeOverlaps: isErSegmentation,
        manualCreation: true,
      });
      drawing.clearDrawing();
      // One stroke, several objects -- or none. A freehand path that crosses
      // itself encloses more than one area and the endpoint keeps every one of
      // them, so the object count moves by more than the gesture suggests; an
      // outline narrower than a pixel is refused and the count does not move at
      // all. Both are in the response, and the canvas looks the same either
      // way once the stroke is cleared.
      const notice = mutationNoticeMessage(response, {
        nothingStoredMessage: isErSegmentation
          ? "Nothing was stored: the drawn area did not add to any confirmed object."
          : "Nothing was stored for that stroke. It encloses no area more than a pixel across.",
      });
      if (notice) showNoticeToast(notice);
    } catch (error) {
      showErrorToast(extractApiErrorMessage(error, "Failed to confirm the drawn shape."));
    }
  }, [
    currentSegmentation,
    drawing,
    isErSegmentation,
    registerAnnotationActivity,
    showErrorToast,
    showNoticeToast,
    submitConfirmedGeometriesOptimistically,
  ]);

  const handleEraseStroke = useCallback(
    async (points: Point[]) => {
      if (!points || points.length === 0) return;
      registerAnnotationActivity();
      // (a) remove un-committed drawn strokes the eraser passes over.
      drawing.eraseBrushStrokesAt(points, drawing.brushSize);
      // (b) delete model candidates under the eraser (never confirmed objects).
      if (!currentSegmentation) return;
      const erasePolygons = brushStrokesToConnectedPolygons([
        { points, size: drawing.brushSize },
      ]);
      const ids = new Set<string>();
      try {
        for (const polygon of erasePolygons) {
          if (polygon.length < 3) continue;
          const coords = polygon.map(
            (point) => [point.x, point.y] as [number, number]
          );
          const { segments } = await querySegmentsInRegion(currentSegmentation.id, {
            polygon_coords: coords,
            states: ["CANDIDATE", "INFERRED"],
            source_model: activeSourceModel,
          });
          for (const segment of segments) {
            ids.add(segment.id);
          }
        }
        if (ids.size === 0) return;
        const response = await deleteSegmentsBatch(currentSegmentation.id, {
          ids: [...ids],
          source_model: activeSourceModel,
        });
        handleOverlayMutationRefresh(response.overlay);
      } catch (error) {
        showErrorToast(extractApiErrorMessage(error, "Failed to erase candidates."));
      }
    },
    [
      activeSourceModel,
      currentSegmentation,
      drawing,
      handleOverlayMutationRefresh,
      registerAnnotationActivity,
      showErrorToast,
    ]
  );

  return {
    handleAcceptPolygon,
    handleEraseStroke,
    canAcceptPolygon:
      drawing.brushStrokes.length > 0 ||
      Boolean(drawing.pendingPolygon && drawing.pendingPolygon.length > 0),
  };
}
