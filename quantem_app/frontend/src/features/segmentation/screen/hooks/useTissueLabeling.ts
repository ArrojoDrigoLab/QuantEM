import { useCallback, useEffect, useState } from "react";
import { useDrawing } from "@/hooks/useDrawing";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { mutationNoticeMessage } from "@/features/segmentation/screen/utils/mutationNotices";
import { usePolygonTraceWorkflow } from "@/features/segmentation/screen/hooks/usePolygonTraceWorkflow";
import type { Point } from "@/utils/geometry";
import type { ImageSegmentation } from "@/shared/types/images";
import type { ConfirmBatchResponse } from "@/shared/types/segmentation";

/** The two shapes available in the minimal tissue-mask labeling view. */
export type TissueTool = "brush" | "polygon";

interface SubmitConfirmedGeometriesOptions {
  geometries?: Array<Array<[number, number]>>;
  geometryRings?: Array<Array<Array<[number, number]>>>;
  operations?: Array<"include" | "exclude">;
  samScores?: Array<number | null | undefined>;
  mergeOverlaps?: boolean;
  manualCreation?: boolean;
}

interface UseTissueLabelingArgs {
  currentSegmentation: ImageSegmentation | null;
  currentSegmentationId: string | null;
  /** Whether the current segmentation is a tissue mask. */
  enabled: boolean;
  isPointInsideImageBounds: (point: Point) => boolean;
  registerAnnotationActivity: () => void;
  showErrorToast: (message: string) => void;
  /** Something that worked but did not do what the gesture looked like. */
  showNoticeToast: (message: string) => void;
  /** The screen-level drawing state (shared with the left-panel brush overlay). */
  drawing: ReturnType<typeof useDrawing>;
  submitConfirmedGeometriesOptimistically: (
    options: SubmitConfirmedGeometriesOptions
  ) => Promise<ConfirmBatchResponse | null>;
}

/**
 * Minimal tissue-mask labeling view.
 *
 * A tissue mask is a single hand-drawn foreground region (used to cut out white
 * space -- vessels, resin, padding). There is no model, no review phase
 * and no confirmed-area (training-mask) concept: the brush and polygon tools
 * patch one binary mask. The independent Include / Exclude toggle decides
 * whether either shape adds to or subtracts from that mask.
 *
 * - "brush": paint strokes, then confirm -> merged into the mask.
 * - "polygon": click-to-trace a ring (R to close) -> patches the mask.
 */
export function useTissueLabeling({
  currentSegmentation,
  currentSegmentationId,
  enabled,
  isPointInsideImageBounds,
  registerAnnotationActivity,
  showErrorToast,
  showNoticeToast,
  drawing,
  submitConfirmedGeometriesOptimistically,
}: UseTissueLabelingArgs) {
  const [tool, setTool] = useState<TissueTool>("brush");

  // Reset to the brush and drop any uncommitted brush strokes when the
  // segmentation changes or the view is left.
  useEffect(() => {
    setTool("brush");
    drawing.setDraftOperation("include");
    drawing.clearDrawing();
    // Intentionally keyed on the segmentation id only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSegmentationId]);

  // Switching away from the brush discards its uncommitted strokes so they can't
  // silently commit later from a different tool.
  useEffect(() => {
    if (tool !== "brush") {
      drawing.clearDrawing();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool]);

  const polygon = usePolygonTraceWorkflow({
    active: enabled && tool === "polygon",
    idPrefix: "tissue-add-polygon",
    isPointInsideImageBounds,
    registerAnnotationActivity,
    showErrorToast,
    resetKey: currentSegmentationId,
    commitErrorMessage: "Failed to add the polygon to the tissue mask.",
    onCommit: useCallback(
      async (ring: Array<[number, number]>) => {
        const response = await submitConfirmedGeometriesOptimistically({
          geometries: [ring],
          operations: [drawing.draftOperation],
          mergeOverlaps: true,
          manualCreation: true,
        });
        const notice = mutationNoticeMessage(response, {
          nothingStoredMessage:
            drawing.draftOperation === "include"
              ? "Nothing was added to the mask: that ring encloses no area more than a pixel across."
              : "Nothing was excluded from the mask.",
        });
        if (notice) showNoticeToast(notice);
      },
      [drawing.draftOperation, showNoticeToast, submitConfirmedGeometriesOptimistically]
    ),
  });

  const hasBrushStrokes = drawing.brushStrokes.length > 0;
  const [confirmingBrush, setConfirmingBrush] = useState(false);
  const canConfirmBrush = enabled && tool === "brush" && hasBrushStrokes && !confirmingBrush;

  const handleConfirmBrush = useCallback(async () => {
    if (!currentSegmentation || confirmingBrush) return;
    const brushPolygons = drawing.getBrushPolygonRings();
    // Posted as drawn. See `SEGMENT_SMOOTHING_TOLERANCE` in `@/config`: the
    // ring a brush produces is the exact pixel boundary of what was painted,
    // and simplifying it at 1.0 px moved that boundary by up to +7.7% of area.
    const geometryRings = brushPolygons
      .map((polygon) =>
        [polygon.exterior, ...polygon.holes].map((ring) =>
          ring.map((point) => [point.x, point.y] as [number, number])
        )
      )
      .filter((rings) => (rings[0]?.length ?? 0) >= 3);
    if (geometryRings.length === 0) return;

    registerAnnotationActivity();
    setConfirmingBrush(true);
    try {
      const response = await submitConfirmedGeometriesOptimistically({
        geometryRings,
        operations: brushPolygons.map((polygon) => polygon.operation),
        mergeOverlaps: true,
        manualCreation: true,
      });
      drawing.clearDrawing();
      const notice = mutationNoticeMessage(response, {
        nothingStoredMessage:
          "Nothing was added to the tissue mask: the strokes enclose no area more than a pixel across.",
      });
      if (notice) showNoticeToast(notice);
    } catch (error) {
      showErrorToast(
        extractApiErrorMessage(error, "Failed to add the brushed area to the tissue mask.")
      );
    } finally {
      setConfirmingBrush(false);
    }
  }, [
    confirmingBrush,
    currentSegmentation,
    drawing,
    registerAnnotationActivity,
    showErrorToast,
    showNoticeToast,
    submitConfirmedGeometriesOptimistically,
  ]);

  const handleClearBrush = useCallback(() => {
    drawing.clearDrawing();
  }, [drawing]);

  const activePolygonTool = tool === "polygon" ? polygon : null;

  return {
    enabled,
    tool,
    setTool,
    operation: drawing.draftOperation,
    setOperation: drawing.setDraftOperation,
    polygon,
    /** The trace whose click/close handlers should receive image events, if any. */
    activePolygonTool,
    brushSize: drawing.brushSize,
    setBrushSize: drawing.setBrushSize,
    hasBrushStrokes,
    canConfirmBrush,
    confirmingBrush,
    handleConfirmBrush,
    handleClearBrush,
  };
}
