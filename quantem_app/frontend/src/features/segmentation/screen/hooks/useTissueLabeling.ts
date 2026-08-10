import { useCallback, useEffect, useState } from "react";
import { useDrawing } from "@/hooks/useDrawing";
import { removeSegmentationArea } from "@/shared/api/segmentations/annotations";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { mutationNoticeMessage } from "@/features/segmentation/screen/utils/mutationNotices";
import { usePolygonTraceWorkflow } from "@/features/segmentation/screen/hooks/usePolygonTraceWorkflow";
import type { Point } from "@/utils/geometry";
import type { ImageSegmentation } from "@/shared/types/images";
import type { ConfirmBatchResponse } from "@/shared/types/segmentation";

/** The three tools of the minimal tissue-mask labeling view. */
export type TissueTool = "brush" | "polygon" | "exclude";

interface SubmitConfirmedGeometriesOptions {
  geometries: Array<Array<[number, number]>>;
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
  /**
   * Force an immediate (non-deferred) overlay + segment refetch. The exclude
   * tool has no optimistic preview, so it must refresh the id-map overlay right
   * away or the cut looks like it did nothing.
   */
  refreshSegmentViews: (options?: { deferOverlayRefresh?: boolean }) => Promise<void>;
  /**
   * Re-enable overlay-manifest polling. Cutting a hole rebuilds the overlay
   * pyramid asynchronously; annotation activity disables polling, so without
   * turning it back on the finished rebuild is never picked up and the hole
   * never appears.
   */
  setOverlayManifestPollingEnabled: (enabled: boolean) => void;
  clearHoverInteraction: () => void;
}

/**
 * Minimal tissue-mask labeling view.
 *
 * A tissue mask is a single hand-drawn foreground region (used to cut out white
 * space -- vessels, resin, padding). There is no model, no review phase
 * and no confirmed-area (training-mask) concept: the brush and polygon tools
 * paint directly into one merged CONFIRMED region, and the exclude-polygon tool
 * carves areas back out of it.
 *
 * - "brush": paint strokes, then confirm -> merged into the mask.
 * - "polygon": click-to-trace a ring (R to close) -> merged into the mask.
 * - "exclude": click-to-trace a ring (R to close) -> subtracted from the mask.
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
  refreshSegmentViews,
  setOverlayManifestPollingEnabled,
  clearHoverInteraction,
}: UseTissueLabelingArgs) {
  const [tool, setTool] = useState<TissueTool>("brush");

  // Reset to the brush and drop any uncommitted brush strokes when the
  // segmentation changes or the view is left.
  useEffect(() => {
    setTool("brush");
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

  const addPolygon = usePolygonTraceWorkflow({
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
          mergeOverlaps: true,
          manualCreation: true,
        });
        const notice = mutationNoticeMessage(response, {
          nothingStoredMessage:
            "Nothing was added to the tissue mask: that ring encloses no area more than a pixel across.",
        });
        if (notice) showNoticeToast(notice);
      },
      [showNoticeToast, submitConfirmedGeometriesOptimistically]
    ),
  });

  const excludePolygon = usePolygonTraceWorkflow({
    active: enabled && tool === "exclude",
    idPrefix: "tissue-exclude-polygon",
    isPointInsideImageBounds,
    registerAnnotationActivity,
    showErrorToast,
    resetKey: currentSegmentationId,
    commitErrorMessage: "Failed to exclude the area from the tissue mask.",
    onCommit: useCallback(
      async (ring: Array<[number, number]>) => {
        if (!currentSegmentation) return;
        const response = await removeSegmentationArea(currentSegmentation.id, {
          areas: [{ geometry_coords: ring }],
        });
        // Cutting a hole edits the confirmed geometry directly and rebuilds the
        // overlay pyramid asynchronously. With no optimistic preview, re-enable
        // polling (annotation activity turned it off) and refetch now so the
        // finished rebuild is picked up and the hole appears.
        setOverlayManifestPollingEnabled(true);
        await refreshSegmentViews();
        clearHoverInteraction();
        // A ring drawn off the mask cuts nothing, and with no optimistic
        // preview the screen after that is identical to the screen after a
        // successful cut. Same for a cut whose objects could not be re-measured.
        const notice = mutationNoticeMessage(response, {
          nothingStoredMessage:
            "Nothing was excluded: that ring does not overlap the tissue mask.",
        });
        if (notice) showNoticeToast(notice);
      },
      [
        clearHoverInteraction,
        currentSegmentation,
        refreshSegmentViews,
        setOverlayManifestPollingEnabled,
        showNoticeToast,
      ]
    ),
  });

  const hasBrushStrokes = drawing.brushStrokes.length > 0;
  const [confirmingBrush, setConfirmingBrush] = useState(false);
  const canConfirmBrush = enabled && tool === "brush" && hasBrushStrokes && !confirmingBrush;

  const handleConfirmBrush = useCallback(async () => {
    if (!currentSegmentation || confirmingBrush) return;
    const brushPolygons = drawing.getBrushPolygons();
    // Posted as drawn. See `SEGMENT_SMOOTHING_TOLERANCE` in `@/config`: the
    // ring a brush produces is the exact pixel boundary of what was painted,
    // and simplifying it at 1.0 px moved that boundary by up to +7.7% of area.
    const geometries = brushPolygons
      .map((polygon) =>
        polygon.map((point) => [point.x, point.y] as [number, number])
      )
      .filter((coords) => coords.length >= 3);
    if (geometries.length === 0) return;

    registerAnnotationActivity();
    setConfirmingBrush(true);
    try {
      const response = await submitConfirmedGeometriesOptimistically({
        geometries,
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

  const activePolygonTool =
    tool === "polygon" ? addPolygon : tool === "exclude" ? excludePolygon : null;

  return {
    enabled,
    tool,
    setTool,
    addPolygon,
    excludePolygon,
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
