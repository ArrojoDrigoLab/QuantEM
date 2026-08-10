import { useCallback, useMemo, useState } from "react";

import {
  activateSegmentationRoi,
  createSegmentationRoi,
  deleteSegmentationRoi,
  setRoiCompleteForSegmentation,
} from "@/shared/api/segmentations/rois";
import { generateRoiFrameOverlay } from "@/features/segmentation/overlays/roi";
import type { Point } from "@/utils/geometry";
import type { SegmentOverlay } from "@/viewer/types";

/** Fixed ER benchmark ROI size, in source pixels. */
export const ER_ROI_SIZE = 2048;

/** An ROI rectangle that has been placed but not yet created server-side. */
export interface PendingRoi {
  roiId: string | null;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface UseErRoiWorkflowArgs {
  currentSegmentationId: string | null;
  isErSegmentation: boolean;
  image: { width: number; height: number } | null;
  isPointInsideImageBounds: (point: Point) => boolean;
  refetchSegmentationRois: () => Promise<unknown> | void;
  registerAnnotationActivity?: () => void;
  /** Called after a new ROI is successfully created (used to switch to Correct mode). */
  onRoiConfirmed?: () => void;
  showErrorToast: (message: string) => void;
}

function errorMessage(error: unknown, fallback: string): string {
  if (error && typeof error === "object" && "message" in error) {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string" && message) {
      return message;
    }
  }
  return fallback;
}

/**
 * ER-only ROI workflow: place a fixed 2048x2048 ROI by clicking a point in the
 * image, then create it; plus per-organelle "mark ROI as done". Reuses the
 * generic ROI-placement slots of {@link useSegmentationInteractionRouter}, so
 * the screen feeds these handlers into those slots when the segmentation is ER.
 */
export function useErRoiWorkflow({
  currentSegmentationId,
  isErSegmentation,
  image,
  isPointInsideImageBounds,
  refetchSegmentationRois,
  registerAnnotationActivity,
  onRoiConfirmed,
  showErrorToast,
}: UseErRoiWorkflowArgs) {
  const [placementActive, setPlacementActive] = useState(false);
  const [pendingRoi, setPendingRoi] = useState<PendingRoi | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [markingRoiId, setMarkingRoiId] = useState<string | null>(null);
  const [deletingRoiId, setDeletingRoiId] = useState<string | null>(null);
  const [activatingRoiId, setActivatingRoiId] = useState<string | null>(null);

  const startPlacement = useCallback(() => {
    if (!isErSegmentation) return;
    setPlacementActive(true);
    setPendingRoi(null);
  }, [isErSegmentation]);

  const cancelPlacement = useCallback(() => {
    setPlacementActive(false);
    setPendingRoi(null);
  }, []);

  const resolvePendingRoi = useCallback(
    (point: Point): PendingRoi | null => {
      if (!image || !isPointInsideImageBounds(point)) {
        return null;
      }
      const width = Math.max(1, Math.min(ER_ROI_SIZE, image.width));
      const height = Math.max(1, Math.min(ER_ROI_SIZE, image.height));
      const x = Math.round(
        Math.max(0, Math.min(point.x - width / 2, image.width - width))
      );
      const y = Math.round(
        Math.max(0, Math.min(point.y - height / 2, image.height - height))
      );
      return { roiId: null, x, y, width, height };
    },
    [image, isPointInsideImageBounds]
  );

  const confirmRoi = useCallback(async () => {
    if (!currentSegmentationId || !pendingRoi || confirming) {
      return;
    }
    setConfirming(true);
    try {
      await createSegmentationRoi(currentSegmentationId, {
        x: pendingRoi.x,
        y: pendingRoi.y,
        width: pendingRoi.width,
        height: pendingRoi.height,
        source: "MANUAL",
      });
      await refetchSegmentationRois();
      setPendingRoi(null);
      setPlacementActive(false);
      registerAnnotationActivity?.();
      // Hand off to Correct mode now that the ROI exists and is the active ROI.
      onRoiConfirmed?.();
    } catch (error) {
      showErrorToast(errorMessage(error, "Failed to create ROI."));
    } finally {
      setConfirming(false);
    }
  }, [
    confirming,
    currentSegmentationId,
    pendingRoi,
    refetchSegmentationRois,
    registerAnnotationActivity,
    onRoiConfirmed,
    showErrorToast,
  ]);

  const markRoiDone = useCallback(
    async (roiId: string, done: boolean) => {
      if (!currentSegmentationId) {
        return;
      }
      setMarkingRoiId(roiId);
      try {
        await setRoiCompleteForSegmentation(currentSegmentationId, roiId, done);
        await refetchSegmentationRois();
      } catch (error) {
        showErrorToast(errorMessage(error, "Failed to update ROI status."));
      } finally {
        setMarkingRoiId(null);
      }
    },
    [currentSegmentationId, refetchSegmentationRois, showErrorToast]
  );

  const activateRoi = useCallback(
    async (roiId: string) => {
      if (!currentSegmentationId || activatingRoiId) {
        return;
      }
      setActivatingRoiId(roiId);
      try {
        await activateSegmentationRoi(currentSegmentationId, roiId);
        // Refetching updates the active ROI, which re-fits the labeling viewer
        // to its bounds (keyed on the active ROI id) — i.e. "show it".
        await refetchSegmentationRois();
      } catch (error) {
        showErrorToast(errorMessage(error, "Failed to switch ROI."));
      } finally {
        setActivatingRoiId(null);
      }
    },
    [activatingRoiId, currentSegmentationId, refetchSegmentationRois, showErrorToast]
  );

  const deleteRoi = useCallback(
    async (roiId: string) => {
      if (!currentSegmentationId) {
        return;
      }
      setDeletingRoiId(roiId);
      try {
        await deleteSegmentationRoi(currentSegmentationId, roiId);
        await refetchSegmentationRois();
        registerAnnotationActivity?.();
      } catch (error) {
        showErrorToast(errorMessage(error, "Failed to delete ROI."));
      } finally {
        setDeletingRoiId(null);
      }
    },
    [
      currentSegmentationId,
      refetchSegmentationRois,
      registerAnnotationActivity,
      showErrorToast,
    ]
  );

  const pendingRoiOverlay = useMemo<SegmentOverlay | null>(
    () =>
      pendingRoi
        ? generateRoiFrameOverlay(
            {
              x: pendingRoi.x,
              y: pendingRoi.y,
              width: pendingRoi.width,
              height: pendingRoi.height,
            },
            "er-roi-pending"
          )
        : null,
    [pendingRoi]
  );

  return {
    placementActive,
    pendingRoi,
    pendingRoiOverlay,
    confirming,
    markingRoiId,
    deletingRoiId,
    activatingRoiId,
    startPlacement,
    cancelPlacement,
    resolvePendingRoi,
    setPendingRoi,
    confirmRoi,
    markRoiDone,
    deleteRoi,
    activateRoi,
  };
}
