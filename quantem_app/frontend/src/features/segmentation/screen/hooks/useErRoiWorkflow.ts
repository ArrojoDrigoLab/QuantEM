import { useCallback, useMemo, useRef, useState } from "react";

import {
  activateSegmentationRoi,
  createSegmentationRoi,
  deleteSegmentationRoi,
  setRoiCompleteForSegmentation,
} from "@/shared/api/segmentations/rois";
import {
  generateRoiEditHandleOverlays,
  generateRoiFrameOverlay,
} from "@/features/segmentation/overlays/roi";
import {
  resolveRoiEditHandle,
  updateRoiForDrag,
  type RoiEditHandle,
} from "@/features/segmentation/roiEditing";
import type { Point } from "@/utils/geometry";
import type { SegmentationRoi } from "@/shared/types/segmentation";

/** Fixed labeling ROI size, in source pixels. */
export const LABELING_ROI_SIZE = 1024;

/** An ROI rectangle that has been placed but not yet created server-side. */
export interface PendingRoi {
  /** The ROI to replace when the user chose Edit Area; null means create a new one. */
  roiId: string | null;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface UseErRoiWorkflowArgs {
  currentSegmentationId: string | null;
  enabled: boolean;
  image: { width: number; height: number } | null;
  isPointInsideImageBounds: (point: Point) => boolean;
  refetchSegmentationRois: () => Promise<unknown> | void;
  registerAnnotationActivity?: () => void;
  /** Called after an ROI is created or area-edited (to enter Correct mode). */
  onRoiConfirmed?: () => void;
  showErrorToast: (message: string) => void;
}

function errorMessage(error: unknown, fallback: string): string {
  if (error && typeof error === "object" && "message" in error) {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string" && message) return message;
  }
  return fallback;
}

/**
 * Place, resize, relocate, activate, remove, and mark ROI windows for any
 * organelle labeling workflow. Editing safely creates the new window before
 * removing the old one, so a failed save never strands the user without an ROI.
 */
export function useErRoiWorkflow({
  currentSegmentationId,
  enabled,
  image,
  isPointInsideImageBounds,
  refetchSegmentationRois,
  registerAnnotationActivity,
  onRoiConfirmed,
  showErrorToast,
}: UseErRoiWorkflowArgs) {
  const [placementActive, setPlacementActive] = useState(false);
  const [pendingRoi, setPendingRoi] = useState<PendingRoi | null>(null);
  const [relocatingRoiId, setRelocatingRoiId] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [markingRoiId, setMarkingRoiId] = useState<string | null>(null);
  const [deletingRoiId, setDeletingRoiId] = useState<string | null>(null);
  const [activatingRoiId, setActivatingRoiId] = useState<string | null>(null);
  const editGestureRef = useRef<{
    handle: RoiEditHandle;
    dragStart: Point;
    bounds: PendingRoi;
  } | null>(null);

  const beginPlacement = useCallback(
    (roiId: string | null) => {
      if (!enabled) return;
      setRelocatingRoiId(roiId);
      setPlacementActive(true);
      setPendingRoi(null);
      editGestureRef.current = null;
    },
    [enabled]
  );

  const startPlacement = useCallback(() => beginPlacement(null), [beginPlacement]);
  const editRoi = useCallback(
    (roi: SegmentationRoi) => {
      if (!enabled) return;
      setRelocatingRoiId(roi.id);
      setPlacementActive(true);
      setPendingRoi({
        roiId: roi.id,
        x: roi.x,
        y: roi.y,
        width: roi.width,
        height: roi.height,
      });
      editGestureRef.current = null;
    },
    [enabled]
  );

  const cancelPlacement = useCallback(() => {
    setPlacementActive(false);
    setPendingRoi(null);
    setRelocatingRoiId(null);
    editGestureRef.current = null;
  }, []);

  const resolvePendingRoi = useCallback(
    (point: Point): PendingRoi | null => {
      if (!image || !isPointInsideImageBounds(point)) return null;

      const width = Math.max(1, Math.min(LABELING_ROI_SIZE, image.width));
      const height = Math.max(1, Math.min(LABELING_ROI_SIZE, image.height));
      const x = Math.round(
        Math.max(0, Math.min(point.x - width / 2, image.width - width))
      );
      const y = Math.round(
        Math.max(0, Math.min(point.y - height / 2, image.height - height))
      );
      return { roiId: relocatingRoiId, x, y, width, height };
    },
    [image, isPointInsideImageBounds, relocatingRoiId]
  );

  const handleEditPress = useCallback((point: Point) => {
    if (!pendingRoi?.roiId) return;
    const handle = resolveRoiEditHandle(pendingRoi, point);
    editGestureRef.current = handle
      ? { handle, dragStart: point, bounds: pendingRoi }
      : null;
  }, [pendingRoi]);

  const handleEditDrag = useCallback(
    (point: Point) => {
      const gesture = editGestureRef.current;
      if (!gesture || !image) return;
      setPendingRoi({
        roiId: gesture.bounds.roiId,
        ...updateRoiForDrag(
          gesture.bounds,
          gesture.handle,
          gesture.dragStart,
          point,
          image
        ),
      });
    },
    [image]
  );

  const handleEditRelease = useCallback(
    (point: Point) => {
      handleEditDrag(point);
      editGestureRef.current = null;
    },
    [handleEditDrag]
  );

  const confirmRoi = useCallback(async () => {
    if (!currentSegmentationId || !pendingRoi || confirming) return;

    setConfirming(true);
    let created = false;
    try {
      await createSegmentationRoi(currentSegmentationId, {
        x: pendingRoi.x,
        y: pendingRoi.y,
        width: pendingRoi.width,
        height: pendingRoi.height,
        source: "MANUAL",
      });
      created = true;

      if (pendingRoi.roiId) {
        await deleteSegmentationRoi(currentSegmentationId, pendingRoi.roiId);
      }

      await refetchSegmentationRois();
      setPendingRoi(null);
      setPlacementActive(false);
      setRelocatingRoiId(null);
      editGestureRef.current = null;
      registerAnnotationActivity?.();
      onRoiConfirmed?.();
    } catch (error) {
      if (created) {
        // The replacement exists and is active even if deleting the old window
        // failed. Refresh before reporting it so the user can remove it later.
        await refetchSegmentationRois();
        setPendingRoi(null);
        setPlacementActive(false);
        setRelocatingRoiId(null);
        editGestureRef.current = null;
        showErrorToast(
          "The new ROI was created, but the previous ROI could not be removed."
        );
      } else {
        showErrorToast(errorMessage(error, "Failed to create ROI."));
      }
    } finally {
      setConfirming(false);
    }
  }, [
    confirming,
    currentSegmentationId,
    onRoiConfirmed,
    pendingRoi,
    refetchSegmentationRois,
    registerAnnotationActivity,
    showErrorToast,
  ]);

  const markRoiDone = useCallback(
    async (roiId: string, done: boolean) => {
      if (!currentSegmentationId) return;
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
      if (!currentSegmentationId || activatingRoiId) return false;
      setActivatingRoiId(roiId);
      try {
        await activateSegmentationRoi(currentSegmentationId, roiId);
        await refetchSegmentationRois();
        return true;
      } catch (error) {
        showErrorToast(errorMessage(error, "Failed to switch ROI."));
        return false;
      } finally {
        setActivatingRoiId(null);
      }
    },
    [activatingRoiId, currentSegmentationId, refetchSegmentationRois, showErrorToast]
  );

  const deleteRoi = useCallback(
    async (roiId: string) => {
      if (!currentSegmentationId) return;
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

  const pendingRoiOverlays = useMemo(
    () => {
      if (!pendingRoi) return [];
      const bounds = {
        x: pendingRoi.x,
        y: pendingRoi.y,
        width: pendingRoi.width,
        height: pendingRoi.height,
      };
      const frame = generateRoiFrameOverlay(bounds, "labeling-roi-pending");
      return [
        ...(frame ? [frame] : []),
        ...(pendingRoi.roiId ? generateRoiEditHandleOverlays(bounds) : []),
      ];
    },
    [pendingRoi]
  );

  return {
    placementActive,
    pendingRoi,
    pendingRoiOverlays,
    relocatingRoiId,
    confirming,
    markingRoiId,
    deletingRoiId,
    activatingRoiId,
    startPlacement,
    editRoi,
    cancelPlacement,
    resolvePendingRoi,
    setPendingRoi,
    handleEditPress,
    handleEditDrag,
    handleEditRelease,
    confirmRoi,
    markRoiDone,
    deleteRoi,
    activateRoi,
  };
}
