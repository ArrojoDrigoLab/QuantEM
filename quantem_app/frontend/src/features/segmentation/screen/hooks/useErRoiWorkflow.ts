import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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
  refreshSegmentViews?: (options?: {
    deferOverlayRefresh?: boolean;
  }) => Promise<void>;
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
  refreshSegmentViews,
  registerAnnotationActivity,
  onRoiConfirmed,
  showErrorToast,
}: UseErRoiWorkflowArgs) {
  const [placementActive, setPlacementActive] = useState(false);
  const [pendingRoi, setPendingRoi] = useState<PendingRoi | null>(null);
  const [relocatingRoiId, setRelocatingRoiId] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [markingRoiId, setMarkingRoiId] = useState<string | null>(null);
  const [markingRoiDone, setMarkingRoiDone] = useState<boolean | null>(null);
  const [deletingRoiId, setDeletingRoiId] = useState<string | null>(null);
  const [activatingRoiId, setActivatingRoiId] = useState<string | null>(null);
  const workspaceRef = useRef(0);
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

  useEffect(() => {
    workspaceRef.current += 1;
    cancelPlacement();
    setConfirming(false);
    setMarkingRoiId(null);
    setMarkingRoiDone(null);
    setDeletingRoiId(null);
    setActivatingRoiId(null);
    return () => {
      workspaceRef.current += 1;
    };
  }, [currentSegmentationId, cancelPlacement]);

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

  const confirmRoi = useCallback(async (prepopulate = true) => {
    if (!currentSegmentationId || !pendingRoi || confirming) return;

    const workspace = workspaceRef.current;
    setConfirming(true);
    let created = false;
    let replacementRemoved = false;
    try {
      await createSegmentationRoi(currentSegmentationId, {
        x: pendingRoi.x,
        y: pendingRoi.y,
        width: pendingRoi.width,
        height: pendingRoi.height,
        source: "MANUAL",
        prepopulate,
      });
      created = true;

      if (pendingRoi.roiId) {
        await deleteSegmentationRoi(currentSegmentationId, pendingRoi.roiId);
      }
      replacementRemoved = true;

      if (workspace !== workspaceRef.current) return;
      await refetchSegmentationRois();
      if (workspace !== workspaceRef.current) return;
      setPendingRoi(null);
      setPlacementActive(false);
      setRelocatingRoiId(null);
      editGestureRef.current = null;
      registerAnnotationActivity?.();
      onRoiConfirmed?.();
    } catch (error) {
      if (workspace !== workspaceRef.current) return;
      if (created) {
        // The replacement exists and is active even if deleting the old window
        // failed. Refresh before reporting it so the user can remove it later.
        try {
          await refetchSegmentationRois();
        } catch {
          // Keep the original save/refresh error useful even if recovery also
          // fails. A rejected recovery must not escape this event handler.
        }
        if (workspace !== workspaceRef.current) return;
        setPendingRoi(null);
        setPlacementActive(false);
        setRelocatingRoiId(null);
        editGestureRef.current = null;
        showErrorToast(
          replacementRemoved
            ? "The ROI was saved, but its list could not be refreshed. Reload the view to see it."
            : "The new ROI was created, but the previous ROI could not be removed."
        );
      } else {
        showErrorToast(errorMessage(error, "Failed to create ROI."));
      }
    } finally {
      if (workspace === workspaceRef.current) setConfirming(false);
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
      if (!currentSegmentationId || markingRoiId) return;
      const workspace = workspaceRef.current;
      setMarkingRoiId(roiId);
      setMarkingRoiDone(done);
      try {
        const response = await setRoiCompleteForSegmentation(
          currentSegmentationId,
          roiId,
          done
        );
        if (workspace !== workspaceRef.current) return;
        await Promise.all([
          Promise.resolve(refetchSegmentationRois()),
          refreshSegmentViews?.({
            deferOverlayRefresh: Boolean(response.overlay),
          }) ?? Promise.resolve(),
        ]);
      } catch (error) {
        if (workspace !== workspaceRef.current) return;
        showErrorToast(errorMessage(error, "Failed to update ROI status."));
      } finally {
        if (workspace === workspaceRef.current) {
          setMarkingRoiId(null);
          setMarkingRoiDone(null);
        }
      }
    },
    [
      currentSegmentationId,
      markingRoiId,
      refetchSegmentationRois,
      refreshSegmentViews,
      showErrorToast,
    ]
  );

  const activateRoi = useCallback(
    async (roiId: string) => {
      if (!currentSegmentationId || activatingRoiId) return false;
      const workspace = workspaceRef.current;
      setActivatingRoiId(roiId);
      try {
        await activateSegmentationRoi(currentSegmentationId, roiId);
        if (workspace !== workspaceRef.current) return false;
        await refetchSegmentationRois();
        return workspace === workspaceRef.current;
      } catch (error) {
        if (workspace !== workspaceRef.current) return false;
        showErrorToast(errorMessage(error, "Failed to switch ROI."));
        return false;
      } finally {
        if (workspace === workspaceRef.current) setActivatingRoiId(null);
      }
    },
    [activatingRoiId, currentSegmentationId, refetchSegmentationRois, showErrorToast]
  );

  const deleteRoi = useCallback(
    async (roiId: string) => {
      if (!currentSegmentationId) return;
      const workspace = workspaceRef.current;
      setDeletingRoiId(roiId);
      try {
        await deleteSegmentationRoi(currentSegmentationId, roiId);
        if (workspace !== workspaceRef.current) return;
        await refetchSegmentationRois();
        if (workspace !== workspaceRef.current) return;
        registerAnnotationActivity?.();
      } catch (error) {
        if (workspace !== workspaceRef.current) return;
        showErrorToast(errorMessage(error, "Failed to delete ROI."));
      } finally {
        if (workspace === workspaceRef.current) setDeletingRoiId(null);
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
    markingRoiDone,
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
