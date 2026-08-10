import type { LeftMode } from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteSegmentsBatch,
  querySegmentsInRegion,
  updateSegmentLabelsBatch,
} from "@/shared/api/segmentations/annotations";
import { RIGHT_BBOX_DRAG_THRESHOLD_PX } from "@/features/segmentation/screen/utils/constants";
import { toSyntheticSegmentObject } from "@/features/segmentation/screen/utils/optimisticSegments";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { Point } from "@/utils/geometry";
import type { LabelState, BBox } from "@/shared/types/common";
import type {
  CorrectionModeState,
  SegmentationOverlayMutationState,
  SegmentObject,
} from "@/shared/types/segmentation";
import type { ImageSegmentation } from "@/shared/types/images";

interface UseReviewGroupSelectionArgs {
  currentSegmentation: ImageSegmentation | null;
  activeSourceModel: string | null;
  isErSegmentation: boolean;
  workflowMode: "annotate" | "review" | "uncertain";
  correctionMode: CorrectionModeState;
  leftMode: LeftMode;
  hoverActionMode: "confirm" | "reject" | "group-confirm" | "group-reject" | "test";
  registerAnnotationActivity: () => void;
  applyOptimisticLabel: (
    segmentId: string,
    labelState: LabelState,
    sourceSegment?: SegmentObject | null,
    options?: { stageOverlay?: boolean }
  ) => void;
  rollbackOptimisticLabel: (segmentId: string) => void;
  clearHoverInteraction: () => void;
  stageOptimisticRevisionTargets: (segmentIds: string[], targetRevision?: number | null) => void;
  getOptimisticTargetRevision: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => number | null;
  handleOverlayMutationRefresh: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => void;
  showErrorToast: (message: string) => void;
}

export function useReviewGroupSelection({
  currentSegmentation,
  activeSourceModel,
  isErSegmentation,
  workflowMode,
  correctionMode,
  leftMode,
  hoverActionMode,
  registerAnnotationActivity,
  applyOptimisticLabel,
  rollbackOptimisticLabel,
  clearHoverInteraction,
  stageOptimisticRevisionTargets,
  getOptimisticTargetRevision,
  handleOverlayMutationRefresh,
  showErrorToast,
}: UseReviewGroupSelectionArgs) {
  const [groupSelectionBBox, setGroupSelectionBBox] = useState<BBox | null>(null);
  const [groupBboxHighlightedSegmentIds, setGroupBboxHighlightedSegmentIds] = useState<
    string[]
  >([]);
  const [groupSelectionPreviewSegments, setGroupSelectionPreviewSegments] = useState<
    SegmentObject[]
  >([]);
  const groupBboxDragStartRef = useRef<{ image: Point; screen: Point } | null>(null);
  const groupSelectionQueryRef = useRef(0);

  const clearGroupBboxSelection = useCallback(() => {
    groupBboxDragStartRef.current = null;
    groupSelectionQueryRef.current += 1;
    setGroupSelectionBBox(null);
    setGroupBboxHighlightedSegmentIds([]);
    setGroupSelectionPreviewSegments([]);
  }, []);

  useEffect(() => {
    if (isGroupActionMode(workflowMode, correctionMode, leftMode, hoverActionMode)) return;
    clearGroupBboxSelection();
  }, [clearGroupBboxSelection, correctionMode, hoverActionMode, leftMode, workflowMode]);

  const handleBatchGroupAction = useCallback(
    async (segmentIds: string[], nextLabelState: "CONFIRMED" | "EXCLUDED") => {
      if (!currentSegmentation || segmentIds.length === 0) return;
      registerAnnotationActivity();

      const previewById = new Map(
        groupSelectionPreviewSegments.map((segment) => [segment.id, segment])
      );
      const actionableSegments = Array.from(new Set(segmentIds.map(String)))
        .map((segmentId) => previewById.get(segmentId) ?? null)
        .filter(
          (segment): segment is SegmentObject =>
            segment !== null &&
            (segment.label_state === "INFERRED" || segment.label_state === "CANDIDATE")
        );
      if (actionableSegments.length === 0) return;

      const uniqueIds = actionableSegments.map((segment) => segment.id);
      const stageOverlay = nextLabelState !== "EXCLUDED";
      const fallbackMessage =
        nextLabelState === "CONFIRMED"
          ? "Failed to confirm selected objects."
          : "Failed to reject selected objects.";

      for (const segment of actionableSegments) {
        applyOptimisticLabel(segment.id, nextLabelState, segment, { stageOverlay });
      }
      clearGroupBboxSelection();
      clearHoverInteraction();

      try {
        if (isErSegmentation && nextLabelState === "EXCLUDED") {
          // ER rejects delete the candidates outright -- ER objects are only
          // CANDIDATE or CONFIRMED, never EXCLUDED.
          const response = await deleteSegmentsBatch(currentSegmentation.id, {
            ids: uniqueIds,
            source_model: activeSourceModel,
          });
          stageOptimisticRevisionTargets(
            uniqueIds,
            getOptimisticTargetRevision(response.overlay)
          );
          handleOverlayMutationRefresh(response.overlay);
        } else {
          const response = await updateSegmentLabelsBatch({
            labels: uniqueIds.map((segmentId) => ({
              id: segmentId,
              label_state: nextLabelState,
            })),
            source_model: activeSourceModel,
          });
          stageOptimisticRevisionTargets(
            uniqueIds,
            getOptimisticTargetRevision(response.overlays?.[currentSegmentation.id])
          );
          handleOverlayMutationRefresh(response.overlays?.[currentSegmentation.id]);
        }
      } catch (error) {
        for (const segmentId of uniqueIds) {
          rollbackOptimisticLabel(segmentId);
        }
        showErrorToast(extractApiErrorMessage(error, fallbackMessage));
        console.error("Failed to apply group action:", error);
      }
    },
    [
      applyOptimisticLabel,
      activeSourceModel,
      isErSegmentation,
      clearGroupBboxSelection,
      clearHoverInteraction,
      currentSegmentation,
      getOptimisticTargetRevision,
      groupSelectionPreviewSegments,
      handleOverlayMutationRefresh,
      registerAnnotationActivity,
      rollbackOptimisticLabel,
      showErrorToast,
      stageOptimisticRevisionTargets,
    ]
  );

  const handleToolbarGroupAction = useCallback(
    (mode: "group-confirm" | "group-reject") => {
      if (!groupSelectionBBox || groupBboxHighlightedSegmentIds.length === 0) {
        return;
      }
      void handleBatchGroupAction(
        [...groupBboxHighlightedSegmentIds],
        mode === "group-reject" ? "EXCLUDED" : "CONFIRMED"
      );
    },
    [groupBboxHighlightedSegmentIds, groupSelectionBBox, handleBatchGroupAction]
  );

  const handleGroupImagePress = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (!isGroupActionMode(workflowMode, correctionMode, leftMode, hoverActionMode)) return;
      registerAnnotationActivity();
      groupBboxDragStartRef.current = {
        image: imagePoint,
        screen: screenPoint,
      };
      setGroupSelectionBBox(null);
      setGroupBboxHighlightedSegmentIds([]);
      setGroupSelectionPreviewSegments([]);
    },
    [correctionMode, hoverActionMode, leftMode, registerAnnotationActivity, workflowMode]
  );

  const handleGroupImageRelease = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (!isGroupActionMode(workflowMode, correctionMode, leftMode, hoverActionMode)) return;

      const start = groupBboxDragStartRef.current;
      groupBboxDragStartRef.current = null;
      if (!start) return;
      registerAnnotationActivity();

      const deltaX = screenPoint.x - start.screen.x;
      const deltaY = screenPoint.y - start.screen.y;
      const dragDistance = Math.hypot(deltaX, deltaY);
      if (dragDistance < RIGHT_BBOX_DRAG_THRESHOLD_PX) {
        clearGroupBboxSelection();
        return;
      }

      const bbox: BBox = {
        x0: Math.min(start.image.x, imagePoint.x),
        y0: Math.min(start.image.y, imagePoint.y),
        x1: Math.max(start.image.x, imagePoint.x),
        y1: Math.max(start.image.y, imagePoint.y),
      };
      if (bbox.x1 <= bbox.x0 || bbox.y1 <= bbox.y0) {
        clearGroupBboxSelection();
        return;
      }

      setGroupSelectionBBox(bbox);
      setGroupBboxHighlightedSegmentIds([]);
    },
    [
      clearGroupBboxSelection,
      correctionMode,
      hoverActionMode,
      leftMode,
      registerAnnotationActivity,
      workflowMode,
    ]
  );

  const handleGroupImageDrag = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (!isGroupActionMode(workflowMode, correctionMode, leftMode, hoverActionMode)) return;
      const start = groupBboxDragStartRef.current;
      if (!start) return;
      registerAnnotationActivity();

      const deltaX = screenPoint.x - start.screen.x;
      const deltaY = screenPoint.y - start.screen.y;
      const dragDistance = Math.hypot(deltaX, deltaY);
      if (dragDistance < RIGHT_BBOX_DRAG_THRESHOLD_PX) {
        setGroupSelectionBBox(null);
        setGroupBboxHighlightedSegmentIds([]);
        setGroupSelectionPreviewSegments([]);
        return;
      }

      const bbox: BBox = {
        x0: Math.min(start.image.x, imagePoint.x),
        y0: Math.min(start.image.y, imagePoint.y),
        x1: Math.max(start.image.x, imagePoint.x),
        y1: Math.max(start.image.y, imagePoint.y),
      };
      if (bbox.x1 <= bbox.x0 || bbox.y1 <= bbox.y0) {
        setGroupSelectionBBox(null);
        setGroupBboxHighlightedSegmentIds([]);
        setGroupSelectionPreviewSegments([]);
        return;
      }

      setGroupSelectionBBox(bbox);
      setGroupBboxHighlightedSegmentIds([]);
    },
    [correctionMode, hoverActionMode, leftMode, registerAnnotationActivity, workflowMode]
  );

  useEffect(() => {
    if (
      !isGroupActionMode(workflowMode, correctionMode, leftMode, hoverActionMode) ||
      !groupSelectionBBox ||
      !currentSegmentation
    ) {
      return;
    }
    const requestId = ++groupSelectionQueryRef.current;
    const timeout = window.setTimeout(() => {
      void querySegmentsInRegion(currentSegmentation.id, {
        bbox: groupSelectionBBox,
        states: ["CANDIDATE", "INFERRED"],
        source_model: activeSourceModel,
        include_geometry: true,
      })
        .then((result) => {
          if (requestId !== groupSelectionQueryRef.current) return;
          const segmentsForSelection = result.segments.map((segment) =>
            toSyntheticSegmentObject(currentSegmentation.id, segment)
          );
          setGroupSelectionPreviewSegments(segmentsForSelection);
          setGroupBboxHighlightedSegmentIds(segmentsForSelection.map((segment) => segment.id));
        })
        .catch((error) => {
          if (requestId !== groupSelectionQueryRef.current) return;
          console.error("Failed to query group-selection region:", error);
          setGroupSelectionPreviewSegments([]);
          setGroupBboxHighlightedSegmentIds([]);
        });
    }, 120);
    return () => window.clearTimeout(timeout);
  }, [
    correctionMode,
    currentSegmentation,
    activeSourceModel,
    groupSelectionBBox,
    hoverActionMode,
    leftMode,
    workflowMode,
  ]);

  return {
    groupSelectionBBox,
    groupBboxHighlightedSegmentIds,
    groupSelectionPreviewSegments,
    clearGroupBboxSelection,
    handleBatchGroupAction,
    handleToolbarGroupAction,
    handleGroupImagePress,
    handleGroupImageDrag,
    handleGroupImageRelease,
  };
}

function isGroupActionMode(
  workflowMode: "annotate" | "review" | "uncertain",
  correctionMode: CorrectionModeState,
  leftMode: LeftMode,
  hoverActionMode: "confirm" | "reject" | "group-confirm" | "group-reject" | "test"
) {
  return (
    workflowMode === "review" &&
    correctionMode.reviewPhase === "model" &&
    leftMode === "hover" &&
    (hoverActionMode === "group-confirm" || hoverActionMode === "group-reject")
  );
}
