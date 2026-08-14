import { useCallback } from "react";
import {
  deleteSegmentsBatch,
  getSegmentsAtPoint,
} from "@/shared/api/segmentations/annotations";
import { useLabelAnswerQueue } from "@/features/segmentation/screen/hooks/useLabelAnswerQueue";
import { selectBestPointActionSegment, type PointActionMode } from "@/utils/pointAction";
import type { Point } from "@/utils/geometry";
import type { LabelState } from "@/shared/types/common";
import type {
  SegmentationOverlayMutationState,
  SegmentObject,
} from "@/shared/types/segmentation";
import type { ImageSegmentation } from "@/shared/types/images";

interface UseReviewPointActionsArgs {
  currentSegmentation: ImageSegmentation | null;
  activeSourceModel: string | null;
  hoverPoint: Point | null;
  hoverSegments: SegmentObject[];
  highlightedSegmentId: string | null;
  applyLabelOverrides: (items: SegmentObject[]) => SegmentObject[];
  applyOptimisticLabel: (
    segmentId: string,
    labelState: LabelState,
    sourceSegment?: SegmentObject | null,
    options?: { stageOverlay?: boolean }
  ) => void;
  rollbackOptimisticLabel: (segmentId: string) => void;
  hideOptimisticallyDeletedSegment: (segmentId: string) => boolean;
  rollbackOptimisticallyDeletedSegment: (segmentId: string) => void;
  clearHoverInteraction: () => void;
  registerAnnotationActivity: () => void;
  stageOptimisticRevisionTargets: (segmentIds: string[], targetRevision?: number | null) => void;
  getOptimisticTargetRevision: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => number | null;
  handleOverlayMutationRefresh: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => void;
  showErrorToast: (message: string) => void;
}

export function useReviewPointActions({
  currentSegmentation,
  activeSourceModel,
  hoverPoint,
  hoverSegments,
  highlightedSegmentId,
  applyLabelOverrides,
  applyOptimisticLabel,
  rollbackOptimisticLabel,
  hideOptimisticallyDeletedSegment,
  rollbackOptimisticallyDeletedSegment,
  clearHoverInteraction,
  registerAnnotationActivity,
  stageOptimisticRevisionTargets,
  getOptimisticTargetRevision,
  handleOverlayMutationRefresh,
  showErrorToast,
}: UseReviewPointActionsArgs) {
  // Answers do not each get their own request. See useLabelAnswerQueue: a
  // reviewer at speed produced one round-trip per keypress, and the keypress
  // rhythm this screen is built around was throttled by them.
  const { enqueueAnswer, flushAnswers } = useLabelAnswerQueue({
    segmentationId: currentSegmentation?.id ?? null,
    activeSourceModel,
    rollbackOptimisticLabel,
    stageOptimisticRevisionTargets,
    getOptimisticTargetRevision,
    handleOverlayMutationRefresh,
    showErrorToast,
  });

  const resolveHoveredPointActionSegment = useCallback(
    (point: Point, mode: PointActionMode): SegmentObject | null => {
      if (!hoverPoint || hoverSegments.length === 0) {
        return null;
      }
      if (Math.hypot(hoverPoint.x - point.x, hoverPoint.y - point.y) > 6) {
        return null;
      }
      const hoveredSegmentsWithOverrides = applyLabelOverrides(hoverSegments);
      const resolvedHoverSegment =
        hoveredSegmentsWithOverrides.find((segment) => segment.id === highlightedSegmentId) ??
        hoveredSegmentsWithOverrides[0] ??
        null;
      if (!resolvedHoverSegment) {
        return null;
      }
      return selectBestPointActionSegment([resolvedHoverSegment], mode);
    },
    [applyLabelOverrides, highlightedSegmentId, hoverPoint, hoverSegments]
  );

  const handleResetConfirmedToCandidate = useCallback(
    async (point: Point) => {
      if (!currentSegmentation) return;
      registerAnnotationActivity();
      try {
        const matches = await getSegmentsAtPoint(currentSegmentation.id, {
          x: point.x,
          y: point.y,
          states: ["CONFIRMED"],
          ...(activeSourceModel ? { source_model: activeSourceModel } : {}),
        });
        const segment = matches.find((item) => item.label_state === "CONFIRMED");
        if (!segment) return;

        applyOptimisticLabel(segment.id, "CANDIDATE", segment);
        enqueueAnswer({
          segmentId: segment.id,
          labelState: "CANDIDATE",
          fallbackMessage: "Failed to un-mark the selected object.",
        });
        clearHoverInteraction();
      } catch (error) {
        // Only the lookup can fail here; the answer's own failure is the
        // queue's to report and to roll back.
        console.error("Failed to find a kept object at that point:", error);
      }
    },
    [
      applyOptimisticLabel,
      activeSourceModel,
      clearHoverInteraction,
      currentSegmentation,
      enqueueAnswer,
      registerAnnotationActivity,
    ]
  );

  const handleDeleteConfirmedObject = useCallback(
    async (segmentId: string) => {
      if (!currentSegmentation) return;
      // This synchronous client-side hide is the interaction. The request below
      // persists the same hard delete but does not gate either viewer update.
      if (!hideOptimisticallyDeletedSegment(segmentId)) return;
      registerAnnotationActivity();
      clearHoverInteraction();
      try {
        const response = await deleteSegmentsBatch(currentSegmentation.id, {
          ids: [segmentId],
          source_model: activeSourceModel,
        });
        handleOverlayMutationRefresh(response.overlay);
      } catch (error) {
        rollbackOptimisticallyDeletedSegment(segmentId);
        showErrorToast("Failed to delete the selected object.");
        console.error("Failed to hard-delete a confirmed object:", error);
      }
    },
    [
      activeSourceModel,
      clearHoverInteraction,
      currentSegmentation,
      handleOverlayMutationRefresh,
      hideOptimisticallyDeletedSegment,
      registerAnnotationActivity,
      rollbackOptimisticallyDeletedSegment,
      showErrorToast,
    ]
  );

  const handleApplyPointAction = useCallback(
    async (point: Point, mode: "confirm" | "reject") => {
      if (!currentSegmentation) return;
      registerAnnotationActivity();

      let targetSegment: SegmentObject | null = null;
      const pointActionMode: PointActionMode = mode;
      const nextLabelState: LabelState = mode === "confirm" ? "CONFIRMED" : "EXCLUDED";
      const fallbackMessage =
        mode === "confirm"
          ? "Failed to confirm the selected object."
          : "Failed to reject the selected object.";

      try {
        targetSegment = resolveHoveredPointActionSegment(point, pointActionMode);
        if (!targetSegment) {
          const candidates = applyLabelOverrides(
            await getSegmentsAtPoint(currentSegmentation.id, {
              x: point.x,
              y: point.y,
              ...(activeSourceModel ? { source_model: activeSourceModel } : {}),
              states:
                mode === "reject"
                  ? ["CONFIRMED", "CANDIDATE", "INFERRED"]
                  : ["CANDIDATE", "INFERRED"],
            })
          );
          targetSegment = selectBestPointActionSegment(candidates, pointActionMode);
        }
        if (!targetSegment) {
          clearHoverInteraction();
          return;
        }

        applyOptimisticLabel(targetSegment.id, nextLabelState, targetSegment, {
          stageOverlay: true,
        });
        clearHoverInteraction();

        enqueueAnswer({
          segmentId: targetSegment.id,
          labelState: nextLabelState,
          fallbackMessage,
        });
      } catch (error) {
        // Reaching here means the object could not be found, not that the
        // answer failed to save: the queue owns that half, including the
        // rollback.
        if (targetSegment) {
          rollbackOptimisticLabel(targetSegment.id);
        }
        showErrorToast(fallbackMessage);
        console.error("Failed to apply point action:", error);
      }
    },
    [
      applyLabelOverrides,
      applyOptimisticLabel,
      activeSourceModel,
      clearHoverInteraction,
      currentSegmentation,
      enqueueAnswer,
      registerAnnotationActivity,
      resolveHoveredPointActionSegment,
      rollbackOptimisticLabel,
      showErrorToast,
    ]
  );

  return {
    handleResetConfirmedToCandidate,
    handleDeleteConfirmedObject,
    handleApplyPointAction,
    /** Send any coalesced answers now; awaited before anything that reads them back. */
    flushLabelAnswers: flushAnswers,
  };
}
