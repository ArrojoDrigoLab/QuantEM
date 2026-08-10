import { useCallback } from "react";
import {
  getSegmentsAtPoint,
  updateSegmentLabelsBatch,
} from "@/shared/api/segmentations/annotations";
import { extractApiErrorMessage } from "@/utils/apiErrors";
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
  clearHoverInteraction,
  registerAnnotationActivity,
  stageOptimisticRevisionTargets,
  getOptimisticTargetRevision,
  handleOverlayMutationRefresh,
  showErrorToast,
}: UseReviewPointActionsArgs) {
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
      let targetSegment: SegmentObject | null = null;
      try {
        const matches = await getSegmentsAtPoint(currentSegmentation.id, {
          x: point.x,
          y: point.y,
          states: ["CONFIRMED"],
          ...(activeSourceModel ? { source_model: activeSourceModel } : {}),
        });
        const segment = matches.find((item) => item.label_state === "CONFIRMED");
        if (!segment) return;
        targetSegment = segment;

        applyOptimisticLabel(segment.id, "CANDIDATE", segment);
        const response = await updateSegmentLabelsBatch({
          labels: [{ id: segment.id, label_state: "CANDIDATE" }],
          source_model: activeSourceModel,
        });
        stageOptimisticRevisionTargets(
          [segment.id],
          getOptimisticTargetRevision(response.overlays?.[currentSegmentation.id])
        );
        handleOverlayMutationRefresh(response.overlays?.[currentSegmentation.id]);
        clearHoverInteraction();
      } catch (error) {
        if (targetSegment) {
          rollbackOptimisticLabel(targetSegment.id);
        }
        console.error("Failed to reset confirmed segment to candidate:", error);
      }
    },
    [
      applyOptimisticLabel,
      activeSourceModel,
      clearHoverInteraction,
      currentSegmentation,
      getOptimisticTargetRevision,
      handleOverlayMutationRefresh,
      registerAnnotationActivity,
      rollbackOptimisticLabel,
      stageOptimisticRevisionTargets,
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

        const response = await updateSegmentLabelsBatch({
          labels: [{ id: targetSegment.id, label_state: nextLabelState }],
          source_model: activeSourceModel,
        });
        stageOptimisticRevisionTargets(
          [targetSegment.id],
          getOptimisticTargetRevision(response.overlays?.[currentSegmentation.id])
        );
        handleOverlayMutationRefresh(response.overlays?.[currentSegmentation.id]);
      } catch (error) {
        if (targetSegment) {
          rollbackOptimisticLabel(targetSegment.id);
        }
        showErrorToast(extractApiErrorMessage(error, fallbackMessage));
        console.error("Failed to apply point action:", error);
      }
    },
    [
      applyLabelOverrides,
      applyOptimisticLabel,
      activeSourceModel,
      clearHoverInteraction,
      currentSegmentation,
      getOptimisticTargetRevision,
      handleOverlayMutationRefresh,
      registerAnnotationActivity,
      resolveHoveredPointActionSegment,
      rollbackOptimisticLabel,
      showErrorToast,
      stageOptimisticRevisionTargets,
    ]
  );

  return {
    handleResetConfirmedToCandidate,
    handleApplyPointAction,
  };
}
