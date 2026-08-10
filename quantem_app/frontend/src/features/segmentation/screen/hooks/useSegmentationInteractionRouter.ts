import { useCallback, useEffect } from "react";
import { LABELING_LEFT_PANEL_STATES } from "@/features/segmentation/screen/utils/constants";
import type { LeftMode } from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
import type { Point } from "@/utils/geometry";
import type { SegmentObject } from "@/shared/types";
import type { HoverSegmentQuery } from "@/features/segmentation/screen/hooks/useSegmentationHoverQuery";

interface UseSegmentationInteractionRouterArgs {
  currentSegmentationId: string | null;
  leftNavigateMode: boolean;
  /** ER ROI placement is active -- the next image click places the ROI. */
  roiPlacementActive: boolean;
  isPointInsideImageBounds: (point: Point) => boolean;
  applyLabelOverrides: (items: SegmentObject[]) => SegmentObject[];
  scheduleHoverSegmentQuery: (
    point: Point,
    query: HoverSegmentQuery,
    resolveSegments: (segments: SegmentObject[]) => SegmentObject[],
    errorMessage: string
  ) => void;
  clearHoverInteraction: () => void;
  onRoiPlacementClick: (point: Point) => void;
  completedRoi: {
    isActive: boolean;
    handlePolygonClick: (point: Point) => void;
    handlePolygonMouseMove: (point: Point) => void;
  };
  erPolygon: {
    isActive: boolean;
    handlePolygonClick: (point: Point) => void;
    handlePolygonMouseMove: (point: Point) => void;
  };
  tissue: {
    enabled: boolean;
    polygon: {
      isActive: boolean;
      handlePolygonClick: (point: Point) => void;
      handlePolygonMouseMove: (point: Point) => void;
    };
  };
  review: {
    hoverActionMode: "confirm" | "reject" | "group-confirm" | "group-reject" | "test";
    leftMode: LeftMode;
    workflowMode: "annotate" | "review" | "uncertain";
    isGroupActionMode: boolean;
    group: {
      handleImagePress: (imagePoint: Point, screenPoint: Point) => void;
      handleImageDrag: (imagePoint: Point, screenPoint: Point) => void;
      handleImageRelease: (imagePoint: Point, screenPoint: Point) => void;
    };
    pointActions: {
      handleApply: (point: Point, mode: "confirm" | "reject") => Promise<void>;
    };
  };
}

export function useSegmentationInteractionRouter({
  currentSegmentationId,
  leftNavigateMode,
  roiPlacementActive,
  isPointInsideImageBounds,
  applyLabelOverrides,
  scheduleHoverSegmentQuery,
  clearHoverInteraction,
  onRoiPlacementClick,
  completedRoi,
  erPolygon,
  tissue,
  review,
}: UseSegmentationInteractionRouterArgs) {
  const onLeftClick = useCallback(
    (point: Point) => {
      if (!currentSegmentationId) return;

      // ROI placement is exclusive and wins over navigate/pan: a placement click
      // must always land, regardless of the underlying tool or navigate mode.
      if (roiPlacementActive) {
        if (!isPointInsideImageBounds(point)) return;
        onRoiPlacementClick(point);
        return;
      }

      if (leftNavigateMode) return;

      if (erPolygon.isActive) {
        if (!isPointInsideImageBounds(point)) return;
        erPolygon.handlePolygonClick(point);
        return;
      }

      if (completedRoi.isActive) {
        if (!isPointInsideImageBounds(point)) return;
        completedRoi.handlePolygonClick(point);
        return;
      }

      if (tissue.enabled) {
        // A tissue polygon/exclude click places a vertex; the brush paints via
        // the viewer stroke path, so a plain click has no other tissue meaning.
        if (tissue.polygon.isActive && isPointInsideImageBounds(point)) {
          tissue.polygon.handlePolygonClick(point);
        }
        return;
      }

      if (review.leftMode === "annotate") return;

      if (review.leftMode === "hover") {
        if (
          review.hoverActionMode === "group-confirm" ||
          review.hoverActionMode === "group-reject"
        ) {
          return;
        }
        if (review.hoverActionMode === "test") {
          return;
        }
        void review.pointActions.handleApply(point, review.hoverActionMode);
      }
    },
    [
      currentSegmentationId,
      isPointInsideImageBounds,
      leftNavigateMode,
      onRoiPlacementClick,
      review,
      roiPlacementActive,
      completedRoi,
      erPolygon,
      tissue,
    ]
  );

  const onLeftImagePress = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (leftNavigateMode || roiPlacementActive) return;
      if (completedRoi.isActive || erPolygon.isActive || tissue.enabled) return;
      if (!review.isGroupActionMode) return;
      review.group.handleImagePress(imagePoint, screenPoint);
    },
    [
      leftNavigateMode,
      review,
      roiPlacementActive,
      completedRoi,
      erPolygon,
      tissue,
    ]
  );

  const onLeftImageDrag = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (leftNavigateMode || roiPlacementActive) return;
      if (completedRoi.isActive || erPolygon.isActive || tissue.enabled) return;
      if (!review.isGroupActionMode) return;
      review.group.handleImageDrag(imagePoint, screenPoint);
    },
    [
      leftNavigateMode,
      review,
      roiPlacementActive,
      completedRoi,
      erPolygon,
      tissue,
    ]
  );

  const onLeftImageRelease = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (leftNavigateMode || roiPlacementActive) return;
      if (completedRoi.isActive || erPolygon.isActive || tissue.enabled) return;
      if (!review.isGroupActionMode) return;
      review.group.handleImageRelease(imagePoint, screenPoint);
    },
    [
      leftNavigateMode,
      review,
      roiPlacementActive,
      completedRoi,
      erPolygon,
      tissue,
    ]
  );

  const onLeftMouseMove = useCallback(
    (point: Point) => {
      if (!currentSegmentationId) return;

      if (completedRoi.isActive) {
        if (leftNavigateMode) return;
        if (!isPointInsideImageBounds(point)) {
          clearHoverInteraction();
          return;
        }
        clearHoverInteraction();
        completedRoi.handlePolygonMouseMove(point);
        return;
      }

      if (erPolygon.isActive) {
        if (leftNavigateMode) return;
        if (!isPointInsideImageBounds(point)) {
          clearHoverInteraction();
          return;
        }
        clearHoverInteraction();
        erPolygon.handlePolygonMouseMove(point);
        return;
      }

      if (tissue.enabled) {
        if (leftNavigateMode) return;
        clearHoverInteraction();
        if (tissue.polygon.isActive && isPointInsideImageBounds(point)) {
          tissue.polygon.handlePolygonMouseMove(point);
        }
        return;
      }

      if (
        leftNavigateMode ||
        review.leftMode !== "hover" ||
        review.workflowMode === "annotate" ||
        review.isGroupActionMode
      ) {
        return;
      }
      if (!isPointInsideImageBounds(point)) {
        clearHoverInteraction();
        return;
      }
      scheduleHoverSegmentQuery(
        point,
        ["CONFIRMED", "CANDIDATE", "INFERRED"],
        (result) =>
          applyLabelOverrides(result).filter((segment) =>
            LABELING_LEFT_PANEL_STATES.has(segment.label_state)
          ),
        "Failed to query hover segments:"
      );
    },
    [
      applyLabelOverrides,
      clearHoverInteraction,
      currentSegmentationId,
      isPointInsideImageBounds,
      leftNavigateMode,
      review,
      scheduleHoverSegmentQuery,
      completedRoi,
      erPolygon,
      tissue,
    ]
  );

  const onLeftMouseLeave = useCallback(() => {
    clearHoverInteraction();
  }, [clearHoverInteraction]);

  useEffect(() => {
    if (
      completedRoi.isActive ||
      erPolygon.isActive ||
      tissue.enabled ||
      leftNavigateMode ||
      review.leftMode !== "hover" ||
      review.isGroupActionMode
    ) {
      clearHoverInteraction();
    }
  }, [
    clearHoverInteraction,
    completedRoi,
    erPolygon,
    tissue,
    leftNavigateMode,
    review,
  ]);

  return {
    onLeftClick,
    onLeftImagePress,
    onLeftImageDrag,
    onLeftImageRelease,
    onLeftMouseMove,
    onLeftMouseLeave,
  };
}
