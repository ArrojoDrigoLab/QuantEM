import { useCallback, useEffect } from "react";
import { LABELING_LEFT_PANEL_STATES } from "@/features/segmentation/screen/utils/constants";
import type {
  LeftMode,
  WorkflowMode,
} from "@/features/segmentation/hooks/useSegmentationWorkflowMode";
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
  onRoiEditPress: (point: Point) => void;
  onRoiEditDrag: (point: Point) => void;
  onRoiEditRelease: (point: Point) => void;
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
  /**
   * Box-to-object (`features/sam`). A drag tool, so it takes press/drag/release
   * and nothing else; it has no click meaning and no hover meaning.
   */
  samBox?: {
    isActive: boolean;
    handleImagePress: (imagePoint: Point, screenPoint: Point) => void;
    handleImageDrag: (imagePoint: Point, screenPoint: Point) => void;
    handleImageRelease: (imagePoint: Point, screenPoint: Point) => void;
  };
  review: {
    hoverActionMode: "confirm" | "reject" | "group-confirm" | "group-reject" | "test";
    leftMode: LeftMode;
    workflowMode: WorkflowMode;
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
  onRoiEditPress,
  onRoiEditDrag,
  onRoiEditRelease,
  completedRoi,
  erPolygon,
  tissue,
  samBox,
  review,
}: UseSegmentationInteractionRouterArgs) {
  const onLeftClick = useCallback(
    (point: Point) => {
      if (!currentSegmentationId) return;
      if (leftNavigateMode) return;

      if (roiPlacementActive) {
        if (!isPointInsideImageBounds(point)) return;
        onRoiPlacementClick(point);
        return;
      }

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
      if (leftNavigateMode) return;
      if (roiPlacementActive) {
        onRoiEditPress(imagePoint);
        return;
      }
      if (completedRoi.isActive || erPolygon.isActive || tissue.enabled) return;
      // Before the group-selection check: both are box drags, and whichever is
      // switched on owns the gesture.
      if (samBox?.isActive) {
        samBox.handleImagePress(imagePoint, screenPoint);
        return;
      }
      if (!review.isGroupActionMode) return;
      review.group.handleImagePress(imagePoint, screenPoint);
    },
    [
      leftNavigateMode,
      onRoiEditPress,
      review,
      roiPlacementActive,
      completedRoi,
      erPolygon,
      tissue,
      samBox,
    ]
  );

  const onLeftImageDrag = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (leftNavigateMode) return;
      if (roiPlacementActive) {
        onRoiEditDrag(imagePoint);
        return;
      }
      if (completedRoi.isActive || erPolygon.isActive || tissue.enabled) return;
      if (samBox?.isActive) {
        samBox.handleImageDrag(imagePoint, screenPoint);
        return;
      }
      if (!review.isGroupActionMode) return;
      review.group.handleImageDrag(imagePoint, screenPoint);
    },
    [
      leftNavigateMode,
      onRoiEditDrag,
      review,
      roiPlacementActive,
      completedRoi,
      erPolygon,
      tissue,
      samBox,
    ]
  );

  const onLeftImageRelease = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (leftNavigateMode) return;
      if (roiPlacementActive) {
        onRoiEditRelease(imagePoint);
        return;
      }
      if (completedRoi.isActive || erPolygon.isActive || tissue.enabled) return;
      if (samBox?.isActive) {
        samBox.handleImageRelease(imagePoint, screenPoint);
        return;
      }
      if (!review.isGroupActionMode) return;
      review.group.handleImageRelease(imagePoint, screenPoint);
    },
    [
      leftNavigateMode,
      onRoiEditRelease,
      review,
      roiPlacementActive,
      completedRoi,
      erPolygon,
      tissue,
      samBox,
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
