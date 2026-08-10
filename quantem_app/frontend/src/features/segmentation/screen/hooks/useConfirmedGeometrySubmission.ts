import { useCallback } from "react";
import { confirmSegmentsBatch } from "@/shared/api/segmentations/annotations";
import { buildSyntheticSegmentsFromGeometries } from "@/features/segmentation/screen/utils/optimisticSegments";
import type {
  SegmentationOverlayMutationState,
  SegmentObject,
} from "@/shared/types/segmentation";
import type { ImageSegmentation } from "@/shared/types/images";

interface UseConfirmedGeometrySubmissionArgs {
  currentSegmentation: ImageSegmentation | null;
  registerAnnotationActivity: () => void;
  stageOptimisticSegments: (
    items: SegmentObject[],
    targetRevision?: number | null
  ) => void;
  clearOptimisticSegments: (segmentIds: string[]) => void;
  stageOptimisticRevisionTargets: (segmentIds: string[], targetRevision?: number | null) => void;
  getOptimisticTargetRevision: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => number | null;
  handleOverlayMutationRefresh: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => void;
}

export function useConfirmedGeometrySubmission({
  currentSegmentation,
  registerAnnotationActivity,
  stageOptimisticSegments,
  clearOptimisticSegments,
  stageOptimisticRevisionTargets,
  getOptimisticTargetRevision,
  handleOverlayMutationRefresh,
}: UseConfirmedGeometrySubmissionArgs) {
  const submitConfirmedGeometriesOptimistically = useCallback(
    async ({
      geometries,
      samScores,
      mergeOverlaps = false,
      manualCreation = true,
    }: {
      geometries: Array<Array<[number, number]>>;
      samScores?: Array<number | null | undefined>;
      mergeOverlaps?: boolean;
      manualCreation?: boolean;
    }) => {
      if (!currentSegmentation || geometries.length === 0) {
        return null;
      }

      const requestSegmentationId = currentSegmentation.id;
      registerAnnotationActivity();
      const optimisticCreatedSegments = buildSyntheticSegmentsFromGeometries(
        requestSegmentationId,
        geometries,
        "CONFIRMED"
      );
      const optimisticIds = optimisticCreatedSegments.map((segment) => segment.id);
      stageOptimisticSegments(optimisticCreatedSegments);

      try {
        const response = await confirmSegmentsBatch(requestSegmentationId, {
          segments: geometries.map((geometryCoords, index) => {
            const samScore = samScores?.[index];
            return {
              geometry_coords: geometryCoords,
              ...(typeof samScore === "number" ? { sam_score: samScore } : {}),
            };
          }),
          merge_overlaps: mergeOverlaps,
          manual_creation: manualCreation,
        });
        if (currentSegmentation?.id !== requestSegmentationId) {
          return null;
        }
        stageOptimisticRevisionTargets(
          optimisticIds,
          getOptimisticTargetRevision(response.overlay)
        );
        handleOverlayMutationRefresh(response.overlay);
        return response;
      } catch (error) {
        if (currentSegmentation?.id === requestSegmentationId) {
          clearOptimisticSegments(optimisticIds);
        }
        throw error;
      }
    },
    [
      clearOptimisticSegments,
      currentSegmentation,
      getOptimisticTargetRevision,
      handleOverlayMutationRefresh,
      registerAnnotationActivity,
      stageOptimisticRevisionTargets,
      stageOptimisticSegments,
    ]
  );

  return {
    submitConfirmedGeometriesOptimistically,
  };
}
