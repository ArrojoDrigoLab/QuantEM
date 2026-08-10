import { useMemo } from "react";
import { getUncertainSegments } from "@/shared/api/segmentations/annotations";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import type { ImageSegmentation } from "@/shared/types/images";
import type { SegmentObject } from "@/shared/types/segmentation";
import type { WorkflowMode } from "@/features/segmentation/hooks/useSegmentationWorkflowMode";

export function useUncertainSegments(
  currentSegmentation: ImageSegmentation | null,
  workflowMode: WorkflowMode,
  limit: number,
  sourceModel?: string | null
) {
  const { data, refetch } = useApiQuery<SegmentObject[]>(
    () => {
      if (!currentSegmentation || workflowMode !== "uncertain") {
        return Promise.resolve([]);
      }
      return getUncertainSegments(currentSegmentation.id, limit, sourceModel);
    },
    [currentSegmentation?.id, workflowMode, limit, sourceModel]
  );

  const uncertainSegments = useMemo(() => data ?? [], [data]);

  return {
    uncertainSegments,
    refetchUncertainSegments: refetch,
  };
}
