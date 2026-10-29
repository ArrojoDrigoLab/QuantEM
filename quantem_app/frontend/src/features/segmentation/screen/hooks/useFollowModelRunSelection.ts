import { useEffect, useRef } from "react";
import type { SegmentationModelRunSelection } from "@/features/segmentation/screen/hooks/useSegmentationProcessingState";

interface FollowedRun {
  segmentationId: string | null;
  jobId: string | null;
  observedActive: boolean;
}

interface UseFollowModelRunSelectionArgs {
  segmentationId: string | null;
  run: SegmentationModelRunSelection | null;
  onBaseModelSelected: (sourceModel: string) => void;
  onAdaptedModelSelected: (adapterId: string, sourceModel: string) => void;
}

/** Follow a newly queued run once. Completion must not replace a later choice. */
export function useFollowModelRunSelection({
  segmentationId,
  run,
  onBaseModelSelected,
  onAdaptedModelSelected,
}: UseFollowModelRunSelectionArgs): void {
  const followedRef = useRef<FollowedRun>({
    segmentationId: null,
    jobId: null,
    observedActive: false,
  });

  useEffect(() => {
    if (followedRef.current.segmentationId !== segmentationId) {
      followedRef.current = {
        segmentationId,
        jobId: null,
        observedActive: false,
      };
    }
    if (!segmentationId || !run) return;

    if (followedRef.current.jobId !== run.jobId) {
      followedRef.current = {
        segmentationId,
        jobId: run.jobId,
        observedActive: false,
      };
    }

    const active = run.status === "PENDING" || run.status === "RETRY" || run.status === "RUNNING";
    const firstActiveObservation = active && !followedRef.current.observedActive;
    if (!firstActiveObservation) return;
    followedRef.current.observedActive = true;

    if (run.adapterId) {
      onAdaptedModelSelected(run.adapterId, run.sourceModel);
    } else {
      onBaseModelSelected(run.sourceModel);
    }
  }, [
    onAdaptedModelSelected,
    onBaseModelSelected,
    run,
    segmentationId,
  ]);
}
