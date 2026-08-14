import { useEffect, useRef } from "react";
import type { SegmentationModelRunSelection } from "@/features/segmentation/screen/hooks/useSegmentationProcessingState";

interface FollowedRun {
  segmentationId: string | null;
  jobId: string | null;
  observedActive: boolean;
  completionHandled: boolean;
}

interface UseFollowModelRunSelectionArgs {
  segmentationId: string | null;
  run: SegmentationModelRunSelection | null;
  onBaseModelSelected: (sourceModel: string) => void;
  onAdaptedModelSelected: (adapterId: string, sourceModel: string) => void;
}

/**
 * Keep the model picker attached to the inference result being produced.
 *
 * Selection happens when a queued/running job is first observed. The same job
 * selects itself once more on success, so an intervening stale/default Manual
 * selection cannot hide the overlay at the moment it becomes available.
 */
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
    completionHandled: false,
  });

  useEffect(() => {
    if (followedRef.current.segmentationId !== segmentationId) {
      followedRef.current = {
        segmentationId,
        jobId: null,
        observedActive: false,
        completionHandled: false,
      };
    }
    if (!segmentationId || !run) return;

    if (followedRef.current.jobId !== run.jobId) {
      followedRef.current = {
        segmentationId,
        jobId: run.jobId,
        observedActive: false,
        completionHandled: false,
      };
    }

    const active = run.status === "PENDING" || run.status === "RETRY" || run.status === "RUNNING";
    const firstActiveObservation = active && !followedRef.current.observedActive;
    const completedAfterObservation =
      run.status === "SUCCESS" &&
      followedRef.current.observedActive &&
      !followedRef.current.completionHandled;
    if (!firstActiveObservation && !completedAfterObservation) return;

    if (firstActiveObservation) followedRef.current.observedActive = true;
    if (completedAfterObservation) followedRef.current.completionHandled = true;

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
