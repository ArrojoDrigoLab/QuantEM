import { useEffect, useMemo } from "react";
import { listUserFeedback } from "@/shared/api/segmentations/annotations";
import { useUserFeedbackStore } from "@/shared/stores/useUserFeedbackStore";
import {
  FEEDBACK_PENDING_STATUSES,
  FEEDBACK_POLL_MS,
} from "@/features/segmentation/screen/utils/constants";
import type { UserFeedback } from "@/shared/types/segmentation";

interface UseSegmentationFeedbackArgs {
  currentSegmentationId: string | null;
  supportsPointFeedback: boolean;
  refreshSegmentViews: (options?: { deferOverlayRefresh?: boolean }) => Promise<void>;
}

export function useSegmentationFeedback({
  currentSegmentationId,
  supportsPointFeedback,
  refreshSegmentViews,
}: UseSegmentationFeedbackArgs) {
  const feedbackBySegmentation = useUserFeedbackStore(
    (state) => state.feedbackBySegmentation
  );
  const upsertFeedbackBatch = useUserFeedbackStore(
    (state) => state.upsertFeedbackBatch
  );

  const feedbackLog = useMemo<UserFeedback[]>(() => {
    if (!currentSegmentationId) return [];
    return Object.values(feedbackBySegmentation[currentSegmentationId] ?? {});
  }, [currentSegmentationId, feedbackBySegmentation]);

  const pendingFeedbackIds = useMemo(
    () =>
      feedbackLog
        .filter((feedback) =>
          FEEDBACK_PENDING_STATUSES.includes(feedback.utilized_status)
        )
        .map((feedback) => feedback.id),
    [feedbackLog]
  );

  useEffect(() => {
    if (!currentSegmentationId || !supportsPointFeedback) return;
    let cancelled = false;
    void (async () => {
      try {
        const feedbackItems = await listUserFeedback(currentSegmentationId);
        if (!cancelled) {
          upsertFeedbackBatch(currentSegmentationId, feedbackItems);
        }
      } catch (error) {
        console.error("Failed to load user feedback log:", error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [currentSegmentationId, supportsPointFeedback, upsertFeedbackBatch]);

  useEffect(() => {
    if (!currentSegmentationId || pendingFeedbackIds.length === 0) return undefined;
    let cancelled = false;
    let pollInFlight = false;

    const poll = async () => {
      if (cancelled || pollInFlight) return;
      pollInFlight = true;
      try {
        const previousById = new Map(
          feedbackLog.map((feedback) => [feedback.id, feedback.utilized_status])
        );
        const updates = await listUserFeedback(currentSegmentationId, {
          ids: pendingFeedbackIds,
        });
        upsertFeedbackBatch(currentSegmentationId, updates);

        const hasNewSuccess = updates.some((feedback) => {
          const previous = previousById.get(feedback.id);
          return feedback.utilized_status === "SUCCESS" && previous !== "SUCCESS";
        });
        if (hasNewSuccess) {
          await refreshSegmentViews({ deferOverlayRefresh: true });
        }
      } catch (error) {
        console.error("Failed to poll user feedback statuses:", error);
      } finally {
        pollInFlight = false;
      }
    };

    void poll();
    const interval = window.setInterval(() => {
      void poll();
    }, FEEDBACK_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [
    currentSegmentationId,
    feedbackLog,
    pendingFeedbackIds,
    refreshSegmentViews,
    upsertFeedbackBatch,
  ]);

  return {
    feedbackLog,
    pendingFeedbackIds,
  };
}
