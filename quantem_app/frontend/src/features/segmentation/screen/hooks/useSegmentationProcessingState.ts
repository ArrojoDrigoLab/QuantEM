import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  activateSegmentationRoi,
  getSegmentationRois,
  rerunSegmentationRoi,
  updateSegmentationConfig,
  createSegmentationRoi,
} from "@/shared/api/segmentations/rois";
import { getJobQueueStatus } from "@/shared/api/jobs";
import {
  markSegmentationComplete,
  unlockSegmentation,
} from "@/shared/api/segmentations/annotations";
import { runFullSegmentation } from "@/shared/api/segmentations/overlays";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import {
  ORGANELLE_ACTION_JOB_TYPES,
  PROCESSING_BANNER_JOB_TYPES,
  ROI_JOB_TYPE,
  FULL_IMAGE_JOB_TYPE,
  STATUS_POLL_MS,
} from "@/features/segmentation/screen/utils/constants";
import { clamp } from "@/features/segmentation/screen/utils/bbox";
import type {
  ImageSegmentation,
  SegmentationInstanceParams,
} from "@/shared/types/images";
import type { JobQueueItem } from "@/shared/types/jobs";

interface UseSegmentationProcessingStateArgs {
  currentSegmentation: ImageSegmentation | null;
  activeSourceModel: string | null;
  supportsPointFeedback: boolean;
  supportsInstanceParams: boolean;
  currentInstanceParams: SegmentationInstanceParams | null;
  refetchSegmentations: () => Promise<void>;
  refreshSegmentViews: (options?: { deferOverlayRefresh?: boolean }) => Promise<void>;
}

export function useSegmentationProcessingState({
  currentSegmentation,
  activeSourceModel,
  supportsPointFeedback,
  supportsInstanceParams,
  currentInstanceParams,
  refetchSegmentations,
  refreshSegmentViews,
}: UseSegmentationProcessingStateArgs) {
  const [isApplyingFull, setIsApplyingFull] = useState(false);
  const [isRerunningRoi, setIsRerunningRoi] = useState(false);
  const [isSavingInstanceParams, setIsSavingInstanceParams] = useState(false);
  const [instanceParamsDraft, setInstanceParamsDraft] =
    useState<SegmentationInstanceParams | null>(currentInstanceParams);
  const instanceParamsSegmentationRef = useRef<string | null>(null);

  const {
    data: segmentationRois,
    loading: segmentationRoisLoading,
    refetch: refetchSegmentationRois,
  } = useApiQuery(
    () => {
      if (!currentSegmentation) return Promise.resolve([]);
      return getSegmentationRois(currentSegmentation.id);
    },
    [currentSegmentation?.id]
  );

  const activeRoi = useMemo(
    () => segmentationRois?.find((roi) => roi.is_active) ?? segmentationRois?.[0] ?? null,
    [segmentationRois]
  );

  const shouldShowProcessingStatus = Boolean(currentSegmentation && supportsPointFeedback);
  const { data: jobQueueStatus, refetch: refetchJobs } = useApiQuery(
    () => getJobQueueStatus(),
    [currentSegmentation?.id]
  );

  const queuePendingJobs: JobQueueItem[] = useMemo(() => {
    if (!jobQueueStatus) return [];
    return jobQueueStatus.queues.flatMap((queue) => queue.pending);
  }, [jobQueueStatus]);

  const allQueueJobs: JobQueueItem[] = useMemo(() => {
    if (!jobQueueStatus) return [];
    return [...jobQueueStatus.running, ...queuePendingJobs];
  }, [jobQueueStatus, queuePendingJobs]);

  const processingJobs: JobQueueItem[] = useMemo(() => {
    if (!currentSegmentation) return [];
    return allQueueJobs.filter(
      (job) =>
        job.segmentation?.id === currentSegmentation.id &&
        PROCESSING_BANNER_JOB_TYPES.has(job.type)
    );
  }, [allQueueJobs, currentSegmentation]);

  const activeFullImageJob = useMemo(() => {
    if (!currentSegmentation) return null;
    const matches = allQueueJobs.filter(
      (job) => job.segmentation?.id === currentSegmentation.id && job.type === FULL_IMAGE_JOB_TYPE
    );
    if (matches.length === 0) return null;
    return matches.find((job) => job.status === "RUNNING") ?? matches[0];
  }, [allQueueJobs, currentSegmentation]);

  const fullImageProgress = useMemo(() => {
    if (!activeFullImageJob) return null;
    if (activeFullImageJob.status === "RUNNING") {
      return Math.max(0, Math.min(100, Math.round(activeFullImageJob.progress)));
    }
    return null;
  }, [activeFullImageJob]);

  const fullImageActive = activeFullImageJob !== null;

  const activeRoiJob = useMemo(() => {
    if (!currentSegmentation) return null;
    const matches = allQueueJobs.filter(
      (job) => job.segmentation?.id === currentSegmentation.id && job.type === ROI_JOB_TYPE
    );
    if (matches.length === 0) return null;
    return matches.find((job) => job.status === "RUNNING") ?? matches[0];
  }, [allQueueJobs, currentSegmentation]);

  const hasQueuedOrRunningOrganelleTask = useMemo(() => {
    if (!currentSegmentation) return false;
    return allQueueJobs.some(
      (job) =>
        job.segmentation?.id === currentSegmentation.id &&
        ORGANELLE_ACTION_JOB_TYPES.has(job.type)
    );
  }, [allQueueJobs, currentSegmentation]);

  useEffect(() => {
    if (
      !shouldShowProcessingStatus &&
      !hasQueuedOrRunningOrganelleTask &&
      !isApplyingFull &&
      !isRerunningRoi
    ) {
      return undefined;
    }
    const interval = window.setInterval(() => {
      void refetchJobs();
    }, STATUS_POLL_MS);
    return () => clearInterval(interval);
  }, [
    hasQueuedOrRunningOrganelleTask,
    isApplyingFull,
    isRerunningRoi,
    refetchJobs,
    shouldShowProcessingStatus,
  ]);

  useEffect(() => {
    const segmentationId = currentSegmentation?.id ?? null;
    if (instanceParamsSegmentationRef.current === segmentationId) {
      return;
    }
    instanceParamsSegmentationRef.current = segmentationId;
    setInstanceParamsDraft(currentInstanceParams);
  }, [currentSegmentation?.id, currentInstanceParams]);

  const updateInstanceParam = useCallback(
    (
      key: keyof SegmentationInstanceParams,
      value: SegmentationInstanceParams[keyof SegmentationInstanceParams]
    ) => {
      let nextValue = value;
      if (
        (key === "segmentation_threshold" || key === "center_confidence_threshold") &&
        typeof nextValue === "number"
      ) {
        nextValue = clamp(nextValue, 0, 1);
      }
      if (
        (key === "center_min_distance" || key === "downsampling_factor") &&
        typeof nextValue === "number"
      ) {
        nextValue = Math.max(1, Math.round(nextValue));
      }
      setInstanceParamsDraft((prev) => (prev ? { ...prev, [key]: nextValue } : prev));
    },
    []
  );

  const handleSaveInstanceParams = useCallback(async () => {
    if (
      !currentSegmentation ||
      !supportsInstanceParams ||
      !instanceParamsDraft ||
      isSavingInstanceParams ||
      hasQueuedOrRunningOrganelleTask
    ) {
      return;
    }
    setIsSavingInstanceParams(true);
    try {
      const updated = await updateSegmentationConfig(currentSegmentation.id, {
        instance_params: instanceParamsDraft,
      });
      if (updated.instance_params) {
        setInstanceParamsDraft(updated.instance_params);
      }
      await refetchSegmentations();
    } catch (error) {
      console.error("Failed to save segmentation instance params:", error);
    } finally {
      setIsSavingInstanceParams(false);
    }
  }, [
    currentSegmentation,
    hasQueuedOrRunningOrganelleTask,
    instanceParamsDraft,
    isSavingInstanceParams,
    refetchSegmentations,
    supportsInstanceParams,
  ]);

  const handleRerunRoi = useCallback(async () => {
    if (!currentSegmentation || isRerunningRoi || hasQueuedOrRunningOrganelleTask) {
      return;
    }
    setIsRerunningRoi(true);
    try {
      await rerunSegmentationRoi(
        currentSegmentation.id,
        activeRoi?.id ?? undefined,
        activeSourceModel
      );
      await Promise.all([refetchJobs(), refetchSegmentations()]);
    } catch (error) {
      console.error("Failed to start ROI rerun:", error);
    } finally {
      setIsRerunningRoi(false);
    }
  }, [
    activeRoi,
    activeSourceModel,
    currentSegmentation,
    hasQueuedOrRunningOrganelleTask,
    isRerunningRoi,
    refetchJobs,
    refetchSegmentations,
  ]);

  const handleApplyFullImage = useCallback(async () => {
    if (
      !currentSegmentation ||
      isApplyingFull ||
      isRerunningRoi ||
      hasQueuedOrRunningOrganelleTask
    ) {
      return;
    }
    setIsApplyingFull(true);
    try {
      await runFullSegmentation(currentSegmentation.id, activeSourceModel);
      await Promise.all([refetchJobs(), refetchSegmentations()]);
    } catch (error) {
      console.error("Failed to start full segmentation run:", error);
    } finally {
      setIsApplyingFull(false);
    }
  }, [
    currentSegmentation,
    activeSourceModel,
    hasQueuedOrRunningOrganelleTask,
    isApplyingFull,
    isRerunningRoi,
    refetchJobs,
    refetchSegmentations,
  ]);

  /**
   * Lock or unlock, and hand any refusal back to the caller.
   *
   * The failure that matters is the `409` the complete endpoint returns when
   * `acknowledged_discard_count` no longer matches -- an inference run finished
   * while the confirmation was open. That refusal exists to protect the dialog,
   * so it has to reach the dialog: swallowing it into `console.error` (which is
   * what this did) would leave the user looking at a confirmation that did
   * nothing and said nothing.
   */
  const handleToggleSegmentationComplete = useCallback(
    async (options?: {
      discardUnconfirmed: boolean;
      acknowledgedDiscardCount: number;
    }) => {
      if (!currentSegmentation) return;
      if (currentSegmentation.status_stage === "COMPLETED") {
        await unlockSegmentation(currentSegmentation.id);
      } else {
        await markSegmentationComplete(currentSegmentation.id, options);
      }
      await refreshSegmentViews({ deferOverlayRefresh: true });
    },
    [currentSegmentation, refreshSegmentViews]
  );

  const prevStatusRef = useRef<string | null>(null);

  useEffect(() => {
    if (!isApplyingFull && !isRerunningRoi && !hasQueuedOrRunningOrganelleTask) {
      prevStatusRef.current = currentSegmentation?.status_stage ?? null;
      return undefined;
    }
    const interval = window.setInterval(() => {
      void refetchSegmentations();
    }, STATUS_POLL_MS);
    return () => clearInterval(interval);
  }, [
    currentSegmentation?.status_stage,
    hasQueuedOrRunningOrganelleTask,
    isApplyingFull,
    isRerunningRoi,
    refetchSegmentations,
  ]);

  useEffect(() => {
    if (!currentSegmentation) return;
    const stage = currentSegmentation.status_stage;
    const prev = prevStatusRef.current;
    prevStatusRef.current = stage;
    if (
      (prev === "RUNNING_INFERENCE" ||
        prev === "EXTRACTING_CANDIDATES" ||
        prev === "UPDATING") &&
      stage === "CANDIDATES_READY"
    ) {
      void refreshSegmentViews();
    }
  }, [currentSegmentation, refreshSegmentViews]);

  return {
    segmentationRois,
    segmentationRoisLoading,
    refetchSegmentationRois,
    activeRoi,
    shouldShowProcessingStatus,
    processingJobs,
    fullImageProgress,
    fullImageActive,
    activeRoiJob,
    hasQueuedOrRunningOrganelleTask,
    isApplyingFull,
    isRerunningRoi,
    isSavingInstanceParams,
    instanceParamsDraft,
    setInstanceParamsDraft,
    updateInstanceParam,
    handleSaveInstanceParams,
    handleRerunRoi,
    handleApplyFullImage,
    handleToggleSegmentationComplete,
    activateSegmentationRoi,
    createSegmentationRoi,
    refetchJobs,
  };
}
