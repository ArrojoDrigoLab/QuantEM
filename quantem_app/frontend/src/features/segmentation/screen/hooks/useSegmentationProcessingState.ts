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
import { ensureModelInstalled } from "@/features/models/ensureModelInstalled";
import { packIdForSourceModel } from "@/features/models/runnable";
import {
  MODEL_DOWNLOAD_JOB_TYPE,
  ORGANELLE_ACTION_JOB_TYPES,
  PROCESSING_BANNER_JOB_TYPES,
  ROI_JOB_TYPE,
  FULL_IMAGE_JOB_TYPE,
  STATUS_POLL_MS,
} from "@/features/segmentation/screen/utils/constants";
import { clamp } from "@/features/segmentation/screen/utils/bbox";
import { isStoppedRunJob } from "@/shared/progress/runProgress";
import type {
  ImageSegmentation,
  SegmentationInstanceParams,
} from "@/shared/types/images";
import type { JobQueueItem } from "@/shared/types/jobs";

/**
 * How long a run that stopped stays on the run panel after it concludes.
 *
 * The panel used to empty itself within one poll of a cancellation -- measured
 * at 1 Hz by the wave-0c verifier, non-empty at t=19.87 s with "11 of 56 tiles"
 * and empty at t=20.90 s -- so the one number a user wants after pressing
 * Cancel ("how far did it get?") disappeared at the moment they asked for it.
 * It has to outlive the run, and it has to stop being news eventually: five
 * minutes is long enough to walk back to the screen and short enough that a
 * stale row is never mistaken for something happening now. The heading changes
 * to "Last run" as soon as nothing is live, so the row is never presented as
 * work in flight.
 */
const STOPPED_RUN_LINGER_MS = 5 * 60 * 1000;

interface UseSegmentationProcessingStateArgs {
  currentSegmentation: ImageSegmentation | null;
  activeSourceModel: string | null;
  supportsPointFeedback: boolean;
  supportsInstanceParams: boolean;
  currentInstanceParams: SegmentationInstanceParams | null;
  refetchSegmentations: () => Promise<void>;
  refreshSegmentViews: (options?: { deferOverlayRefresh?: boolean }) => Promise<void>;
  onModelInstalled?: () => void | Promise<void>;
  onRunError?: (message: string) => void;
}

export function useSegmentationProcessingState({
  currentSegmentation,
  activeSourceModel,
  supportsPointFeedback,
  supportsInstanceParams,
  currentInstanceParams,
  refetchSegmentations,
  refreshSegmentViews,
  onModelInstalled,
  onRunError,
}: UseSegmentationProcessingStateArgs) {
  const [isApplyingFull, setIsApplyingFull] = useState(false);
  const [rerunningRoiId, setRerunningRoiId] = useState<string | null>(null);
  const [previewRoiId, setPreviewRoiId] = useState<string | null>(null);
  const isRerunningRoi = rerunningRoiId !== null;
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

  useEffect(() => {
    setPreviewRoiId(null);
  }, [activeSourceModel, currentSegmentation?.id]);

  const { data: jobQueueSnapshot, refetch: refetchJobs } = useApiQuery(
    async () => ({ status: await getJobQueueStatus(), receivedAt: Date.now() }),
    [currentSegmentation?.id]
  );
  const jobQueueStatus = jobQueueSnapshot?.status ?? null;

  const queuePendingJobs: JobQueueItem[] = useMemo(() => {
    if (!jobQueueStatus) return [];
    return jobQueueStatus.queues.flatMap((queue) => queue.pending);
  }, [jobQueueStatus]);

  const allQueueJobs: JobQueueItem[] = useMemo(() => {
    if (!jobQueueStatus) return [];
    return [...jobQueueStatus.running, ...queuePendingJobs];
  }, [jobQueueStatus, queuePendingJobs]);

  /**
   * Everything worth watching while this image is being segmented.
   *
   * Widened from "runs on the segmentation currently open" to "runs on this
   * image, plus any model coming down the wire", because the owner asked for
   * three indicators and two of them are not about the open segmentation:
   *
   * * the **aggregate** is across every organelle for the image, so it needs
   *   the sibling runs -- with only the open one, "Everything" would equal the
   *   line beneath it and the second organelle would be invisible until the
   *   user switched to it;
   * * a **model download** belongs to no segmentation at all. It is what the
   *   run is waiting for, and it is shown as its own kind of row so it can
   *   never be read as segmentation progress.
   */
  const processingJobs: JobQueueItem[] = useMemo(() => {
    if (!currentSegmentation) return [];
    const imageId = currentSegmentation.asset ?? null;
    const openWaves = new Set(
      allQueueJobs.map((job) => job.batch_id).filter(Boolean) as string[]
    );
    // Two reasons a concluded run stays on the panel.
    //
    // Its wave is still going: mitochondria completing while nucleus still has
    // 88 tiles to walk is the ordinary case, and a row that disappears reads as
    // work lost, not work done.
    //
    // Or it stopped -- cancelled or failed -- recently, whether or not anything
    // else in its wave is still open. This is the case the wave-0c verifier
    // caught: cancel the only run in a wave and the wave closes with it, so the
    // gate above dropped the row one poll later and the tile count the user was
    // watching went with it. `organelleRow` has always had the copy for this
    // ("stopped at 18 of 56 tiles · you stopped this one"); nothing could reach
    // it.
    const now = jobQueueSnapshot?.receivedAt ?? 0;
    const stillWorthShowing = [
      ...(jobQueueStatus?.completed ?? []),
      ...(jobQueueStatus?.failed ?? []),
    ].filter((job) => {
      if (!PROCESSING_BANNER_JOB_TYPES.has(job.type)) return false;
      if (job.batch_id && openWaves.has(job.batch_id)) return true;
      if (!isStoppedRunJob(job)) return false;
      // Server clock, browser clock. A skew makes the row linger a little
      // longer or a little less; it cannot make it wrong, because the row
      // states its own outcome rather than implying freshness.
      const finished = job.finished_at ? Date.parse(job.finished_at) : NaN;
      if (!Number.isFinite(finished)) return false;
      return now - finished <= STOPPED_RUN_LINGER_MS;
    });
    return [...allQueueJobs, ...stillWorthShowing]
      .filter((job) => {
        if (job.type === MODEL_DOWNLOAD_JOB_TYPE) {
          return job.status === "RUNNING" || job.status === "PENDING";
        }
        if (!PROCESSING_BANNER_JOB_TYPES.has(job.type)) return false;
        if (job.segmentation?.id === currentSegmentation.id) return true;
        return Boolean(imageId && job.image?.id === imageId);
      })
      // Enqueue order, so the rows do not reshuffle as runs start and finish.
      .sort((a, b) => a.created_at.localeCompare(b.created_at));
  }, [allQueueJobs, currentSegmentation, jobQueueSnapshot, jobQueueStatus]);

  /**
   * Whether the run panel is worth a strip of the screen.
   *
   * It used to be "this segmentation type takes point feedback", which is a
   * fact about the model and not about whether anything is happening: on a type
   * outside that set, a run could be walking 858 tiles with nothing on screen
   * to say so. Now it is the honest question -- is there work in flight for
   * this image.
   */
  const shouldShowProcessingStatus = Boolean(
    currentSegmentation && (supportsPointFeedback || processingJobs.length > 0)
  );

  const activeFullImageJob = useMemo(() => {
    if (!currentSegmentation) return null;
    const matches = allQueueJobs.filter(
      (job) => job.segmentation?.id === currentSegmentation.id && job.type === FULL_IMAGE_JOB_TYPE
    );
    if (matches.length === 0) return null;
    return matches.find((job) => job.status === "RUNNING") ?? matches[0];
  }, [allQueueJobs, currentSegmentation]);

  /**
   * The percentage on the run button, on the tiling plan's divisor.
   *
   * `job.progress` divides by the whole job -- the model load, the tiles,
   * finding objects, saving them -- so during the tiles it reads a point or two
   * below the tile fraction the panel above the button is showing for the same
   * run. Two numbers for one thing, disagreeing, is the defect. While tiles
   * remain, this is the tile fraction; once they are walked there is no tile
   * fraction left to quote and the whole-job number takes over, which is also
   * the moment it stops being the smaller of the two.
   */
  const fullImageProgress = useMemo(() => {
    if (!activeFullImageJob) return null;
    if (activeFullImageJob.status !== "RUNNING") return null;
    // While the model loads there is no fraction of the work done, and the
    // header's fallback for null is the word "Starting". A number here would be
    // the frozen 5% again, wearing a smaller number.
    if (activeFullImageJob.progress_stage === "loading_model") return null;
    const units = activeFullImageJob.unit_progress;
    const value =
      units && units.total > 0 && units.done < units.total && units.percent !== null
        ? units.percent
        : activeFullImageJob.progress;
    return Math.max(0, Math.min(100, Math.round(value)));
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

  const handleRerunRoi = useCallback(async (roiId?: string) => {
    const targetRoi = roiId
      ? segmentationRois?.find((roi) => roi.id === roiId) ?? null
      : activeRoi;
    if (
      !currentSegmentation ||
      !targetRoi ||
      targetRoi.completed_for_segmentation === true ||
      isRerunningRoi ||
      hasQueuedOrRunningOrganelleTask
    ) {
      return;
    }
    setRerunningRoiId(targetRoi.id);
    setPreviewRoiId(targetRoi.id);
    try {
      const packId = packIdForSourceModel(activeSourceModel);
      if (!packId) {
        throw new Error("Select QuantEM or OmniEM before running a model.");
      }
      await ensureModelInstalled(packId, {
        onDownloadQueued: () => refetchJobs(),
        onInstalled: onModelInstalled,
      });
      await rerunSegmentationRoi(
        currentSegmentation.id,
        targetRoi.id,
        activeSourceModel
      );
      await Promise.all([refetchJobs(), refetchSegmentations()]);
    } catch (error) {
      console.error("Failed to start ROI rerun:", error);
      onRunError?.(
        error instanceof Error ? error.message : "The model could not be started."
      );
    } finally {
      setRerunningRoiId(null);
    }
  }, [
    activeRoi,
    activeSourceModel,
    currentSegmentation,
    hasQueuedOrRunningOrganelleTask,
    isRerunningRoi,
    onModelInstalled,
    onRunError,
    refetchJobs,
    refetchSegmentations,
    segmentationRois,
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
    setPreviewRoiId(null);
    try {
      const packId = packIdForSourceModel(activeSourceModel);
      if (!packId) {
        throw new Error("Select QuantEM or OmniEM before running a model.");
      }
      await ensureModelInstalled(packId, {
        onDownloadQueued: () => refetchJobs(),
        onInstalled: onModelInstalled,
      });
      await runFullSegmentation(currentSegmentation.id, activeSourceModel);
      await Promise.all([refetchJobs(), refetchSegmentations()]);
    } catch (error) {
      console.error("Failed to start full segmentation run:", error);
      onRunError?.(
        error instanceof Error ? error.message : "The model could not be started."
      );
    } finally {
      setIsApplyingFull(false);
    }
  }, [
    currentSegmentation,
    activeSourceModel,
    hasQueuedOrRunningOrganelleTask,
    isApplyingFull,
    isRerunningRoi,
    onModelInstalled,
    onRunError,
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
    rerunningRoiId,
    previewRoiId,
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
