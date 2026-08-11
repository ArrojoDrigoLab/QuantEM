/**
 * The asset and its segmentations, kept fresh while either is still working.
 *
 * Moved out of `ViewerScreen.tsx` unchanged. Two polls live here — the 2 s
 * asset poll that runs until preprocessing reaches a terminal stage, and the
 * 3 s segmentation poll that runs while any run is still going — because they
 * are the same question asked of two endpoints, and because separating them
 * from the overlay state is what lets the run work and the overlay work be
 * owned by different people.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { getAsset, getAssetSegmentations } from "@/shared/api/assets";
import { getRunPlan, startImageRun } from "@/shared/api/runs";
import type { PreprocessStage } from "@/shared/types/common";
import type { ImageSegmentation, StatusStage } from "@/shared/types/images";
import type { RunPlan } from "@/shared/types/runs";

export const IMAGE_READY_STAGES: PreprocessStage[] = ["DONE", "SKIPPED"];
export const IMAGE_TERMINAL_STAGES: PreprocessStage[] = [
  "DONE",
  "SKIPPED",
  "FAILED",
  "CANCELLED",
];

export const PROCESSING_STATUS_STAGES: StatusStage[] = [
  "UNSTARTED",
  "RUNNING_INFERENCE",
  "EXTRACTING_CANDIDATES",
];

export function getStageLabel(stage: PreprocessStage): string {
  switch (stage) {
    case "ENCODING": return "Encoding image";
    case "SAM": return "Running segmentation model";
    case "FEATURES": return "Extracting features";
    case "FAILED": return "Preprocessing failed";
    case "CANCELLED": return "Preprocessing cancelled";
    case "NONE": return "Queued for preprocessing";
    default: return "Processing";
  }
}

export function useViewerAssetState(selectedAssetId: string | null) {
  const { data: image, refetch: refetchImage } = useApiQuery(
    () => {
      if (!selectedAssetId) throw new Error("No asset selected");
      return getAsset(selectedAssetId);
    },
    [selectedAssetId]
  );

  const imageReady = image
    ? IMAGE_READY_STAGES.includes(image.preprocess_stage)
    : false;

  const imageTerminal = image
    ? IMAGE_TERMINAL_STAGES.includes(image.preprocess_stage)
    : false;

  // Poll image status while preprocessing is still running
  useEffect(() => {
    if (!image || imageTerminal) return undefined;
    const interval = setInterval(() => {
      void refetchImage();
    }, 2000);
    return () => clearInterval(interval);
  }, [image, imageTerminal, refetchImage]);

  const { data: segmentations, refetch: refetchSegmentations } = useApiQuery<
    ImageSegmentation[]
  >(
    () => {
      if (!selectedAssetId) throw new Error("No asset selected");
      return getAssetSegmentations(selectedAssetId);
    },
    [selectedAssetId]
  );

  const visibleSegmentations = useMemo(() => {
    if (!segmentations) return [];
    return segmentations;
  }, [segmentations]);

  // Poll segmentation statuses while any are still processing
  const hasProcessingSegmentations = useMemo(() => {
    return visibleSegmentations.some((seg) =>
      PROCESSING_STATUS_STAGES.includes(seg.status_stage)
    );
  }, [visibleSegmentations]);

  useEffect(() => {
    if (!hasProcessingSegmentations) return;
    const interval = setInterval(() => void refetchSegmentations(), 3000);
    return () => clearInterval(interval);
  }, [hasProcessingSegmentations, refetchSegmentations]);

  return {
    image,
    refetchImage,
    imageReady,
    imageTerminal,
    refetchSegmentations,
    visibleSegmentations,
    hasProcessingSegmentations,
  };
}

/**
 * The organelles this image will run, and one button to start them together.
 *
 * The rules the workspace asked for, in one place:
 *
 * * **Mitochondria is ticked, alone.** It is installed on most machines and it
 *   is the common ask. Nothing else is pre-ticked -- four pre-ticked packs at
 *   1.1 GB is a trap a user will not untick.
 * * **A model that is not on this machine may still be ticked.** It downloads
 *   in the background and the run waits for it. The cost of doing so is quoted
 *   before the click, deduped, from the plan.
 * * **The plan is refetched whenever the ticks change** and never queues
 *   anything, so the price is visible while the user is still deciding.
 */
export const DEFAULT_TICKED_ORGANELLES: readonly string[] = ["mito"];

export function useImageRunSelection(assetId: string | null) {
  const [ticked, setTicked] = useState<string[]>([...DEFAULT_TICKED_ORGANELLES]);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const { data: plan, refetch: refetchPlan } = useApiQuery<RunPlan>(() => {
    if (!assetId) throw new Error("No image selected");
    return getRunPlan(assetId, ticked);
  }, [assetId, ticked.join(",")]);

  const toggle = useCallback((organelle: string) => {
    setTicked((current) =>
      current.includes(organelle)
        ? current.filter((item) => item !== organelle)
        : [...current, organelle]
    );
  }, []);

  const start = useCallback(async (): Promise<string | null> => {
    if (!assetId || ticked.length === 0) return null;
    setStarting(true);
    setStartError(null);
    try {
      const response = await startImageRun(assetId, ticked);
      return response.job_id;
    } catch (error) {
      setStartError(
        error instanceof Error ? error.message : "This run could not be started."
      );
      return null;
    } finally {
      setStarting(false);
    }
  }, [assetId, ticked]);

  return {
    plan: plan ?? null,
    refetchPlan,
    ticked,
    toggle,
    setTicked,
    start,
    starting,
    startError,
  };
}
