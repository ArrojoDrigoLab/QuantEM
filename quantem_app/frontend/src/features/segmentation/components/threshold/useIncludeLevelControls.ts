import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { packIdForSourceModel } from "@/features/models/runnable";
import { decodeProbImage } from "@/features/segmentation/erPreview/overlayCanvas";
import { getModelCatalogue, installModelPack } from "@/shared/api/finetune";
import { runFullSegmentation } from "@/shared/api/segmentations/overlays";
import { rerunSegmentationRoi } from "@/shared/api/segmentations/rois";
import { useJobProgress } from "@/shared/hooks/useJobProgress";
import type { ModelCatalogue, ModelPack, OrganelleKey } from "@/shared/types/finetune";
import type { SegmentationOverlayMutationState } from "@/shared/types/segmentation";
import { extractApiErrorMessage } from "@/utils/apiErrors";

import { confirmModelOutput, getIncludeLevel, setIncludeLevel } from "./api";
import type { IncludeLevelState } from "./api";
import {
  DIAL_FAILED_FALLBACK,
  levelsDiffer,
} from "./copy";
import { useThresholdPreviewStore } from "./useThresholdPreviewStore";

type ModelAction = {
  kind: "install" | "run";
  jobId: string;
  packId: string;
};

function organelleFrom(
  segmentationInternalName: string | null | undefined,
  sourceModel: string | null | undefined
): OrganelleKey | null {
  const packId = packIdForSourceModel(sourceModel);
  const suffix = packId?.split(":")[1];
  if (suffix === "mito" || suffix === "er" || suffix === "nucleus" || suffix === "ld") {
    return suffix;
  }
  const name = String(segmentationInternalName ?? "").toLowerCase();
  if (name.includes("mito")) return "mito";
  if (name.includes("nucleus") || name.includes("nuclei")) return "nucleus";
  if (name.includes("lipid") || name.endsWith("_ld")) return "ld";
  if (name.includes("_er") || name.endsWith("er")) return "er";
  return null;
}

function choosePack(
  catalogue: ModelCatalogue,
  organelle: OrganelleKey,
  sourceModel: string | null | undefined
): ModelPack | null {
  const requested = packIdForSourceModel(sourceModel);
  const packs = catalogue.packs.filter((pack) => pack.organelle === organelle);
  if (requested) {
    return packs.find((pack) => pack.id === requested &&
      (!pack.installed || pack.runnable !== false)) ?? null;
  }
  const usableInstalled = (pack: ModelPack) => pack.installed && pack.runnable !== false;
  return (
    packs.find((pack) => pack.id === requested && usableInstalled(pack)) ??
    packs.find(usableInstalled) ??
    packs.find((pack) => pack.id === requested && !pack.installed) ??
    packs.find((pack) => pack.id === `quantem:${organelle}` && !pack.installed) ??
    packs.find((pack) => !pack.installed) ??
    null
  );
}

export interface IncludeLevelOptions {
  segmentationId: string;
  enabled?: boolean;
  busy?: boolean;
  sourceModel?: string | null;
  adapterId?: string | null;
  segmentationInternalName?: string | null;
  statusStage?: string | null;
  /** Scope the model preview and Preview action to the ROI most recently tested. */
  roiId?: string | null;
  onSourceModelChange?: (sourceModel: string) => void;
  onRunQueued?: () => void;
  onRunFinished?: () => void;
  /** Refresh objects and overlays after a successful Preview or Confirm. */
  onReextracted?: () => void;
  /** Keep the confirmed-pane veil active across the asynchronous raster rebuild. */
  onConfirmStarted?: () => void;
  onConfirmCommitted?: (overlay: SegmentationOverlayMutationState | null) => void;
  onConfirmFailed?: () => void;
}

export function useIncludeLevelControls({
  segmentationId,
  enabled = true,
  busy = false,
  sourceModel = null,
  adapterId = null,
  segmentationInternalName = null,
  statusStage = null,
  roiId = null,
  onSourceModelChange,
  onRunQueued,
  onRunFinished,
  onReextracted,
  onConfirmStarted,
  onConfirmCommitted,
  onConfirmFailed,
}: IncludeLevelOptions) {
  const [state, setState] = useState<IncludeLevelState | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [draft, setDraft] = useState<number | null>(null);
  const [applyJobId, setApplyJobId] = useState<string | null>(null);
  const [modelAction, setModelAction] = useState<ModelAction | null>(null);
  const [startingModel, setStartingModel] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [confirmationMessage, setConfirmationMessage] = useState<string | null>(null);
  const setPreviewOverlay = useThresholdPreviewStore((preview) => preview.setOverlay);
  const setPreviewThreshold = useThresholdPreviewStore((preview) => preview.setThreshold);
  const clearPreview = useThresholdPreviewStore((preview) => preview.clear);
  const loadRequest = useRef(0);

  const load = useCallback(async () => {
    if (!enabled || !segmentationId) return null;
    const request = ++loadRequest.current;
    try {
      const next = await getIncludeLevel(
        segmentationId,
        sourceModel,
        roiId,
        adapterId
      );
      if (request !== loadRequest.current) return null;
      setState(next);
      setLoadError(null);
      return next;
    } catch (error) {
      if (request !== loadRequest.current) return null;
      setLoadError(extractApiErrorMessage(error, "The threshold could not be read."));
      return null;
    }
  }, [adapterId, enabled, roiId, segmentationId, sourceModel]);

  useEffect(() => {
    setState(null);
    setLoadError(null);
    setModelAction(null);
    setDraft(null);
    setApplyJobId(null);
    setSubmitError(null);
    setPreviewError(null);
    setConfirmationMessage(null);
    clearPreview();
    void load();
    return () => {
      loadRequest.current += 1;
      clearPreview();
    };
  }, [clearPreview, load]);

  // A run publishes THRESHOLD_READY before its queue row finishes. Re-read on
  // stage changes so the slider unlocks at that point, not at job settlement.
  useEffect(() => {
    if (statusStage) void load();
  }, [load, statusStage]);

  const position = useMemo(() => {
    if (draft !== null) return draft;
    if (state?.include_level !== null && state?.include_level !== undefined) {
      return state.include_level;
    }
    if (state?.default_include_level !== null && state?.default_include_level !== undefined) {
      return state.default_include_level;
    }
    return 0.5;
  }, [draft, state]);

  useEffect(() => {
    setPreviewThreshold(position);
  }, [position, setPreviewThreshold]);

  const settled = state?.include_level ?? state?.default_include_level ?? null;
  const moved = draft !== null && levelsDiffer(draft, settled);
  const needsPreview = state?.include_level === null || moved;

  const previewUrl = state?.preview_url;
  const previewBounds = state?.preview_bounds;
  const previewBoundsKey = previewBounds?.join(":") ?? "";
  useEffect(() => {
    // The probability bitmap is only a live threshold-selection aid. Once the
    // selected level has been materialized, the candidate overlay is the
    // authoritative post-processed preview (closing, hole fill, size filter),
    // and drawing both produces misleading double opacity and apparent holes.
    if (!needsPreview || !previewUrl || !state?.can_move) {
      clearPreview();
      return undefined;
    }
    let cancelled = false;
    void decodeProbImage(previewUrl)
      .then((decoded) => {
        if (cancelled) return;
        setPreviewError(null);
        setPreviewOverlay({
          probData: decoded.data,
          width: decoded.width,
          height: decoded.height,
          bounds: previewBounds ?? [0, 0, decoded.width, decoded.height],
          color: [239, 68, 68],
          sourceModel: sourceModel || "model",
        });
      })
      .catch(() => {
        if (!cancelled) setPreviewError("The threshold preview could not be displayed.");
      });
    return () => {
      cancelled = true;
    };
  }, [
    clearPreview,
    previewBounds,
    previewBoundsKey,
    previewUrl,
    needsPreview,
    setPreviewOverlay,
    sourceModel,
    state?.can_move,
  ]);

  const { job: applyJob } = useJobProgress(applyJobId);
  const applyJobStatus = applyJob?.status;
  useEffect(() => {
    if (!applyJobStatus || applyJobStatus === "PENDING" || applyJobStatus === "RUNNING") return;
    setApplyJobId(null);
    if (applyJobStatus === "FAILED" || applyJobStatus === "CANCELLED") {
      setSubmitError(applyJob?.message?.trim() || DIAL_FAILED_FALLBACK);
    } else if (
      applyJobStatus === "SUCCESS" &&
      applyJob?.result_json?.feature_job_id
    ) {
      setConfirmationMessage(
        "Preview geometry is ready. Measuring intensity and object details in the background."
      );
    }
    void load().then((next) => {
      if (next) setDraft(null);
      if (applyJobStatus === "SUCCESS") onReextracted?.();
    });
  }, [applyJob?.message, applyJob?.result_json, applyJobStatus, load, onReextracted]);

  const startRun = useCallback(
    async (packId: string) => {
      if (!adapterId) onSourceModelChange?.(packId);
      const queued = roiId
        ? await rerunSegmentationRoi(segmentationId, roiId, packId, adapterId)
        : await runFullSegmentation(segmentationId, packId, adapterId);
      setModelAction({ kind: "run", jobId: queued.job_id, packId });
      onRunQueued?.();
    },
    [adapterId, onRunQueued, onSourceModelChange, roiId, segmentationId]
  );

  const { job: modelJob } = useJobProgress(modelAction?.jobId ?? null);
  const modelJobStatus = modelJob?.status;
  useEffect(() => {
    if (!modelAction || !modelJobStatus) return;
    if (modelJobStatus === "PENDING" || modelJobStatus === "RUNNING") return;
    const action = modelAction;
    setModelAction(null);
    if (modelJobStatus === "FAILED" || modelJobStatus === "CANCELLED") {
      setSubmitError(modelJob?.message?.trim() || "The model could not be run.");
      return;
    }
    if (action.kind === "install") {
      void getModelCatalogue()
        .then((catalogue) => {
          const installed = catalogue.packs.find((pack) => pack.id === action.packId);
          if (installed?.runnable === false) {
            throw new Error(installed.reason || "The downloaded model cannot run here.");
          }
          return startRun(action.packId);
        })
        .catch((error) => {
          setSubmitError(extractApiErrorMessage(error, "The model could not be prepared."));
        });
      return;
    }
    void load();
    onRunFinished?.();
  }, [load, modelAction, modelJob?.message, modelJobStatus, onRunFinished, startRun]);

  // The saved result may become available while the worker is still closing
  // out the job. Poll only during the run and stop as soon as the action ends.
  useEffect(() => {
    if (modelAction?.kind !== "run") return undefined;
    const timer = window.setInterval(() => void load(), 750);
    return () => window.clearInterval(timer);
  }, [load, modelAction?.kind]);

  const runModel = useCallback(async () => {
    if (startingModel || modelAction || busy || !enabled) return;
    setStartingModel(true);
    setSubmitError(null);
    try {
      const catalogue = await getModelCatalogue();
      const organelle = organelleFrom(segmentationInternalName, sourceModel);
      if (!organelle) throw new Error("No model is available for this segmentation type.");
      const pack = choosePack(catalogue, organelle, sourceModel);
      if (!pack) {
        const blocked = catalogue.packs.find((candidate) => candidate.organelle === organelle);
        throw new Error(blocked?.reason || "No runnable model is available for this organelle.");
      }
      if (pack.installed) {
        await startRun(pack.id);
      } else {
        const install = await installModelPack(pack.id);
        if (install.job_id) {
          setModelAction({ kind: "install", jobId: install.job_id, packId: pack.id });
        } else {
          await startRun(pack.id);
        }
      }
    } catch (error) {
      setSubmitError(extractApiErrorMessage(error, "The model could not be started."));
    } finally {
      setStartingModel(false);
    }
  }, [busy, enabled, modelAction, segmentationInternalName, sourceModel, startRun, startingModel]);

  const working = submitting || confirming || applyJobId !== null;

  const preview = useCallback(async () => {
    if (working) return;
    setSubmitting(true);
    setSubmitError(null);
    setConfirmationMessage(null);
    try {
      const queued = await setIncludeLevel(
        segmentationId,
        position,
        sourceModel,
        roiId,
        adapterId
      );
      setApplyJobId(queued.job_id);
    } catch (error) {
      setSubmitError(extractApiErrorMessage(error, DIAL_FAILED_FALLBACK));
    } finally {
      setSubmitting(false);
    }
  }, [adapterId, position, roiId, segmentationId, sourceModel, working]);

  const confirm = useCallback(async () => {
    if (working || !sourceModel || roiId) return;
    setConfirming(true);
    onConfirmStarted?.();
    setSubmitError(null);
    try {
      const result = await confirmModelOutput(segmentationId, sourceModel);
      onConfirmCommitted?.(result.overlay);
      const confirmed = result.confirmed_count;
      const skipped = result.skipped_manual_roi_count;
      setConfirmationMessage(
        confirmed > 0
          ? `Confirmed ${confirmed} model object${confirmed === 1 ? "" : "s"} for analysis.${
              skipped > 0
                ? ` ${skipped} candidate${skipped === 1 ? "" : "s"} inside manually annotated ROIs remained unchanged.`
                : ""
            }`
          : skipped > 0
            ? `No candidates outside manually annotated ROIs needed confirmation. ${skipped} remained unchanged.`
            : "No model candidates needed confirmation."
      );
      const next = await load();
      if (next) setDraft(null);
      onReextracted?.();
    } catch (error) {
      onConfirmFailed?.();
      setSubmitError(
        extractApiErrorMessage(error, "The model output could not be confirmed.")
      );
    } finally {
      setConfirming(false);
    }
  }, [
    load,
    onConfirmCommitted,
    onConfirmFailed,
    onConfirmStarted,
    onReextracted,
    roiId,
    segmentationId,
    sourceModel,
    working,
  ]);

  const blocked = !state?.can_move;
  const candidateCount = state?.candidate_count ?? 0;
  const canConfirmWholeImage = Boolean(
    state && state.measurement_mode !== "global" && !roiId && !needsPreview &&
    (state.confirmable_candidate_count ?? candidateCount) > 0
  );

  return {
    state, loadError, position, settled, moved, needsPreview, blocked,
    working: working || busy, confirming, submitting, applyJobId,
    modelWorking: startingModel || modelAction !== null || busy,
    downloading: modelAction?.kind === "install",
    submitError, previewError, confirmationMessage, canConfirmWholeImage, roiId,
    runModel, preview, confirm,
    changeThreshold(value: number) {
      setDraft(value);
      setPreviewThreshold(value);
      setConfirmationMessage(null);
    },
  };
}

export type IncludeLevelControls = ReturnType<typeof useIncludeLevelControls>;
