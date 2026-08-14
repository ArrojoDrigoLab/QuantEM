import { useCallback, useEffect, useMemo, useState } from "react";

import "./IncludeLevelDial.css";
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
  DIAL_BLOCKED_TOOLTIP,
  DIAL_FAILED_FALLBACK,
  DIAL_TITLE,
  DIAL_TOOLTIP,
  DIAL_WORKING,
  formatIncludeLevel,
  levelsDiffer,
} from "./copy";
import { useThresholdPreviewStore } from "./useThresholdPreviewStore";

/** The saved probability map has 8-bit precision; hundredths are ample here. */
const STEP = 0.01;
const PREVIEW_PROCESS_TOOLTIP =
  "Turn this model and threshold setting into objects. Normalizes the shapes and applies size thresholds.";

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

export interface IncludeLevelDialProps {
  segmentationId: string;
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

export function IncludeLevelDial({
  segmentationId,
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
}: IncludeLevelDialProps) {
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

  const load = useCallback(async () => {
    try {
      const next = await getIncludeLevel(
        segmentationId,
        sourceModel,
        roiId,
        adapterId
      );
      setState(next);
      setLoadError(null);
      return next;
    } catch (error) {
      setLoadError(extractApiErrorMessage(error, "The threshold could not be read."));
      return null;
    }
  }, [adapterId, roiId, segmentationId, sourceModel]);

  useEffect(() => {
    setDraft(null);
    setApplyJobId(null);
    setSubmitError(null);
    setPreviewError(null);
    setConfirmationMessage(null);
    clearPreview();
    void load();
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
    }
    void load().then((next) => {
      if (next) setDraft(null);
      if (applyJobStatus === "SUCCESS") onReextracted?.();
    });
  }, [applyJob?.message, applyJobStatus, load, onReextracted]);

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
    if (startingModel || modelAction) return;
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
  }, [modelAction, segmentationInternalName, sourceModel, startRun, startingModel]);

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

  if (loadError) {
    return (
      <div className="include-level-dial" data-testid="include-level-dial">
        <h4 className="include-level-title">{DIAL_TITLE}</h4>
        <p className="include-level-problem" role="alert">{loadError}</p>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="include-level-dial" data-testid="include-level-dial">
        <h4 className="include-level-title">{DIAL_TITLE}</h4>
      </div>
    );
  }

  const blocked = !state.can_move;
  const modelWorking = startingModel || modelAction !== null;
  const objectMode = state.measurement_mode !== "global";
  const candidateCount = state.candidate_count ?? 0;
  const confirmableCandidateCount = state.confirmable_candidate_count ?? candidateCount;
  const canConfirmWholeImage =
    objectMode && !roiId && !needsPreview && confirmableCandidateCount > 0;
  const actionLabel = needsPreview
    ? "Preview / Process"
    : canConfirmWholeImage
      ? "Confirm"
      : roiId
        ? "Previewed"
        : !objectMode
          ? "Ready"
          : state.confirmed_model_count > 0
            ? "Confirmed"
            : "No candidates";
  const blockedTooltip = state.error_code === "probability_map_missing"
    ? DIAL_BLOCKED_TOOLTIP
    : state.detail || DIAL_BLOCKED_TOOLTIP;

  return (
    <div className="include-level-dial" data-testid="include-level-dial">
      <div className="include-level-heading">
        <h4 className="include-level-title">{DIAL_TITLE}</h4>
        <output className="include-level-value" data-testid="include-level-value" aria-live="polite">
          {formatIncludeLevel(position)}
        </output>
        <span
          className="include-level-help"
          role="img"
          tabIndex={0}
          aria-label={blocked ? blockedTooltip : DIAL_TOOLTIP}
          title={blocked ? blockedTooltip : DIAL_TOOLTIP}
        >
          ⓘ
        </span>
      </div>

      <input
        id="include-level-input"
        className="include-level-slider"
        type="range"
        min={state.minimum}
        max={state.maximum}
        step={STEP}
        value={position}
        disabled={blocked || working}
        aria-label="Threshold"
        onChange={(event) => {
          const value = Number(event.target.value);
          setDraft(value);
          setPreviewThreshold(value);
          setConfirmationMessage(null);
        }}
      />

      {blocked ? (
        <button
          type="button"
          className="include-level-apply"
          data-testid="include-level-run-model"
          disabled={modelWorking}
          onClick={() => void runModel()}
        >
          {modelAction?.kind === "install"
            ? "Downloading model…"
            : modelWorking
              ? "Running model…"
              : "Run model"}
        </button>
      ) : (
        <button
          type="button"
          className="include-level-apply"
          data-testid="include-level-apply"
          title={needsPreview ? PREVIEW_PROCESS_TOOLTIP : undefined}
          disabled={working || (!needsPreview && !canConfirmWholeImage)}
          onClick={() => void (needsPreview ? preview() : confirm())}
        >
          {confirming
            ? "Confirming…"
            : submitting || applyJobId !== null
              ? DIAL_WORKING
              : actionLabel}
        </button>
      )}

      {submitError ? <p className="include-level-problem" role="alert">{submitError}</p> : null}
      {previewError ? <p className="include-level-problem" role="alert">{previewError}</p> : null}
      {!needsPreview && !roiId && state.manual_roi_candidate_count > 0 ? (
        <p className="include-level-note">
          {`${state.manual_roi_candidate_count} candidate${
            state.manual_roi_candidate_count === 1 ? "" : "s"
          } inside manually annotated ROIs will remain unchanged.`}
        </p>
      ) : null}
      {confirmationMessage ? (
        <p className="include-level-note" role="status">{confirmationMessage}</p>
      ) : null}
    </div>
  );
}
