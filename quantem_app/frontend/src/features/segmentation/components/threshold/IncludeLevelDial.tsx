import { useCallback, useEffect, useMemo, useState } from "react";

import "./IncludeLevelDial.css";
import { packIdForSourceModel } from "@/features/models/runnable";
import { decodeProbImage } from "@/features/segmentation/erPreview/overlayCanvas";
import { getModelCatalogue, installModelPack } from "@/shared/api/finetune";
import { runFullSegmentation } from "@/shared/api/segmentations/overlays";
import { useJobProgress } from "@/shared/hooks/useJobProgress";
import type { ModelCatalogue, ModelPack, OrganelleKey } from "@/shared/types/finetune";
import { extractApiErrorMessage } from "@/utils/apiErrors";

import { getIncludeLevel, setIncludeLevel } from "./api";
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
  segmentationInternalName?: string | null;
  statusStage?: string | null;
  onSourceModelChange?: (sourceModel: string) => void;
  onRunQueued?: () => void;
  onRunFinished?: () => void;
  /** Refresh objects and overlays after a successful Apply. */
  onReextracted?: () => void;
}

export function IncludeLevelDial({
  segmentationId,
  sourceModel = null,
  segmentationInternalName = null,
  statusStage = null,
  onSourceModelChange,
  onRunQueued,
  onRunFinished,
  onReextracted,
}: IncludeLevelDialProps) {
  const [state, setState] = useState<IncludeLevelState | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [draft, setDraft] = useState<number | null>(null);
  const [applyJobId, setApplyJobId] = useState<string | null>(null);
  const [modelAction, setModelAction] = useState<ModelAction | null>(null);
  const [startingModel, setStartingModel] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const setPreviewOverlay = useThresholdPreviewStore((preview) => preview.setOverlay);
  const setPreviewThreshold = useThresholdPreviewStore((preview) => preview.setThreshold);
  const clearPreview = useThresholdPreviewStore((preview) => preview.clear);

  const load = useCallback(async () => {
    try {
      const next = await getIncludeLevel(segmentationId, sourceModel);
      setState(next);
      setLoadError(null);
      return next;
    } catch (error) {
      setLoadError(extractApiErrorMessage(error, "The threshold could not be read."));
      return null;
    }
  }, [segmentationId, sourceModel]);

  useEffect(() => {
    setDraft(null);
    setApplyJobId(null);
    setSubmitError(null);
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

  const previewUrl = state?.preview_url;
  const previewBounds = state?.preview_bounds;
  const previewBoundsKey = previewBounds?.join(":") ?? "";
  useEffect(() => {
    if (!previewUrl || !state?.can_move) {
      clearPreview();
      return undefined;
    }
    let cancelled = false;
    void decodeProbImage(previewUrl)
      .then((decoded) => {
        if (cancelled) return;
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
        if (!cancelled) setSubmitError("The threshold preview could not be displayed.");
      });
    return () => {
      cancelled = true;
    };
  }, [
    clearPreview,
    previewBounds,
    previewBoundsKey,
    previewUrl,
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
      onSourceModelChange?.(packId);
      const queued = await runFullSegmentation(segmentationId, packId);
      setModelAction({ kind: "run", jobId: queued.job_id, packId });
      onRunQueued?.();
    },
    [onRunQueued, onSourceModelChange, segmentationId]
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

  const working = submitting || applyJobId !== null;
  const settled = state?.include_level ?? state?.default_include_level ?? null;
  const moved = draft !== null && levelsDiffer(draft, settled);
  const needsApply = state?.include_level === null || moved;

  const apply = useCallback(async () => {
    if (working) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const queued = await setIncludeLevel(segmentationId, position, sourceModel);
      setApplyJobId(queued.job_id);
    } catch (error) {
      setSubmitError(extractApiErrorMessage(error, DIAL_FAILED_FALLBACK));
    } finally {
      setSubmitting(false);
    }
  }, [position, segmentationId, sourceModel, working]);

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
          disabled={working || !needsApply}
          onClick={() => void apply()}
        >
          {working ? DIAL_WORKING : "Apply"}
        </button>
      )}

      {submitError ? <p className="include-level-problem" role="alert">{submitError}</p> : null}
    </div>
  );
}
