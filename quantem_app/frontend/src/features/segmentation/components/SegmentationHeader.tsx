/** Image, model selection, result coverage, and analysis navigation. */

import {
  NONE_SOURCE_MODEL,
  describeDisplayedObjects,
  resolveSourceModelLabel,
} from "@/features/segmentation/components/segmentationHeaderProvenance";
import {
  AboutThisResult,
  HeaderRouteLinks,
} from "@/features/segmentation/components/header/AboutThisResult";
import { HeaderNotices } from "@/features/segmentation/components/header/notices";
import {
  ClearAndRerunConfirmDialog,
  CompletionConfirmDialog,
} from "@/features/segmentation/components/header/CompletionDialogs";
import {
  useClearRerunConfirm,
  useCompletionConfirm,
} from "@/features/segmentation/components/header/completionDialogState";
import { describeObjectsPixelSize } from "@/shared/objectsPixelSize";
import type { ReactNode } from "react";
import type {
  AssetDetail,
  ImageSegmentation,
  SourceModelOption,
} from "@/shared/types";
import { packIdForSourceModel } from "@/features/models/runnable";
import {
  ModelAvailabilityIcon,
} from "@/features/models/ModelAvailabilityIcon";
import type { ModelCatalogue } from "@/shared/types/finetune";
import type { FineTuneAppliedImageEvent } from "@/shared/types/finetune";
import type { LabelingModelOption } from "@/features/models/labelingModelOptions";
import "./SegmentationHeader.css";

interface SegmentationHeaderProps {
  image: AssetDetail;
  currentSegmentation: ImageSegmentation | null;
  sourceModelOptions?: SourceModelOption[];
  activeSourceModel?: string | null;
  modelOptions?: LabelingModelOption[];
  activeModelValue?: string | null;
  /**
   * `source_model` off the overlay manifest: the model whose objects the raster
   * on screen was actually built from. Null while the manifest is loading, or
   * when there is no raster overlay.
   */
  displayedSourceModel?: string | null;
  fineTuneEligibilityRevision?: string;
  onBackToHome: () => void;
  onBackToExperiment?: () => void;
  onBackToViewer: () => void;
  onSourceModelChange?: (sourceModel: string) => void;
  onModelChange?: (model: string) => void;
  onFineTuneImageCompleted?: (event: FineTuneAppliedImageEvent) => void;
  /** Explicit acknowledgement of whole-image manual coverage. */
  onToggleSegmentationComplete: (options?: {
    discardUnconfirmed: boolean;
    acknowledgedDiscardCount: number;
  }) => void | Promise<void>;
  isApplyingFull?: boolean;
  isApplyingActiveRoi?: boolean;
  hasQueuedOrRunningOrganelleTask?: boolean;
  /** Download state for the two released choices in the model picker. */
  modelCatalogue?: ModelCatalogue | null;
  modelRunControl?: ReactNode;
  /**
   * Delete every reviewed object on this segmentation and queue a fresh run.
   *
   * The recovery for calibrated-after-the-fact: the warning chip beside the
   * pixel-size badge says the objects predate the calibration, and until this
   * existed the only route it could point at was an endpoint no screen called
   * (`POST .../labels/clear`) — re-running alone is a no-op, because a new
   * rejected candidate is dropped; confirmed objects stay above the full model
   * preview and are resolved only when that preview is confirmed. Rejections
   * are expected: the confirm dialog stays open and prints them.
   */
  onClearMislabeledObjects?: () => Promise<void>;
}

export function SegmentationHeader({
  image,
  currentSegmentation,
  sourceModelOptions = [],
  activeSourceModel = null,
  modelOptions,
  activeModelValue = null,
  displayedSourceModel = null,
  fineTuneEligibilityRevision = "",
  onBackToHome,
  onBackToExperiment,
  onBackToViewer,
  onSourceModelChange,
  onModelChange,
  onFineTuneImageCompleted,
  onToggleSegmentationComplete,
  isApplyingFull = false,
  isApplyingActiveRoi = false,
  hasQueuedOrRunningOrganelleTask = false,
  modelCatalogue = null,
  modelRunControl,
  onClearMislabeledObjects,
}: SegmentationHeaderProps) {
  const segmentationId = currentSegmentation?.id ?? null;
  const completion = useCompletionConfirm({
    segmentationId,
    onToggleSegmentationComplete,
  });
  const clearRerun = useClearRerunConfirm({ onClearMislabeledObjects });

  const isOrganelle = Boolean(currentSegmentation?.config);
  const isBusy =
    isApplyingFull || isApplyingActiveRoi || hasQueuedOrRunningOrganelleTask;
  const isComplete =
    currentSegmentation?.status_stage === "COMPLETED" || currentSegmentation?.is_complete === true;
  const manualOption = sourceModelOptions.find(
    (option) => option.value === "manual"
  );
  const sourceModelPickerOptions: LabelingModelOption[] =
    modelOptions ??
    [
      manualOption,
      sourceModelOptions.find((option) => option.model_family === "quantem"),
      sourceModelOptions.find((option) => option.model_family === "omniem"),
    ]
      .filter((option): option is SourceModelOption => Boolean(option))
      .map((option) => ({
        value: option.value,
        label: option.value === "manual" ? "Manual segmentation" : option.label,
        sourceModel: option.value,
        adapterId: null,
        packId: option.value === "manual" ? null : option.value,
        adapted: null,
      }));
  const requestedModel =
    activeModelValue || activeSourceModel || sourceModelPickerOptions[0]?.value || NONE_SOURCE_MODEL;
  // "None" was a legacy picker entry. A model-free workspace is manual
  // segmentation, so never surface a fourth, ambiguous method.
  const selectedModel =
    sourceModelPickerOptions.find((option) => option.value === requestedModel) ??
    sourceModelPickerOptions[0] ??
    null;
  const manualRequested =
    requestedModel === NONE_SOURCE_MODEL || requestedModel === "manual";
  const selectedSourceModel = manualRequested
    ? "manual"
    : selectedModel?.sourceModel ?? "manual";
  const selectedModelValue = manualRequested
    ? "manual"
    : selectedModel?.value ?? "manual";
  const selectedPackId = selectedModel?.packId ?? packIdForSourceModel(selectedSourceModel);
  const selectedPack = selectedPackId
    ? modelCatalogue?.packs.find((candidate) => candidate.id === selectedPackId) ?? null
    : null;
  const runTargetLabel =
    selectedModel?.label ?? resolveSourceModelLabel(selectedSourceModel, sourceModelOptions);
  const displayedObjects = describeDisplayedObjects({
    segmentation: currentSegmentation,
    sourceModelOptions,
    activeSourceModel: selectedSourceModel,
    displayedSourceModel,
  });
  /**
   * Whether the objects on this segmentation were made at that pixel size.
   *
   * The resolution tag can say "5 nm/px" while every object on screen was
   * produced before that number existed — the state
   * `run_analysis` blanks its physical units on. This screen is where the user
   * decides the work is finished, so it has to say so here, not in the
   * finished bundle. The verdict is the server's
   * (`objects_pixel_size.predates_calibration`), never re-derived.
   */
  const objectsPixelSize = describeObjectsPixelSize(currentSegmentation);

  return (
    <header className="segmentation-header">
      <div className="header-left">
        <div className="header-nav">
          <button
            type="button"
            className="header-back-button"
            onClick={onBackToHome}
          >
            ← Back to Home
          </button>
          <button
            type="button"
            className="header-back-button"
            onClick={onBackToExperiment}
            disabled={!onBackToExperiment}
            title={
              onBackToExperiment
                ? "Return to this image's experiment"
                : "This image is not assigned to an experiment"
            }
          >
            Back to Experiment
          </button>
          <button
            type="button"
            className="header-back-button"
            onClick={onBackToViewer}
          >
            Back to Viewer
          </button>
        </div>
        <div className="header-info" contentEditable={false}>
          <h2>{image.display_name}</h2>
        </div>
      </div>

      <div className="header-controls">
        {sourceModelPickerOptions.length > 0 && (
          <div className="header-source-model">
            {/* "Model" and "Objects shown" are separated on purpose: the
                picker chooses which model the next run uses *and* which
                model's objects are listed, and those two are not the same claim
                when nothing has been run with the selected model yet. */}
            <label htmlFor="source-model-select" className="header-source-model-caption">
              Model
            </label>
            <div className="header-source-model-select-row">
              <select
                id="source-model-select"
                aria-label="Model"
                value={selectedModelValue}
                onChange={(event) =>
                  (onModelChange ?? onSourceModelChange)?.(event.target.value)
                }
              >
                {sourceModelPickerOptions.map((option) => {
                  const packId = option.packId;
                  const pack = packId
                    ? modelCatalogue?.packs.find(
                        (candidate) => candidate.id === packId
                      ) ?? null
                    : null;
                  const blocked =
                    pack?.installed === true && pack.runnable === false;
                  return (
                    <option
                      key={option.value}
                      value={option.value}
                      disabled={blocked}
                    >
                      {option.label}
                    </option>
                  );
                })}
              </select>
              {selectedPackId && selectedPack && !selectedPack.installed && (
                <ModelAvailabilityIcon
                  pack={selectedPack}
                  className="header-model-availability"
                />
              )}
            </div>
          </div>
        )}
        <AboutThisResult
          isOrganelle={isOrganelle}
          displayedObjects={displayedObjects}
          modelRunControl={modelRunControl}
          objectsPixelSize={objectsPixelSize}
          currentSegmentation={currentSegmentation}
          isBusy={isBusy}
          isComplete={isComplete}
          onOpenClearRerun={onClearMislabeledObjects ? clearRerun.open : undefined}
        />
        <HeaderNotices
          currentSegmentation={currentSegmentation}
          isComplete={isComplete}
        />
        <HeaderRouteLinks
          image={image}
          currentSegmentation={currentSegmentation}
          fineTuneEligibilityRevision={fineTuneEligibilityRevision}
          activeBaseModel={selectedPackId}
          onFineTuneImageCompleted={onFineTuneImageCompleted}
        />
        <button className="segmentation-complete-button" disabled={!currentSegmentation || isComplete}
          onClick={() => completion.open()}>
          {isComplete ? "Whole image segmented" : "Confirm whole-image manual segmentation…"}
        </button>
      </div>

      <ClearAndRerunConfirmDialog
        state={clearRerun}
        currentSegmentation={currentSegmentation}
        runTargetLabel={runTargetLabel}
      />

      <CompletionConfirmDialog
        state={completion}
        sourceModelOptions={sourceModelOptions}
      />
    </header>
  );
}
