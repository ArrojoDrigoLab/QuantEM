/**
 * Header for the labeling screen: what image, what objects, what produced them.
 *
 * Provenance is the reason this component is fussy. It used to render
 * "Model: MitoNet v1_mini (Default)" out of a `config.mitonet_model_variant`
 * that the serializer stopped emitting when MitoNet was dropped -- so the
 * annotation screen named a model that produced nothing in the object set the
 * user was correcting. Everything shown here now comes from data the server
 * actually sends: `source_models` (with per-model object counts), the overlay
 * manifest's `source_model` (what the displayed raster was built from).
 *
 * ## Why this file is a composition and not a component
 *
 * It was 966 lines holding roughly twenty distinct surfaces, and five separate
 * packages needed to change different ones of them at the same time. The
 * surfaces now live in `./header/`, one file per thing an owner would want:
 *
 * * `header/AboutThisResult.tsx` — the provenance chips,
 *   the discard-and-re-run trigger, and the two route links;
 * * `header/notices.tsx` — every `role="status"` sentence under the controls;
 * * `header/CompletionDialogs.tsx` — Mark Image Done and delete-and-re-run,
 *   with the state each of them needs in `header/completionDialogState.ts`.
 *
 * Nothing moved changed. The props of `SegmentationHeader` are what they were,
 * the children of `.header-controls` are the same elements in the same order,
 * and every `className`, `data-testid`, `role` and literal string is untouched.
 */

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
import type { AppliedAdapterState } from "@/features/models/appliedAdapter";
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
import "./SegmentationHeader.css";

interface SegmentationHeaderProps {
  image: AssetDetail;
  currentSegmentation: ImageSegmentation | null;
  sourceModelOptions?: SourceModelOption[];
  activeSourceModel?: string | null;
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
  /**
   * Lock the segmentation, or unlock it.
   *
   * `discardUnconfirmed` is the destructive half, and is only ever passed after
   * the user has ticked it in the confirmation. `acknowledgedDiscardCount` is
   * the number they were shown; the endpoint refuses a stale one.
   */
  onToggleSegmentationComplete: (options?: {
    discardUnconfirmed: boolean;
    acknowledgedDiscardCount: number;
  }) => void | Promise<void>;
  isApplyingFull?: boolean;
  isApplyingActiveRoi?: boolean;
  hasQueuedOrRunningOrganelleTask?: boolean;
  /** Download state for the two released choices in the model picker. */
  modelCatalogue?: ModelCatalogue | null;
  /**
   * The adapter applied to this segmentation, and whether the selected source
   * model means the next run will actually go through it. Null when there is
   * none, or when the catalogue has not answered.
   */
  appliedAdapter?: AppliedAdapterState | null;
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
  displayedSourceModel = null,
  fineTuneEligibilityRevision = "",
  onBackToHome,
  onBackToExperiment,
  onBackToViewer,
  onSourceModelChange,
  onToggleSegmentationComplete,
  isApplyingFull = false,
  isApplyingActiveRoi = false,
  hasQueuedOrRunningOrganelleTask = false,
  modelCatalogue = null,
  appliedAdapter = null,
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
  const sourceModelPickerOptions = [
    manualOption,
    sourceModelOptions.find((option) => option.model_family === "quantem"),
    sourceModelOptions.find((option) => option.model_family === "omniem"),
  ].filter((option): option is SourceModelOption => Boolean(option));
  const requestedSourceModel =
    activeSourceModel || sourceModelPickerOptions[0]?.value || NONE_SOURCE_MODEL;
  // "None" was a legacy picker entry. A model-free workspace is manual
  // segmentation, so never surface a fourth, ambiguous method.
  const selectedSourceModel =
    requestedSourceModel === NONE_SOURCE_MODEL ? "manual" : requestedSourceModel;
  const selectedPackId = packIdForSourceModel(selectedSourceModel);
  const selectedPack = selectedPackId
    ? modelCatalogue?.packs.find((candidate) => candidate.id === selectedPackId) ?? null
    : null;
  const runTargetLabel = resolveSourceModelLabel(
    selectedSourceModel,
    sourceModelOptions
  );
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
            ← Back to Library
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
        <div className="header-info">
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
                value={selectedSourceModel}
                onChange={(event) => onSourceModelChange?.(event.target.value)}
              >
                {sourceModelPickerOptions.map((option) => {
                  const packId = packIdForSourceModel(option.value);
                  const pack = packId
                    ? modelCatalogue?.packs.find(
                        (candidate) => candidate.id === packId
                      ) ?? null
                    : null;
                  const blocked =
                    pack?.installed === true && pack.runnable === false;
                  const adapted = Boolean(
                    appliedAdapter && appliedAdapter.adapter.base === option.value
                  );
                  return (
                    <option
                      key={option.value}
                      value={option.value}
                      disabled={blocked}
                    >
                      {option.value === "manual"
                        ? "Manual segmentation"
                        : option.label}
                      {adapted ? " (adapted)" : ""}
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
          objectsPixelSize={objectsPixelSize}
          currentSegmentation={currentSegmentation}
          isBusy={isBusy}
          isComplete={isComplete}
          onOpenClearRerun={onClearMislabeledObjects ? clearRerun.open : undefined}
        />
        <HeaderNotices
          currentSegmentation={currentSegmentation}
          isOrganelle={isOrganelle}
          appliedAdapter={appliedAdapter}
          runTargetLabel={runTargetLabel}
          isComplete={isComplete}
        />
        <HeaderRouteLinks
          image={image}
          currentSegmentation={currentSegmentation}
          fineTuneEligibilityRevision={fineTuneEligibilityRevision}
        />
        <button
          className="segmentation-complete-button"
          onClick={() => {
            // Unlocking only moves the stage back and restores whatever the
            // last completion archived, so it stays a single click. Completing
            // is a state change with a destructive option attached, so it asks.
            if (isComplete) {
              onToggleSegmentationComplete();
              return;
            }
            completion.open();
          }}
          disabled={!currentSegmentation}
        >
          {isComplete ? "Unlock segmentation" : "Mark Image Done"}
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
