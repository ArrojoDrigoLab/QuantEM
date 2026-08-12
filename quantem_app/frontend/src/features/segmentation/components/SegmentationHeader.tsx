/**
 * Header for the labeling screen: what image, what objects, what produced them.
 *
 * Provenance is the reason this component is fussy. It used to render
 * "Model: MitoNet v1_mini (Default)" out of a `config.mitonet_model_variant`
 * that the serializer stopped emitting when MitoNet was dropped -- so the
 * annotation screen named a model that produced nothing in the object set the
 * user was correcting. Everything shown here now comes from data the server
 * actually sends: `source_models` (with per-model object counts), the overlay
 * manifest's `source_model` (what the displayed raster was built from) and
 * `Asset.pixel_size_nm`.
 *
 * ## Why this file is a composition and not a component
 *
 * It was 966 lines holding roughly twenty distinct surfaces, and five separate
 * packages needed to change different ones of them at the same time. The
 * surfaces now live in `./header/`, one file per thing an owner would want:
 *
 * * `header/AboutThisResult.tsx` — the provenance chips, the pixel-size badge,
 *   the discard-and-re-run trigger, and the two route links;
 * * `header/RunControls.tsx` — Run Full Segmentation, its progress readout and
 *   its uncalibrated-run confirmation;
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
import { segmentationDisplayName } from "@/shared/segmentationNames";
import {
  AboutThisResult,
  HeaderRouteLinks,
} from "@/features/segmentation/components/header/AboutThisResult";
import {
  RunControls,
  RunScaleMismatchDialog,
} from "@/features/segmentation/components/header/RunControls";
import { useRunScaleConfirm } from "@/features/segmentation/components/header/runControlsState";
import { HeaderNotices } from "@/features/segmentation/components/header/notices";
import {
  ClearAndRerunConfirmDialog,
  CompletionConfirmDialog,
} from "@/features/segmentation/components/header/CompletionDialogs";
import {
  useClearRerunConfirm,
  useCompletionConfirm,
} from "@/features/segmentation/components/header/completionDialogState";
import { resolvePixelSize } from "@/shared/pixelSize";
import { describeObjectsPixelSize } from "@/shared/objectsPixelSize";
import type { AppliedAdapterState } from "@/features/models/appliedAdapter";
import type { ScaleMismatch } from "@/features/models/scaleMismatch";
import type {
  AssetDetail,
  ImageSegmentation,
  SourceModelOption,
} from "@/shared/types";
import type { SegmentationRoi } from "@/shared/types/segmentation";
import type { Runnability } from "@/features/models/runnable";
import "./SegmentationHeader.css";

/**
 * The default: say nothing about whether a model can run.
 *
 * A header rendered without a catalogue (tests, or a build whose `/api/models/`
 * did not answer) must behave exactly as it did before runnability existed.
 */
const UNKNOWN_RUNNABILITY: Runnability = {
  state: "unknown",
  reason: null,
  label: "run state unknown",
};

interface SegmentationHeaderProps {
  image: AssetDetail;
  currentSegmentation: ImageSegmentation | null;
  visibleSegmentations: ImageSegmentation[];
  sourceModelOptions?: SourceModelOption[];
  activeSourceModel?: string | null;
  /**
   * `source_model` off the overlay manifest: the model whose objects the raster
   * on screen was actually built from. Null while the manifest is loading, or
   * when there is no raster overlay.
   */
  displayedSourceModel?: string | null;
  fineTuneEligibilityRevision?: string;
  fullImageActive?: boolean;
  fullImageProgress?: number | null;
  onBackToHome: () => void;
  onSegmentationChange: (segId: string) => void;
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
  onApplyFullImage?: () => void;
  isApplyingFull?: boolean;
  /** The current per-segmentation ROI, when one has been selected. */
  activeRoi?: SegmentationRoi | null;
  /** Queue the selected model only over the active, unfinished ROI. */
  onApplyActiveRoi?: () => void;
  isApplyingActiveRoi?: boolean;
  hasQueuedOrRunningOrganelleTask?: boolean;
  /**
   * Whether the selected source model can actually be loaded on this machine.
   *
   * Defaults to "unknown", which changes nothing — the button behaves as it
   * always did when the catalogue has not answered. Only a definite "blocked"
   * disables it, and then the reason is on screen rather than in a job error
   * the queue banner overwrites.
   */
  modelRunnability?: Runnability;
  /**
   * The adapter applied to this segmentation, and whether the selected source
   * model means the next run will actually go through it. Null when there is
   * none, or when the catalogue has not answered.
   */
  appliedAdapter?: AppliedAdapterState | null;
  /**
   * Set when the model this run would use declares a working resolution and
   * this image has no pixel size to resample to.
   *
   * The create-segmentation dialog has warned about this for a while; this
   * button queues the identical inference pass and fired instantly with no
   * dialog at all. Null when there is nothing to warn about, or when the
   * catalogue has not answered.
   */
  runScaleMismatch?: ScaleMismatch | null;
  /**
   * Delete every reviewed object on this segmentation and queue a fresh run.
   *
   * The recovery for calibrated-after-the-fact: the warning chip beside the
   * pixel-size badge says the objects predate the calibration, and until this
   * existed the only route it could point at was an endpoint no screen called
   * (`POST .../labels/clear`) — re-running alone is a no-op, because a new
   * candidate landing on a confirmed or excluded object is dropped. Rejections
   * are expected: the confirm dialog stays open and prints them.
   */
  onClearMislabeledObjects?: () => Promise<void>;
}

export function SegmentationHeader({
  image,
  currentSegmentation,
  visibleSegmentations,
  sourceModelOptions = [],
  activeSourceModel = null,
  displayedSourceModel = null,
  fineTuneEligibilityRevision = "",
  fullImageActive = false,
  fullImageProgress = null,
  onBackToHome,
  onSegmentationChange,
  onSourceModelChange,
  onToggleSegmentationComplete,
  onApplyFullImage,
  isApplyingFull = false,
  activeRoi = null,
  onApplyActiveRoi,
  isApplyingActiveRoi = false,
  hasQueuedOrRunningOrganelleTask = false,
  modelRunnability = UNKNOWN_RUNNABILITY,
  appliedAdapter = null,
  runScaleMismatch = null,
  onClearMislabeledObjects,
}: SegmentationHeaderProps) {
  const segmentationId = currentSegmentation?.id ?? null;
  const completion = useCompletionConfirm({
    segmentationId,
    onToggleSegmentationComplete,
  });
  const runScaleConfirm = useRunScaleConfirm();
  const clearRerun = useClearRerunConfirm({ onClearMislabeledObjects });

  const isOrganelle = Boolean(currentSegmentation?.config);
  const isBusy =
    isApplyingFull || isApplyingActiveRoi || hasQueuedOrRunningOrganelleTask;
  const modelBlocked = modelRunnability.state === "blocked";
  const showFullImageProgress =
    isApplyingFull || fullImageActive || fullImageProgress !== null;
  const isComplete =
    currentSegmentation?.status_stage === "COMPLETED" || currentSegmentation?.is_complete === true;
  /**
   * A locked segmentation refuses mutations server-side, so the button that
   * queues a whole inference pass must not still look available.
   *
   * The dialog has always promised "Marking it done locks the segmentation";
   * until the lock was enforced, every mutation control stayed enabled and the
   * promise was simply false. Disabling here is the visible half of it.
   */
  const applyFullDisabled = isBusy || modelBlocked || isComplete;
  const showActiveRoiRun = Boolean(
    activeRoi && activeRoi.completed_for_segmentation !== true
  );
  const selectedSourceModel =
    activeSourceModel || sourceModelOptions[0]?.value || NONE_SOURCE_MODEL;
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
  const pixelSize = resolvePixelSize(image);
  /**
   * Whether the objects on this segmentation were made at that pixel size.
   *
   * The badge beside this chip can say "5 nm/px · entered by hand" while every
   * object on screen was produced before that number existed — the state
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
        </div>
        <div className="header-info">
          <h2>{image.display_name}</h2>
          <span className="filename">{image.original_filename}</span>
        </div>
      </div>

      <div className="header-controls">
        <select
          id="segmentation-select"
          aria-label="Segmentation type"
          value={currentSegmentation?.id || ""}
          onChange={(e) => onSegmentationChange(e.target.value)}
        >
          {visibleSegmentations.length > 0 ? (
            visibleSegmentations.map((seg) => (
              <option key={seg.id} value={seg.id}>
                {segmentationDisplayName(seg)}
              </option>
            ))
          ) : (
            <option value="">No segmentations</option>
          )}
        </select>
        {sourceModelOptions.length > 0 && (
          <label className="header-source-model" htmlFor="source-model-select">
            {/* "Model to run" and "Objects shown" are separated on purpose: the
                selector chooses which model the next run uses *and* which
                model's objects are listed, and those two are not the same claim
                when nothing has been run with the selected model yet. */}
            <span className="header-source-model-caption">Model to run</span>
            <select
              id="source-model-select"
              aria-label="Source model"
              value={selectedSourceModel}
              onChange={(e) => onSourceModelChange?.(e.target.value)}
            >
              {sourceModelOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {/* The picker used to offer "QuantEM / OmniEM / Manual /
                      None" and never mention that one of them is routed
                      through an adapted model. Choosing any other entry turns
                      the adaptation off, so the choice has to be visible in
                      the list where it is made. */}
                  {option.label}
                  {appliedAdapter && appliedAdapter.adapter.base === option.value
                    ? " (adapted)"
                    : ""}
                </option>
              ))}
              <option value={NONE_SOURCE_MODEL}>None</option>
            </select>
          </label>
        )}
        <AboutThisResult
          isOrganelle={isOrganelle}
          displayedObjects={displayedObjects}
          pixelSize={pixelSize}
          objectsPixelSize={objectsPixelSize}
          currentSegmentation={currentSegmentation}
          isBusy={isBusy}
          isComplete={isComplete}
          onOpenClearRerun={onClearMislabeledObjects ? clearRerun.open : undefined}
        />
        <RunControls
          isOrganelle={isOrganelle}
          runScaleMismatch={runScaleMismatch}
          applyFullDisabled={applyFullDisabled}
          isComplete={isComplete}
          modelBlocked={modelBlocked}
          isBusy={isBusy}
          runTargetLabel={runTargetLabel}
          modelRunnability={modelRunnability}
          showFullImageProgress={showFullImageProgress}
          fullImageProgress={fullImageProgress}
          showActiveRoiRun={showActiveRoiRun}
          applyActiveRoiDisabled={applyFullDisabled}
          onApplyFullImage={onApplyFullImage}
          onApplyActiveRoi={onApplyActiveRoi}
          onRequestRunConfirm={runScaleConfirm.open}
        />
        <HeaderNotices
          currentSegmentation={currentSegmentation}
          isOrganelle={isOrganelle}
          appliedAdapter={appliedAdapter}
          runTargetLabel={runTargetLabel}
          isComplete={isComplete}
          modelBlocked={modelBlocked}
          modelRunnability={modelRunnability}
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

      <RunScaleMismatchDialog
        isOpen={runScaleConfirm.isOpen}
        runScaleMismatch={runScaleMismatch}
        runTargetLabel={runTargetLabel}
        onApplyFullImage={onApplyFullImage}
        onClose={runScaleConfirm.close}
      />

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
