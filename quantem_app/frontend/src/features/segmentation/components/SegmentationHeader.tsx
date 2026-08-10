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
 */

import { useCallback, useEffect, useState } from "react";
import {
  NONE_SOURCE_MODEL,
  describeDisplayedObjects,
  resolveSourceModelLabel,
} from "@/features/segmentation/components/segmentationHeaderProvenance";
import {
  discardBreakdown,
  discardBySourceModel,
  pluraliseObjects,
} from "@/features/segmentation/components/segmentationCompletionLoss";
import { getSegmentationCompletionPreview } from "@/shared/api/segmentations/annotations";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { SegmentationCompletionPreview } from "@/shared/types";
import { resolvePixelSize } from "@/shared/pixelSize";
import { describeObjectsPixelSize } from "@/shared/objectsPixelSize";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { PixelSizeBadge } from "@/shared/ui/PixelSize";
import {
  formatThreshold,
  type AppliedAdapterState,
} from "@/features/models/appliedAdapter";
import { UncalibratedScaleWarning } from "@/features/models/components/UncalibratedScaleWarning";
import type { ScaleMismatch } from "@/features/models/scaleMismatch";
import type {
  AssetDetail,
  ImageSegmentation,
  SegmentationRunNotice,
  SourceModelOption,
} from "@/shared/types";
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
  fullImageActive = false,
  fullImageProgress = null,
  onBackToHome,
  onSegmentationChange,
  onSourceModelChange,
  onToggleSegmentationComplete,
  onApplyFullImage,
  isApplyingFull = false,
  hasQueuedOrRunningOrganelleTask = false,
  modelRunnability = UNKNOWN_RUNNABILITY,
  appliedAdapter = null,
  runScaleMismatch = null,
  onClearMislabeledObjects,
}: SegmentationHeaderProps) {
  const [completeConfirmOpen, setCompleteConfirmOpen] = useState(false);
  const [runConfirmOpen, setRunConfirmOpen] = useState(false);
  const [clearRerunConfirmOpen, setClearRerunConfirmOpen] = useState(false);
  const [clearingRerun, setClearingRerun] = useState(false);
  const [clearRerunError, setClearRerunError] = useState<string | null>(null);
  const [discardUnconfirmed, setDiscardUnconfirmed] = useState(false);
  const [preview, setPreview] = useState<SegmentationCompletionPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewNonce, setPreviewNonce] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const isOrganelle = Boolean(currentSegmentation?.config);
  const isBusy = isApplyingFull || hasQueuedOrRunningOrganelleTask;
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

  /**
   * What `POST .../labels/clear` will actually delete, from the counts the
   * screen already holds.
   *
   * The endpoint deletes by label state alone — every CONFIRMED and every
   * EXCLUDED object, whatever produced it — so hand-drawn objects (stored as
   * CONFIRMED, source "manual") go with it, and the dialog must say so rather
   * than promise they survive. `segment_counts` can be one poll behind an
   * in-flight edit, which is why the sentences say "currently" and the
   * endpoint's own `deleted` count is the fact of record; unlike Mark Done
   * there is no acknowledged-count contract to hold the dialog to.
   */
  const counts = currentSegmentation?.segment_counts;
  const clearConfirmedCount = counts?.CONFIRMED ?? 0;
  const clearExcludedCount = counts?.EXCLUDED ?? 0;
  const clearDoomedCount = clearConfirmedCount + clearExcludedCount;
  const clearSurvivorCount = (counts?.CANDIDATE ?? 0) + (counts?.INFERRED ?? 0);
  // No per-source arithmetic here: `segment_counts_by_source_model` reports
  // every source's CONFIRMED as the all-bundles total (a confirmed object is
  // a member of every bundle), so "how many of the doomed are hand-drawn" is
  // not a number this payload can honestly give. The dialog says hand-drawn
  // objects go too as a categorical fact instead of quoting a wrong count.

  const openClearRerunConfirm = useCallback(() => {
    setClearRerunError(null);
    setClearRerunConfirmOpen(true);
  }, []);

  /**
   * Same contract as `confirmCompletion`: the dialog closes only when the
   * server agreed. The refusal worth keeping open for is the completion
   * lock's 409 — the segmentation was marked done in another tab — and
   * closing on it would replace the explanation with silence.
   */
  const confirmClearRerun = useCallback(async () => {
    if (!onClearMislabeledObjects) return;
    setClearingRerun(true);
    setClearRerunError(null);
    try {
      await onClearMislabeledObjects();
      setClearRerunConfirmOpen(false);
    } catch (error) {
      setClearRerunError(
        extractApiErrorMessage(
          error,
          "The objects could not be deleted; nothing was changed."
        )
      );
    } finally {
      setClearingRerun(false);
    }
  }, [onClearMislabeledObjects]);

  /**
   * Read the count fresh, every time the dialog opens.
   *
   * Not from `segment_counts` on the segmentation payload: that can be a poll
   * behind, and `POST` compares the acknowledged count against a fresh read and
   * returns 409 on a mismatch. A dialog quoting a stale number would simply
   * fail at the last click.
   */
  const openCompleteConfirm = useCallback(() => {
    setPreview(null);
    setPreviewError(null);
    setSubmitError(null);
    setDiscardUnconfirmed(false);
    setCompleteConfirmOpen(true);
    setPreviewNonce((current) => current + 1);
  }, []);

  const segmentationId = currentSegmentation?.id ?? null;
  useEffect(() => {
    if (!completeConfirmOpen || !segmentationId) return undefined;
    let cancelled = false;
    void getSegmentationCompletionPreview(segmentationId)
      .then((next) => {
        if (!cancelled) setPreview(next);
      })
      .catch((error) => {
        if (cancelled) return;
        setPreviewError(
          extractApiErrorMessage(
            error,
            "The number of objects this would delete could not be read."
          )
        );
      });
    return () => {
      cancelled = true;
    };
  }, [completeConfirmOpen, segmentationId, previewNonce]);

  const discardCount = preview?.discard_count ?? null;

  // A refreshed preview can leave nothing to discard -- someone else completed
  // the segmentation while this dialog was open. The tick box is then gone, so
  // its state has to go with it or the confirm button offers to "delete 0
  // objects".
  useEffect(() => {
    if (discardCount === 0) setDiscardUnconfirmed(false);
  }, [discardCount]);

  /**
   * Confirming keeps the dialog open until the server agrees.
   *
   * The refusal this handles is the `409`: the count moved while the dialog was
   * open, usually because an inference run finished, and nothing was deleted.
   * Closing on click would have replaced that with silence, which is the exact
   * failure the endpoint's check exists to prevent.
   */
  const confirmCompletion = useCallback(async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      await onToggleSegmentationComplete(
        discardUnconfirmed && discardCount !== null
          ? { discardUnconfirmed: true, acknowledgedDiscardCount: discardCount }
          : undefined
      );
      setCompleteConfirmOpen(false);
    } catch (error) {
      setSubmitError(
        extractApiErrorMessage(error, "The segmentation could not be marked done.")
      );
      // Re-read, so the number beside the tick box is the one that would now go.
      setPreviewNonce((current) => current + 1);
    } finally {
      setSubmitting(false);
    }
  }, [discardCount, discardUnconfirmed, onToggleSegmentationComplete]);

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
                {seg.segmentation_type.long_name}
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
        {isOrganelle && (
          <span
            className={`source-model-provenance ${displayedObjects.tone}`}
            title={displayedObjects.detail}
            data-testid="displayed-objects-provenance"
          >
            {displayedObjects.summary}
          </span>
        )}
        <PixelSizeBadge resolved={pixelSize} />
        {objectsPixelSize && (
          // Beside the badge it qualifies, because the two read as one claim:
          // the badge says what the image records, this says the objects were
          // not made at it. Same summary-plus-tooltip shape as the provenance
          // chip; the full sentences also stand as body text on the Analysis
          // screen, before a run is spent.
          <span
            className="source-model-provenance warning"
            role="status"
            title={`${objectsPixelSize.detail} ${objectsPixelSize.consequence}`}
            data-testid="objects-pixel-size-warning"
          >
            {objectsPixelSize.summary}
          </span>
        )}
        {objectsPixelSize && isOrganelle && onClearMislabeledObjects && (
          // The way out of the state the chip describes, beside the chip that
          // describes it. Re-running alone is a no-op — a new candidate
          // landing on a confirmed or excluded object is dropped, which is
          // what protects proofreading — so recovery is delete-then-re-run,
          // and it asks first.
          <button
            type="button"
            className="header-clear-rerun-button"
            data-testid="clear-rerun-button"
            onClick={openClearRerunConfirm}
            disabled={isBusy || isComplete || !currentSegmentation}
            title={
              isComplete
                ? "This segmentation is locked. Unlock it first."
                : isBusy
                  ? "Processing in progress"
                  : "Delete the objects made before the pixel size was set, then run the model again at it"
            }
          >
            Discard objects and re-run…
          </button>
        )}
        {isOrganelle && (
          <div className="header-action-buttons">
            <button
              className="apply-full-button"
              onClick={() => {
                // Same inference pass the create dialog gates, so the same
                // question gets asked. Nothing else about the run has changed,
                // so a calibrated image still starts on the first click.
                if (runScaleMismatch) {
                  setRunConfirmOpen(true);
                  return;
                }
                onApplyFullImage?.();
              }}
              disabled={applyFullDisabled}
              title={
                isComplete
                  ? "This segmentation is locked. Unlock it to run again."
                  : modelBlocked
                    ? `${runTargetLabel} cannot run here: ${modelRunnability.reason ?? "no reason given"}`
                    : isBusy
                      ? "Processing in progress"
                      : `Run ${runTargetLabel} over the full image`
              }
            >
              {showFullImageProgress ? "Running..." : "Run Full Segmentation"}
            </button>
            {showFullImageProgress && (
              <span className="apply-full-progress">
                <span className="apply-full-spinner" aria-hidden />
                {fullImageProgress !== null ? `${fullImageProgress}%` : "Starting"}
              </span>
            )}
          </div>
        )}
        {currentSegmentation?.status_stage === "FAILED" ? (
          // Not gated on `isOrganelle`, and not left to the chip's tooltip. A
          // failed run is the one state where every other number on this
          // header belongs to a different run, and `status_error` is the only
          // text that says whether the worker died, the user cancelled, or the
          // job was removed before it started.
          <FailedRunNotice
            error={currentSegmentation.status_error}
            objectCount={countAllObjects(currentSegmentation)}
          />
        ) : currentSegmentation?.status_error?.trim() ? (
          // A non-empty error on a stage that is not FAILED is the queue's
          // note about the *newest* attempt. The retry path writes
          // "Attempt N of M failed; retrying automatically. <error>" onto the
          // segmentation after every failed attempt without touching the
          // stage, and the abandoned-run repair leaves its own sentence the
          // same way -- so this header must render `status_error` whenever it
          // is present, or it goes back to showing an older error (or
          // nothing) while newer failures accrue only in Tasks & Queues. The
          // words are the server's, verbatim: they are the only text that
          // says which attempt failed and whether another is coming, and a
          // successful attempt clears the field.
          <div
            className="header-failed-notice"
            role="status"
            data-testid="latest-attempt-notice"
          >
            {currentSegmentation.status_error.trim()}
          </div>
        ) : null}
        {isOrganelle && currentSegmentation?.run_notice && (
          // Directly under the button it is about. Running again is the obvious
          // response to an empty screen and it is the wrong one: the run
          // already finished, and the server's own finding says to check the
          // pixel size before the threshold.
          <RunNotice notice={currentSegmentation.run_notice} />
        )}
        {isOrganelle && appliedAdapter && (
          <AppliedAdapterNotice
            state={appliedAdapter}
            selectedLabel={runTargetLabel}
          />
        )}
        {isComplete && (
          // The same reasoning as the blocked-model notice: a disabled control
          // with no sentence beside it reads as a bug, and a tooltip is
          // unreachable by keyboard. This is also the only place the app says
          // how to get the segmentation back.
          <span className="header-locked-notice" role="status">
            <strong>This segmentation is locked.</strong> It was marked done, so
            objects cannot be added, edited or removed and no new run can be
            started. Use "Unlock segmentation" to change it again.
          </span>
        )}
        {isOrganelle && !isComplete && modelBlocked && (
          // The one place the app can say, before the click, why a run will not
          // happen. A tooltip alone is not enough: it is unreachable by
          // keyboard and the disabled button otherwise looks like a bug.
          <span className="header-model-blocked" role="status">
            <strong>{runTargetLabel} cannot run here.</strong>{" "}
            {modelRunnability.reason}{" "}
            <a className="header-model-blocked-link" href="#/models">
              Models
            </a>
          </span>
        )}
        <a
          className="header-route-link"
          href={`#/assets/${image.id}/analysis${
            currentSegmentation ? `?seg=${currentSegmentation.id}` : ""
          }`}
        >
          Analysis
        </a>
        <a
          className="header-route-link"
          href={`#/assets/${image.id}/adapt${
            currentSegmentation ? `?seg=${currentSegmentation.id}` : ""
          }`}
        >
          Adapt model
        </a>
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
            openCompleteConfirm();
          }}
          disabled={!currentSegmentation}
        >
          {isComplete ? "Unlock segmentation" : "Mark Image Done"}
        </button>
      </div>

      {/* The other door into an uncalibrated run. Creating a segmentation
          opened a written confirmation naming the pack and the resolution it
          was trained at; this button queues the identical pass and used to fire
          on the first click with nothing said. Same check, same sentence. */}
      <ConfirmDialog
        isOpen={runConfirmOpen && runScaleMismatch !== null}
        title={`Run ${runTargetLabel} without a pixel size?`}
        message={
          "This queues one inference pass over the whole image, and the objects " +
          "it produces will carry no physical scale."
        }
        details={
          runScaleMismatch ? (
            <UncalibratedScaleWarning mismatch={runScaleMismatch} />
          ) : null
        }
        detailsTone="warning"
        confirmText="Run uncalibrated"
        cancelText="Cancel"
        onConfirm={() => {
          setRunConfirmOpen(false);
          onApplyFullImage?.();
        }}
        onCancel={() => setRunConfirmOpen(false)}
      />

      {/* Recovery from calibrated-after-the-fact, to the Mark-Done standard:
          name what is discarded before anything is. The wording is the
          endpoint's truth, not the convenient version — labels/clear deletes
          by label state, so hand-drawn objects (stored as confirmed) are
          deleted too, and only unreviewed model candidates survive. */}
      <ConfirmDialog
        isOpen={clearRerunConfirmOpen}
        title="Delete these objects and re-run at the corrected pixel size?"
        message={
          "These objects were made before the image's pixel size was set, so " +
          "their measurements cannot be converted to physical units. " +
          "Deleting them and running again produces a set measured at the " +
          "pixel size the image records now."
        }
        details={
          <>
            {clearRerunError ? (
              <p className="segmentation-discard-refusal" role="alert">
                {clearRerunError}
              </p>
            ) : null}
            <p>
              This deletes every reviewed object on this segmentation —
              currently {pluraliseObjects(clearConfirmedCount)} confirmed and{" "}
              {clearExcludedCount} rejected. That includes any drawn by hand:
              hand-drawn objects are stored as confirmed, and they are{" "}
              <strong>not</strong> spared. Nothing is archived, so this cannot
              be undone.
              {clearSurvivorCount > 0
                ? ` ${pluraliseObjects(clearSurvivorCount)} nobody reviewed ${
                    clearSurvivorCount === 1 ? "is" : "are"
                  } kept.`
                : ""}
            </p>
            {clearExcludedCount > 0 ? (
              <p>
                Rejections are also ground truth: "Adapt a model" trains
                against them as negative examples, and deleting them shrinks
                that record.
              </p>
            ) : null}
            <p>
              Then one full inference pass over the image is queued with{" "}
              {runTargetLabel}.
            </p>
          </>
        }
        detailsTone="warning"
        confirmText={
          clearingRerun
            ? "Working…"
            : `Delete ${pluraliseObjects(clearDoomedCount)} and re-run`
        }
        cancelText="Cancel"
        onConfirm={() => {
          void confirmClearRerun();
        }}
        onCancel={() => setClearRerunConfirmOpen(false)}
      />

      {/* Creating a segmentation -- cheap and reversible -- already gets a
          written confirmation. Throwing away a run's output, which is neither,
          was the most prominent button on the screen and fired on the first
          click. It is now two separate decisions: locking (what the button
          says) and discarding (a tick box that starts off, quotes a count read
          live from the server, and says whether it can be undone). */}
      <ConfirmDialog
        isOpen={completeConfirmOpen}
        title="Mark this image done?"
        message={
          "Marking it done locks the segmentation. Nothing is deleted unless you ask for it below."
        }
        details={
          <CompletionNotice
            preview={preview}
            previewError={previewError}
            submitError={submitError}
            sourceModelOptions={sourceModelOptions}
            discardUnconfirmed={discardUnconfirmed}
            onDiscardChange={setDiscardUnconfirmed}
          />
        }
        detailsTone={discardUnconfirmed || submitError ? "warning" : "default"}
        confirmText={
          submitting
            ? "Working…"
            : discardUnconfirmed && discardCount !== null
              ? `Mark done and delete ${pluraliseObjects(discardCount)}`
              : "Mark done"
        }
        cancelText="Cancel"
        onConfirm={() => {
          void confirmCompletion();
        }}
        onCancel={() => setCompleteConfirmOpen(false)}
      />
    </header>
  );
}

/** Every object on the segmentation, whatever its label state. */
function countAllObjects(segmentation: ImageSegmentation): number {
  const counts = segmentation.segment_counts;
  if (!counts) return 0;
  return Object.values(counts).reduce(
    (sum, value) => sum + (typeof value === "number" ? value : 0),
    0
  );
}

/**
 * A run that stopped without saving anything, said out loud.
 *
 * `status_error` was reaching no screen at all. The stage went FAILED, the
 * message was written, and the labeling header carried on describing the
 * objects from the *previous* run in its ordinary neutral chip -- so a
 * cancelled ER re-run read "190 confirmed of 190 from QuantEM" and a re-run at
 * a corrected pixel size left the wrongly-scaled objects looking finished.
 *
 * The object count is repeated here rather than left to the chip because the
 * sentence that matters is not "the run failed" on its own; it is that the
 * objects still on screen predate it. `status_error` is reproduced verbatim
 * (`quantem.jobs.failure_reconcile` writes it) since it is the only text that
 * separates a dead worker from a cancellation from a job removed from the
 * queue, and each of those wants a different response.
 */
function FailedRunNotice({
  error,
  objectCount,
}: {
  error?: string | null;
  objectCount: number;
}) {
  const reason = error?.trim();
  return (
    <div className="header-failed-notice" role="status">
      <strong>The last run on this segmentation failed.</strong>{" "}
      {reason || "The server recorded no reason for it."}{" "}
      {objectCount === 0
        ? "It saved no objects, and there are none here from an earlier run."
        : `It saved no objects: the ${objectCount} shown here ${
            objectCount === 1 ? "was" : "were"
          } already on this segmentation before it started.`}
    </div>
  );
}

/**
 * The server's finding about a run whose stage does not tell the whole story.
 *
 * This is the one thing on the screen that distinguishes "not run yet" from
 * "ran and found nothing", and the two want opposite actions. It is rendered as
 * text rather than as a tooltip for the reason the locked and blocked notices
 * beside it are: a tooltip is unreachable by keyboard, invisible unless you
 * happen to hover the right ten pixels, and this is a numbered list of things
 * to check, not a label.
 *
 * The words are the server's, verbatim. The pixel size leads because the server
 * ranked it first and it is the input that turns a working model into one that
 * finds nothing -- lowering the threshold on a wrongly-scaled run produces
 * different rubbish, not the missing objects. Rewriting any of it here would
 * put a second, drifting copy of that advice in the application.
 */
function RunNotice({ notice }: { notice: SegmentationRunNotice }) {
  return (
    <div className="header-run-notice" role="status">
      <strong>{notice.message}</strong>
      {notice.next_steps.length > 0 && (
        <ol className="header-run-notice-steps">
          {notice.next_steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      )}
    </div>
  );
}

/**
 * What locking costs, and the separate, opt-in question of what to delete.
 *
 * The tick box starts off and the count beside it is read live, because the
 * endpoint refuses an acknowledged count that no longer matches -- usually
 * because an inference run finished while the dialog was open.
 *
 * The doomed set is also *described*, not just counted. `discardable_queryset`
 * is "everything that is not CONFIRMED", which puts candidates nobody has
 * opened in the same bucket as objects somebody opened and rejected. The
 * arithmetic is the same; the loss is not. A rejection is the record of a
 * review, and it is ground truth: `fetchGroundTruthProvenance` feeds EXCLUDED
 * objects to the fine-tuning wizard as negative examples.
 */
function CompletionNotice({
  preview,
  previewError,
  submitError,
  sourceModelOptions,
  discardUnconfirmed,
  onDiscardChange,
}: {
  preview: SegmentationCompletionPreview | null;
  previewError: string | null;
  submitError: string | null;
  sourceModelOptions: SourceModelOption[];
  discardUnconfirmed: boolean;
  onDiscardChange: (next: boolean) => void;
}) {
  const refusal = submitError ? (
    <p className="segmentation-discard-refusal" role="alert">
      {submitError}
    </p>
  ) : null;

  if (previewError) {
    return (
      <>
        {refusal}
        <p>
          {previewError} Marking it done will still lock the segmentation and
          keep every object.
        </p>
      </>
    );
  }

  if (!preview) {
    return (
      <>
        {refusal}
        <p>Counting what is on this segmentation…</p>
      </>
    );
  }

  const { discard_count: discardCount, confirmed_count: confirmedCount } = preview;
  const bySource = discardBySourceModel(preview, sourceModelOptions);
  const { rejected, neverReviewed } = discardBreakdown(preview);

  if (discardCount === 0) {
    return (
      <>
        {refusal}
        <p>
          Nothing here is unconfirmed, so there is nothing to delete.{" "}
          {confirmedCount > 0
            ? `All ${pluraliseObjects(confirmedCount)} are kept.`
            : "This segmentation has no objects yet."}
        </p>
      </>
    );
  }

  return (
    <>
      {refusal}
      <p>
        This segmentation holds {pluraliseObjects(discardCount)} nobody has
        confirmed
        {bySource.length > 0 ? (
          <>
            {" "}
            —{" "}
            {bySource
              .map((entry) => `${entry.count} from ${entry.label}`)
              .join(", ")}
          </>
        ) : null}
        , and {pluraliseObjects(confirmedCount)} you confirmed.
      </p>
      {/* "Nobody confirmed" is arithmetically right and descriptively wrong:
          it covers candidates nobody has opened *and* objects somebody opened
          and rejected. Same effect on the count, different thing to lose. */}
      {rejected > 0 ? (
        <p>
          Of those, {pluraliseObjects(neverReviewed)}{" "}
          {neverReviewed === 1 ? "was" : "were"} never reviewed and{" "}
          {rejected === 1 ? "1 is one you rejected" : `${rejected} are ones you rejected`}.
          A rejection is the record of a review, not a leftover, and "Adapt a
          model" trains against rejections as negative examples — deleting them
          shrinks the ground truth for the next adaptation.
        </p>
      ) : null}
      <label className="segmentation-discard-choice">
        <input
          type="checkbox"
          checked={discardUnconfirmed}
          onChange={(event) => onDiscardChange(event.target.checked)}
        />
        <span>
          {rejected > 0
            ? `Also delete all ${pluraliseObjects(discardCount)}: the ${neverReviewed} never reviewed and the ${rejected} you rejected.`
            : `Also delete the ${pluraliseObjects(discardCount)} nobody confirmed.`}
        </span>
      </label>
      {discardUnconfirmed ? (
        <p>
          {preview.restorable ? (
            <>
              They are archived first, so "Unlock segmentation" puts them back.
              Leave the box unticked if you only meant to lock the image.
            </>
          ) : (
            <>
              <strong>
                This is more than {preview.archive_max_objects} objects, so it
                cannot be archived and "Unlock segmentation" will not bring them
                back.
              </strong>{" "}
              Getting them again means another full segmentation run.
            </>
          )}
        </p>
      ) : null}
    </>
  );
}

/**
 * That an adapted model is in play, on the screen where the run is started.
 *
 * Two states, and the second is the one worth the pixels: an adapter applied to
 * this segmentation is only used when the selected source model is the one it
 * was fitted on. Pick anything else and `apply_active_adapter` falls back to
 * the released pack at its published threshold, with nothing on screen to say
 * so -- your numbers change and the header still reads the same.
 */
function AppliedAdapterNotice({
  state,
  selectedLabel,
}: {
  state: AppliedAdapterState;
  selectedLabel: string;
}) {
  const { adapter, active, publishedThreshold, trainedHead } = state;
  const calibrated = formatThreshold(adapter.calibrated_threshold);
  const published = formatThreshold(publishedThreshold);
  const name = adapter.name || adapter.base;

  if (!active) {
    return (
      <span className="header-adapter-notice is-inactive" role="status">
        <strong>Adapter not in use.</strong> "{name}" was fitted on{" "}
        {adapter.base}, and this run is set to {selectedLabel}, so it will use
        the published model
        {published ? ` at threshold ${published}` : ""}.
      </span>
    );
  }

  return (
    <span className="header-adapter-notice" role="status">
      <strong>Adapted model: {name}.</strong> Run Full Segmentation will use{" "}
      {trainedHead ? "your fine-tuned head" : "your calibration"}
      {calibrated ? ` at threshold ${calibrated}` : ""}
      {published && calibrated && published !== calibrated
        ? `, not the published ${published}`
        : ""}
      .
    </span>
  );
}
