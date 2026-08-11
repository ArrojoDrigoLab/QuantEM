/**
 * The two destructive confirmations the labeling header owns, and their state.
 *
 * Moved out of `SegmentationHeader.tsx` unchanged. Both are here rather than
 * beside the buttons that open them because they render at the end of the
 * `<header>`, not in the controls row, and because between them they were about
 * a third of a 966-line file that five packages wanted to edit at once.
 *
 * Each dialog's state lives in a hook beside this file, in
 * `completionDialogState.ts`, so the header holds the trigger and nothing else:
 *
 * * `useCompletionConfirm` — Mark Image Done, whose tick box quotes a count
 *   read live from the server because `POST` refuses a stale one;
 * * `useClearRerunConfirm` — delete every reviewed object and run again, the
 *   recovery for an image calibrated after its objects were made.
 */

import {
  discardBreakdown,
  discardBySourceModel,
  pluraliseObjects,
} from "@/features/segmentation/components/segmentationCompletionLoss";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import type {
  ClearRerunConfirmState,
  CompletionConfirmState,
} from "@/features/segmentation/components/header/completionDialogState";
import type {
  ImageSegmentation,
  SegmentationCompletionPreview,
  SourceModelOption,
} from "@/shared/types";

/**
 * Recovery from calibrated-after-the-fact, to the Mark-Done standard: name what
 * is discarded before anything is.
 *
 * The wording is the endpoint's truth, not the convenient version —
 * `labels/clear` deletes by label state, so hand-drawn objects (stored as
 * confirmed) are deleted too, and only unreviewed model candidates survive.
 *
 * `segment_counts` can be one poll behind an in-flight edit, which is why the
 * sentences say "currently" and the endpoint's own `deleted` count is the fact
 * of record; unlike Mark Done there is no acknowledged-count contract to hold
 * the dialog to. No per-source arithmetic either:
 * `segment_counts_by_source_model` reports every source's CONFIRMED as the
 * all-bundles total (a confirmed object is a member of every bundle), so "how
 * many of the doomed are hand-drawn" is not a number this payload can honestly
 * give. The dialog says hand-drawn objects go too as a categorical fact instead
 * of quoting a wrong count.
 */
export function ClearAndRerunConfirmDialog({
  state,
  currentSegmentation,
  runTargetLabel,
}: {
  state: ClearRerunConfirmState;
  currentSegmentation: ImageSegmentation | null;
  runTargetLabel: string;
}) {
  const counts = currentSegmentation?.segment_counts;
  const clearConfirmedCount = counts?.CONFIRMED ?? 0;
  const clearExcludedCount = counts?.EXCLUDED ?? 0;
  const clearDoomedCount = clearConfirmedCount + clearExcludedCount;
  const clearSurvivorCount = (counts?.CANDIDATE ?? 0) + (counts?.INFERRED ?? 0);
  const clearRerunError = state.error;

  return (
    <ConfirmDialog
      isOpen={state.isOpen}
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
        state.clearing
          ? "Working…"
          : `Delete ${pluraliseObjects(clearDoomedCount)} and re-run`
      }
      cancelText="Cancel"
      onConfirm={() => {
        void state.confirm();
      }}
      onCancel={() => state.close()}
    />
  );
}

/**
 * Creating a segmentation -- cheap and reversible -- already gets a written
 * confirmation. Throwing away a run's output, which is neither, was the most
 * prominent button on the screen and fired on the first click. It is now two
 * separate decisions: locking (what the button says) and discarding (a tick box
 * that starts off, quotes a count read live from the server, and says whether
 * it can be undone).
 */
export function CompletionConfirmDialog({
  state,
  sourceModelOptions,
}: {
  state: CompletionConfirmState;
  sourceModelOptions: SourceModelOption[];
}) {
  const { discardCount, discardUnconfirmed, submitError, submitting } = state;
  return (
    <ConfirmDialog
      isOpen={state.isOpen}
      title="Mark this image done?"
      message={
        "Marking it done locks the segmentation. Nothing is deleted unless you ask for it below."
      }
      details={
        <CompletionNotice
          preview={state.preview}
          previewError={state.previewError}
          submitError={submitError}
          sourceModelOptions={sourceModelOptions}
          discardUnconfirmed={discardUnconfirmed}
          onDiscardChange={state.setDiscardUnconfirmed}
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
        void state.confirm();
      }}
      onCancel={() => state.close()}
    />
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
export function CompletionNotice({
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
