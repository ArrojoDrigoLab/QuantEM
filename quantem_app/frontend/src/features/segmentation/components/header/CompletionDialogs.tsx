import {
  pluraliseObjects,
} from "@/features/segmentation/components/segmentationCompletionLoss";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import type {
  ClearRerunConfirmState,
  CompletionConfirmState,
} from "@/features/segmentation/components/header/completionDialogState";
import type {
  ImageSegmentation,
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

/** Confirm that manual annotation covers the entire image. */
export function CompletionConfirmDialog({ state }: { state: CompletionConfirmState; sourceModelOptions: SourceModelOption[] }) {
  return <ConfirmDialog isOpen={state.isOpen} title="Confirm whole-image manual segmentation?"
    message="Confirm that you have manually segmented the entire image, including every region outside your ROIs. No model outputs will be included in this manual-only result."
    details={state.submitError ? <p role="alert">{state.submitError}</p> : undefined}
    confirmText={state.submitting ? "Saving…" : "Entire image manually segmented; no model outputs"}
    cancelText="Cancel" onConfirm={() => { if (!state.submitting) void state.confirm(); }} onCancel={() => state.close()} />;
}
