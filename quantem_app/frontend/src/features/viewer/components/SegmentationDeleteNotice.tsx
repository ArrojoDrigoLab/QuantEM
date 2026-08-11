/**
 * The delete-a-segmentation confirmation, and the sentences inside it.
 *
 * Moved out of `ViewerScreen.tsx` unchanged. Deleting a run's output is at
 * least as destructive as Mark Image Done's discard, so it gets the same
 * standard: live counts read when the dialog opens, refusals rendered in the
 * dialog, and the confirm button naming the number it deletes.
 */

import { pluraliseObjects } from "@/features/segmentation/components/segmentationCompletionLoss";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import type { SegmentationDeleteState } from "@/features/viewer/state/useSegmentationDelete";
import type { SegmentationDeletePreview } from "@/shared/types/segmentation";

export function SegmentationDeleteDialog({
  state,
}: {
  state: SegmentationDeleteState;
}) {
  const {
    deleteTarget,
    deletePreview,
    deletePreviewError,
    deleteSubmitError,
    deleting,
  } = state;
  return (
    <ConfirmDialog
      isOpen={deleteTarget !== null}
      title={`Delete ${deleteTarget?.segmentation_type.long_name ?? "this segmentation"} from this image?`}
      message={
        "This permanently deletes the segmentation and everything it produced. " +
        "Nothing is archived, so it cannot be undone — getting the objects " +
        "back means running the model again."
      }
      details={
        <SegmentationDeleteNotice
          preview={deletePreview}
          previewError={deletePreviewError}
          submitError={deleteSubmitError}
        />
      }
      detailsTone="warning"
      confirmText={
        deleting
          ? "Deleting…"
          : deletePreview && deletePreview.object_count > 0
            ? `Delete ${pluraliseObjects(deletePreview.object_count)} and this segmentation`
            : "Delete segmentation"
      }
      cancelText="Cancel"
      onConfirm={() => {
        void state.confirmDeleteSegmentation();
      }}
      onCancel={() => state.setDeleteTarget(null)}
    />
  );
}

/**
 * What deleting this segmentation destroys, what it keeps, and what it frees.
 *
 * The counts are the server's, read when the dialog opened; the DELETE carries
 * the object count so a run finishing mid-dialog is refused rather than
 * silently included. Analysis runs are named as *kept* because deleting the
 * numbers a paper may already cite would be the greater destruction — they
 * survive marked "segmentation deleted" and can no longer be traced back to
 * objects in the app.
 */
export function SegmentationDeleteNotice({
  preview,
  previewError,
  submitError,
}: {
  preview: SegmentationDeletePreview | null;
  previewError: string | null;
  submitError: string | null;
}) {
  const refusal = submitError ? (
    // Same class as Mark Done's refusal box — styled in ConfirmDialog.css.
    <p className="segmentation-discard-refusal" role="alert">
      {submitError}
    </p>
  ) : null;

  if (previewError) {
    return (
      <>
        {refusal}
        <p>
          {previewError} Deleting will still remove every object, overlay
          raster, probability map and adapted model this segmentation holds.
        </p>
      </>
    );
  }

  if (!preview) {
    return (
      <>
        {refusal}
        <p>Counting what this segmentation holds…</p>
      </>
    );
  }

  const confirmed = preview.objects_by_label_state.CONFIRMED ?? 0;
  const excluded = preview.objects_by_label_state.EXCLUDED ?? 0;
  const unreviewed = Math.max(preview.object_count - confirmed - excluded, 0);

  const alsoDeleted: string[] = [];
  if (preview.overlay_count > 0) {
    alsoDeleted.push(
      `${preview.overlay_count} overlay raster${preview.overlay_count === 1 ? "" : "s"}`
    );
  }
  if (preview.probability_map_count > 0) {
    alsoDeleted.push(
      `${preview.probability_map_count} probability map${
        preview.probability_map_count === 1 ? "" : "s"
      }`
    );
  }
  if (preview.adapter_count > 0) {
    alsoDeleted.push(
      `${preview.adapter_count} adapted model${
        preview.adapter_count === 1 ? "" : "s"
      } (including any trained weights)`
    );
  }

  return (
    <>
      {refusal}
      {preview.locked ? (
        <p>
          <strong>This segmentation is locked.</strong> It was marked done, and
          the server refuses to delete it in that state. Unlock it first
          ("Unlock segmentation" on the labeling screen), then delete.
        </p>
      ) : null}
      <p>
        {preview.object_count === 0
          ? "This segmentation holds no objects"
          : `This deletes all ${pluraliseObjects(preview.object_count)} on this segmentation — ${confirmed} confirmed, ${excluded} rejected and ${unreviewed} nobody reviewed`}
        {alsoDeleted.length > 0
          ? `${preview.object_count === 0 ? ", but deleting it removes" : " — together with"} its ${alsoDeleted.join(", ")}`
          : ""}
        .
      </p>
      {excluded > 0 ? (
        <p>
          Rejections are ground truth: "Adapt a model" trains against them as
          negative examples, and deleting them deletes that record.
        </p>
      ) : null}
      {preview.analysis_run_count > 0 ? (
        <p>
          The {preview.analysis_run_count} analysis run
          {preview.analysis_run_count === 1 ? "" : "s"} made from it{" "}
          {preview.analysis_run_count === 1 ? "is" : "are"} <strong>kept</strong>,
          with {preview.analysis_run_count === 1 ? "its" : "their"} export
          bundle{preview.analysis_run_count === 1 ? "" : "s"}: the numbers are
          the record of an analysis that happened. They are marked "segmentation
          deleted" and can no longer be traced back to objects in the app.
        </p>
      ) : null}
      <p>
        Afterwards the {preview.segmentation_type} preset returns to "Add
        segmentation", so it can be recreated — recreating it queues a fresh
        model run.
      </p>
    </>
  );
}
