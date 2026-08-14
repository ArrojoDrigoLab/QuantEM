/**
 * "About this result": where the objects on screen came from.
 *
 * These are the chips a reader checks before believing a number -- what
 * produced the displayed objects and whether the objects predate the current
 * calibration -- plus the one
 * recovery those chips point at, the two route links that leave for the screens
 * where the numbers are used, and the Fine-Tune trigger for this organelle.
 *
 * Both exports render fragments, so the header's `.header-controls` row keeps
 * the children it had, in the order it had them, with the Fine-Tune button
 * appended after the links rather than inserted among them. The chips and
 * the links are two exports rather than one component because they are not
 * adjacent in the row: the notice stack sits between them, and collapsing them
 * into one component would reorder the DOM.
 */

import { FineTuneOrganelleButton } from "@/features/finetune/FineTuneOrganelleButton";
import type { DisplayedObjectsDescription } from "@/features/segmentation/components/segmentationHeaderProvenance";
import type { ObjectsPixelSizeWarning } from "@/shared/objectsPixelSize";
import type { AssetDetail, ImageSegmentation } from "@/shared/types";

/**
 * The provenance chips and the way out of the state the warning chip describes.
 */
export function AboutThisResult({
  isOrganelle,
  displayedObjects,
  objectsPixelSize,
  currentSegmentation,
  isBusy,
  isComplete,
  onOpenClearRerun,
}: {
  isOrganelle: boolean;
  displayedObjects: DisplayedObjectsDescription;
  objectsPixelSize: ObjectsPixelSizeWarning | null;
  currentSegmentation: ImageSegmentation | null;
  isBusy: boolean;
  isComplete: boolean;
  /**
   * Opens the delete-and-re-run confirmation. Undefined when the screen has no
   * clear handler at all, which is what decides whether the trigger exists.
   */
  onOpenClearRerun?: () => void;
}) {
  const finalNote = currentSegmentation?.final_result_provenance;
  const finalLevel =
    finalNote?.final_level === null || finalNote?.final_level === undefined
      ? finalNote?.final_level_kind ?? "unknown"
      : `${finalNote.final_level_kind.replaceAll("_", " ")} ${finalNote.final_level.toFixed(3)}`;
  const finalAdapter = finalNote?.adapter_identifier ?? "unknown";
  const finalDetail = finalNote
    ? `Final result: ${finalNote.model_identifier}; QuantEM ${finalNote.quantem_version}; ${finalLevel}; adapter ${finalAdapter}. Finalized ${finalNote.finalized_at}.`
    : "";
  return (
    <>
      {isOrganelle && (
        <span
          className={`source-model-provenance ${displayedObjects.tone}`}
          title={displayedObjects.detail}
          data-testid="displayed-objects-provenance"
        >
          {displayedObjects.summary}
        </span>
      )}
      {isComplete && finalNote && (
        <span
          className="source-model-provenance"
          data-testid="final-result-provenance"
          title={finalDetail}
        >
          Final: {finalNote.model_identifier} · {finalLevel}
          {finalAdapter !== "unknown" && finalAdapter !== "manual"
            ? ` · adapter ${finalAdapter}`
            : ""}
        </span>
      )}
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
      {objectsPixelSize && isOrganelle && onOpenClearRerun && (
        // The way out of the state the chip describes, beside the chip that
        // describes it. Re-running alone is a no-op — a new candidate
        // rejected proposal is dropped; confirmed geometry stays above the
        // preview and is resolved only when the preview is confirmed, which is
        // what protects proofreading — so recovery is delete-then-re-run,
        // and it asks first.
        <button
          type="button"
          className="header-clear-rerun-button"
          data-testid="clear-rerun-button"
          onClick={onOpenClearRerun}
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
    </>
  );
}

/**
 * The ways off this screen: two links, and the fine-tune trigger for the
 * organelle the header is describing.
 *
 * The links carry `?seg=` so the destination opens on the segmentation the
 * header is describing rather than on whatever that screen would have picked.
 * The Fine-Tune button opens a dialog in place instead, because what it needs
 * next is a *scope* — which datasets and images to train across — and that is a
 * question about the library, not about this image.
 */
export function HeaderRouteLinks({
  image,
  currentSegmentation,
  fineTuneEligibilityRevision,
}: {
  image: AssetDetail;
  currentSegmentation: ImageSegmentation | null;
  fineTuneEligibilityRevision?: string;
}) {
  return (
    <>
      <a
        className="header-route-link"
        href={`#/assets/${image.id}/analysis${
          currentSegmentation ? `?seg=${currentSegmentation.id}` : ""
        }`}
      >
        Analysis
      </a>
      <FineTuneOrganelleButton
        image={image}
        currentSegmentation={currentSegmentation}
        eligibilityRevision={fineTuneEligibilityRevision}
      />
    </>
  );
}
