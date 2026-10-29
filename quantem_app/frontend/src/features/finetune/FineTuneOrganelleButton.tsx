/**
 * The labeling view's way in, for the organelle on screen.
 *
 * Always opens the dialog so users can choose other annotations or apply a
 * saved model. The preview informs the tooltip; training eligibility is
 * enforced when submitting the selected scope.
 */

import { useEffect, useState } from "react";
import { previewFineTuneScope } from "@/shared/api/finetune";
import { FineTuneDialog } from "@/features/finetune/FineTuneDialog";
import type { AssetDetail, ImageSegmentation } from "@/shared/types";
import type { FineTuneAppliedImageEvent } from "@/shared/types/finetune";
import "./FineTuneOrganelleButton.css";

export function FineTuneOrganelleButton({
  image,
  currentSegmentation,
  eligibilityRevision = "",
  initialBaseModel = null,
  initialAdapterId,
  onAppliedImageCompleted,
}: {
  image: AssetDetail;
  currentSegmentation: ImageSegmentation | null;
  eligibilityRevision?: string;
  /** Released pack currently selected in the labeling header. */
  initialBaseModel?: string | null;
  initialAdapterId?: string | null;
  onAppliedImageCompleted?: (event: FineTuneAppliedImageEvent) => void;
}) {
  const [open, setOpen] = useState(false);
  const [annotationCount, setAnnotationCount] = useState<number | null>(null);
  const segmentationTypeId = currentSegmentation?.segmentation_type?.id ?? null;

  useEffect(() => {
    if (!segmentationTypeId) {
      setAnnotationCount(null);
      return undefined;
    }
    let cancelled = false;
    setAnnotationCount(null);
    void previewFineTuneScope({
      segmentation_type: segmentationTypeId,
      asset_ids: [image.id],
      dataset_ids: [],
    })
      .then((preview) => {
        if (!cancelled) setAnnotationCount(preview.annotation_count);
      })
      .catch(() => {
        // No answer is not "zero annotations", but it is not permission to
        // start a run either. The button stays disabled and says so.
        if (!cancelled) setAnnotationCount(null);
      });
    return () => {
      cancelled = true;
    };
  }, [eligibilityRevision, segmentationTypeId, image.id]);

  const organelle =
    currentSegmentation?.segmentation_type?.short_name ||
    currentSegmentation?.segmentation_type?.long_name ||
    "this organelle";
  const enabled = (annotationCount ?? 0) > 0;
  const tooltip = enabled
    ? `Train a ${organelle.toLowerCase()} model on your own annotations.`
    : "Open fine-tuning to choose annotated images or apply a saved model.";

  return (
    <>
      <span className="finetune-header-button-tooltip" title={tooltip}>
        <button
          type="button"
          className="header-route-link finetune-header-button"
          data-testid="finetune-organelle-button"
          title={tooltip}
          onClick={() => setOpen(true)}
        >
          Fine-Tune
        </button>
      </span>
      <FineTuneDialog
        open={open}
        onClose={() => setOpen(false)}
        segmentationType={currentSegmentation?.segmentation_type ?? null}
        initialAssetIds={[image.id]}
        initialExperimentId={image.experiment_id ?? undefined}
        initialParentAdapterId={initialAdapterId ?? undefined}
        initialBaseModel={initialBaseModel}
        segmentationId={currentSegmentation?.id ?? null}
        onAppliedImageCompleted={onAppliedImageCompleted}
        onAppliedImageQueued={onAppliedImageCompleted}
      />
    </>
  );
}
