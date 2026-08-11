/**
 * The labeling view's way in, for the organelle on screen.
 *
 * Owner R13 puts one condition on it: it is enabled only once this organelle
 * has **at least one confirmed area or one ROI marked done** on this image. The
 * condition is answered by `POST /api/finetune/preview/` over this one asset —
 * the same count the dialog shows and the same rule the run endpoint enforces,
 * rather than a second guess assembled from whatever the labeling screen
 * happens to have loaded.
 *
 * Until that answer arrives the button is present and disabled, and it says
 * why. A control that appears only once a condition is met teaches nobody what
 * the condition was.
 */

import { useEffect, useState } from "react";
import { previewFineTuneScope } from "@/shared/api/finetune";
import { TRAINING_ANNOTATION_SOURCES } from "@/shared/constants/confirmedArea";
import { FineTuneDialog } from "@/features/finetune/FineTuneDialog";
import type { AssetDetail, ImageSegmentation } from "@/shared/types";
import "./FineTuneOrganelleButton.css";

export function FineTuneOrganelleButton({
  image,
  currentSegmentation,
}: {
  image: AssetDetail;
  currentSegmentation: ImageSegmentation | null;
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
  }, [segmentationTypeId, image.id]);

  const organelle =
    currentSegmentation?.segmentation_type?.short_name ||
    currentSegmentation?.segmentation_type?.long_name ||
    "this organelle";
  const enabled = (annotationCount ?? 0) > 0;

  return (
    <>
      <button
        type="button"
        className="header-route-link finetune-header-button"
        data-testid="finetune-organelle-button"
        disabled={!enabled}
        title={
          enabled
            ? `Train a ${organelle.toLowerCase()} model on your own annotations.`
            : `Nothing to train on yet for ${organelle.toLowerCase()} on this image. ${TRAINING_ANNOTATION_SOURCES}`
        }
        onClick={() => setOpen(true)}
      >
        Fine-Tune
      </button>
      <FineTuneDialog
        open={open}
        onClose={() => setOpen(false)}
        segmentationType={currentSegmentation?.segmentation_type ?? null}
        initialAssetIds={[image.id]}
      />
    </>
  );
}
