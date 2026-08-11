/**
 * Deleting a segmentation, to the Mark-Done standard.
 *
 * Moved out of `ViewerScreen.tsx` unchanged. The dialog quotes counts read
 * fresh from `GET /api/segmentations/<id>/` when it opens — not
 * `segment_counts` off the list payload, which can be a poll behind — and the
 * DELETE carries the object count the user was shown. The server refuses a
 * stale count, an active job and the completion lock with a 409, and each
 * refusal is rendered in the dialog rather than closing it into silence.
 */

import { useCallback, useEffect, useState } from "react";
import {
  deleteSegmentation,
  getSegmentationDetail,
} from "@/shared/api/segmentations/lifecycle";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { ImageSegmentation } from "@/shared/types/images";
import type { SegmentationDeletePreview } from "@/shared/types/segmentation";

export function useSegmentationDelete({
  visibleSegmentations,
  refetchSegmentations,
}: {
  visibleSegmentations: ImageSegmentation[];
  refetchSegmentations: () => Promise<unknown> | unknown;
}) {
  const [deleteTarget, setDeleteTarget] = useState<ImageSegmentation | null>(null);
  const [deletePreview, setDeletePreview] =
    useState<SegmentationDeletePreview | null>(null);
  const [deletePreviewError, setDeletePreviewError] = useState<string | null>(null);
  const [deleteSubmitError, setDeleteSubmitError] = useState<string | null>(null);
  const [deletePreviewNonce, setDeletePreviewNonce] = useState(0);
  const [deleting, setDeleting] = useState(false);

  const handleRequestDeleteSegmentation = useCallback(
    (segmentationId: string) => {
      const target =
        visibleSegmentations.find((seg) => seg.id === segmentationId) ?? null;
      if (!target) return;
      setDeletePreview(null);
      setDeletePreviewError(null);
      setDeleteSubmitError(null);
      setDeleteTarget(target);
      setDeletePreviewNonce((current) => current + 1);
    },
    [visibleSegmentations]
  );

  const deleteTargetId = deleteTarget?.id ?? null;
  useEffect(() => {
    if (!deleteTargetId) return undefined;
    let cancelled = false;
    void getSegmentationDetail(deleteTargetId)
      .then((detail) => {
        if (!cancelled) setDeletePreview(detail.delete_preview);
      })
      .catch((error) => {
        if (cancelled) return;
        setDeletePreviewError(
          extractApiErrorMessage(
            error,
            "What this would delete could not be counted."
          )
        );
      });
    return () => {
      cancelled = true;
    };
  }, [deleteTargetId, deletePreviewNonce]);

  // Read out of the preview into a plain binding first: the dependency is the
  // count, not the preview object, exactly as it was before this hook existed.
  const deletePreviewObjectCount = deletePreview?.object_count;
  const confirmDeleteSegmentation = useCallback(async () => {
    if (!deleteTargetId) return;
    setDeleting(true);
    setDeleteSubmitError(null);
    try {
      await deleteSegmentation(deleteTargetId, deletePreviewObjectCount);
      setDeleteTarget(null);
      setDeletePreview(null);
      await refetchSegmentations();
    } catch (error) {
      setDeleteSubmitError(
        extractApiErrorMessage(
          error,
          "The segmentation could not be deleted; nothing was changed."
        )
      );
      // Re-read, so the numbers beside the refusal are the ones that would
      // now go — the usual cause of the 409 is a run that just finished.
      setDeletePreviewNonce((current) => current + 1);
    } finally {
      setDeleting(false);
    }
  }, [deleteTargetId, deletePreviewObjectCount, refetchSegmentations]);

  return {
    deleteTarget,
    deletePreview,
    deletePreviewError,
    deleteSubmitError,
    deleting,
    handleRequestDeleteSegmentation,
    confirmDeleteSegmentation,
    setDeleteTarget,
  };
}

export type SegmentationDeleteState = ReturnType<typeof useSegmentationDelete>;
