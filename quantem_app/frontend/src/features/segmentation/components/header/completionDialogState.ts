/**
 * The state behind the labeling header's two destructive confirmations.
 *
 * Split out of `SegmentationHeader.tsx` unchanged, and kept apart from
 * `CompletionDialogs.tsx` so that file exports components only. What each hook
 * holds is what its dialog needs and nothing else, which is what lets the
 * header hold the trigger and no dialog state at all.
 */

import { useCallback, useEffect, useState } from "react";
import { getSegmentationCompletionPreview } from "@/shared/api/segmentations/annotations";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { SegmentationCompletionPreview } from "@/shared/types";

/**
 * Mark Image Done: the lock, and the separate opt-in to delete.
 *
 * The count is read fresh every time the dialog opens — not from
 * `segment_counts` on the segmentation payload, which can be a poll behind, and
 * `POST` compares the acknowledged count against a fresh read and returns 409
 * on a mismatch. A dialog quoting a stale number would simply fail at the last
 * click.
 */
export function useCompletionConfirm({
  segmentationId,
  onToggleSegmentationComplete,
}: {
  segmentationId: string | null;
  onToggleSegmentationComplete: (options?: {
    discardUnconfirmed: boolean;
    acknowledgedDiscardCount: number;
  }) => void | Promise<void>;
}) {
  const [completeConfirmOpen, setCompleteConfirmOpen] = useState(false);
  const [discardUnconfirmed, setDiscardUnconfirmed] = useState(false);
  const [preview, setPreview] = useState<SegmentationCompletionPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewNonce, setPreviewNonce] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const open = useCallback(() => {
    setPreview(null);
    setPreviewError(null);
    setSubmitError(null);
    setDiscardUnconfirmed(false);
    setCompleteConfirmOpen(true);
    setPreviewNonce((current) => current + 1);
  }, []);

  const close = useCallback(() => setCompleteConfirmOpen(false), []);

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
  const confirm = useCallback(async () => {
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

  return {
    isOpen: completeConfirmOpen,
    open,
    close,
    preview,
    previewError,
    submitError,
    submitting,
    discardCount,
    discardUnconfirmed,
    setDiscardUnconfirmed,
    confirm,
  };
}

export type CompletionConfirmState = ReturnType<typeof useCompletionConfirm>;

/**
 * Delete every reviewed object and queue a fresh run.
 *
 * Same contract as the completion confirm: the dialog closes only when the
 * server agreed. The refusal worth keeping open for is the completion lock's
 * 409 — the segmentation was marked done in another tab — and closing on it
 * would replace the explanation with silence.
 */
export function useClearRerunConfirm({
  onClearMislabeledObjects,
}: {
  onClearMislabeledObjects?: () => Promise<void>;
}) {
  const [clearRerunConfirmOpen, setClearRerunConfirmOpen] = useState(false);
  const [clearingRerun, setClearingRerun] = useState(false);
  const [clearRerunError, setClearRerunError] = useState<string | null>(null);

  const open = useCallback(() => {
    setClearRerunError(null);
    setClearRerunConfirmOpen(true);
  }, []);

  const close = useCallback(() => setClearRerunConfirmOpen(false), []);

  const confirm = useCallback(async () => {
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

  return {
    isOpen: clearRerunConfirmOpen,
    open,
    close,
    clearing: clearingRerun,
    error: clearRerunError,
    confirm,
  };
}

export type ClearRerunConfirmState = ReturnType<typeof useClearRerunConfirm>;
