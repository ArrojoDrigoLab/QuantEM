/** State for explicit manual coverage confirmation and rerun recovery. */
import { useCallback, useState } from "react";
import { extractApiErrorMessage } from "@/utils/apiErrors";

export function useCompletionConfirm({ onToggleSegmentationComplete }: {
  segmentationId: string | null;
  onToggleSegmentationComplete: (options?: { discardUnconfirmed: boolean; acknowledgedDiscardCount: number }) => void | Promise<void>;
}) {
  const [isOpen, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const open = useCallback(() => { setSubmitError(null); setOpen(true); }, []);
  const close = useCallback(() => setOpen(false), []);
  const confirm = useCallback(async () => {
    setSubmitting(true); setSubmitError(null);
    try { await onToggleSegmentationComplete(); setOpen(false); }
    catch (error) { setSubmitError(extractApiErrorMessage(error, "Manual coverage could not be confirmed.")); }
    finally { setSubmitting(false); }
  }, [onToggleSegmentationComplete]);
  return { isOpen, open, close, submitting, submitError, confirm };
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
