import { useCallback, useEffect, useState } from "react";
import { TOAST_AUTO_DISMISS_MS } from "@/shared/ui/toast";

interface UseSegmentationScreenUiStateArgs {
  currentSegmentationId: string | null;
}

export function useSegmentationScreenUiState({
  currentSegmentationId,
}: UseSegmentationScreenUiStateArgs) {
  const [leftNavigateMode, setLeftNavigateMode] = useState(true);
  const [showConfirmedPanel, setShowConfirmedPanel] = useState(true);
  const [uncertainLimit, setUncertainLimit] = useState(50);
  /**
   * The one transient message slot on this screen.
   *
   * `tone` exists because not everything worth saying here is a failure. A
   * self-crossing outline now stores every lobe it encloses and the server says
   * so in the response ("segments[0] crosses itself: it encloses 2 separate
   * areas rather than one"); the drawing succeeded exactly as asked, and
   * reporting it in the red error toast would read as though it had not. One
   * slot rather than two, so a notice and an error cannot stack on top of each
   * other in the same corner.
   */
  const [toast, setToast] = useState<{
    id: string;
    message: string;
    tone: "error" | "notice";
  } | null>(null);

  const pushToast = useCallback((message: string, tone: "error" | "notice") => {
    setToast({
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      message,
      tone,
    });
  }, []);

  const showErrorToast = useCallback(
    (message: string) => pushToast(message, "error"),
    [pushToast]
  );

  /** Something the user should know about an action that worked. */
  const showNoticeToast = useCallback(
    (message: string) => pushToast(message, "notice"),
    [pushToast]
  );

  const dismissToast = useCallback(() => {
    setToast(null);
  }, []);

  const toggleLeftNavigateMode = useCallback(() => {
    setLeftNavigateMode((previous) => !previous);
  }, []);

  /**
   * Turn Navigate off.
   *
   * Called when the user picks a drawing tool or enters Correct: while Navigate
   * is on the interaction router drops every labeling click and drag, so the
   * first stroke after choosing a tool silently did nothing.
   */
  const exitNavigateMode = useCallback(() => {
    setLeftNavigateMode(false);
  }, []);

  useEffect(() => {
    setLeftNavigateMode(true);
    setToast(null);
  }, [currentSegmentationId]);

  useEffect(() => {
    if (!toast) return undefined;
    const timeout = window.setTimeout(() => {
      setToast((current) => (current?.id === toast.id ? null : current));
    }, TOAST_AUTO_DISMISS_MS);
    return () => window.clearTimeout(timeout);
  }, [toast]);

  return {
    leftNavigateMode,
    setLeftNavigateMode,
    toggleLeftNavigateMode,
    exitNavigateMode,
    showConfirmedPanel,
    setShowConfirmedPanel,
    uncertainLimit,
    setUncertainLimit,
    toast,
    showErrorToast,
    showNoticeToast,
    dismissToast,
  };
}
