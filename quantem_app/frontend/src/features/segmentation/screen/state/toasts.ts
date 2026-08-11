/**
 * The one transient message slot on the labeling screen.
 *
 * Split out of `useSegmentationScreenUiState.ts` unchanged. `tone` exists
 * because not everything worth saying here is a failure. A self-crossing
 * outline now stores every lobe it encloses and the server says so in the
 * response ("segments[0] crosses itself: it encloses 2 separate areas rather
 * than one"); the drawing succeeded exactly as asked, and reporting it in the
 * red error toast would read as though it had not. One slot rather than two, so
 * a notice and an error cannot stack on top of each other in the same corner.
 */

import { useCallback, useEffect, useState } from "react";
import { TOAST_AUTO_DISMISS_MS } from "@/shared/ui/toast";

export interface ScreenToast {
  id: string;
  message: string;
  tone: "error" | "notice";
}

export function useSegmentationToast({
  currentSegmentationId,
}: {
  currentSegmentationId: string | null;
}) {
  const [toast, setToast] = useState<ScreenToast | null>(null);

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

  useEffect(() => {
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
    toast,
    showErrorToast,
    showNoticeToast,
    dismissToast,
  };
}
