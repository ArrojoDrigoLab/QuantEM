import { useCallback, useEffect, useRef } from "react";
import {
  OVERLAY_REFRESH_IDLE_DELAY_MS,
  OVERLAY_REFRESH_MAX_DELAY_MS,
} from "@/features/segmentation/screen/utils/constants";
import type { SegmentationOverlayMutationState } from "@/shared/types";

interface UseOverlayRefreshSchedulerArgs {
  currentSegmentationId: string | null;
  overlayManifestNeedsPolling: boolean;
  refetchOverlayManifest: () => Promise<unknown>;
  refetchSegmentations: () => Promise<void>;
  refetchLeftSegments: () => Promise<void>;
  setOverlayManifestPollingEnabled: (enabled: boolean) => void;
}

export function useOverlayRefreshScheduler({
  currentSegmentationId,
  overlayManifestNeedsPolling,
  refetchOverlayManifest,
  refetchSegmentations,
  refetchLeftSegments,
  setOverlayManifestPollingEnabled,
}: UseOverlayRefreshSchedulerArgs) {
  const overlayIdleRefreshTimeoutRef = useRef<number | null>(null);
  const overlayMaxRefreshTimeoutRef = useRef<number | null>(null);
  const pendingOverlayManifestRefreshRef = useRef(false);

  const clearDeferredOverlayManifestRefresh = useCallback(() => {
    if (overlayIdleRefreshTimeoutRef.current !== null) {
      window.clearTimeout(overlayIdleRefreshTimeoutRef.current);
      overlayIdleRefreshTimeoutRef.current = null;
    }
    if (overlayMaxRefreshTimeoutRef.current !== null) {
      window.clearTimeout(overlayMaxRefreshTimeoutRef.current);
      overlayMaxRefreshTimeoutRef.current = null;
    }
  }, []);

  useEffect(() => {
    clearDeferredOverlayManifestRefresh();
    pendingOverlayManifestRefreshRef.current = false;
  }, [clearDeferredOverlayManifestRefresh, currentSegmentationId]);

  useEffect(() => {
    return () => clearDeferredOverlayManifestRefresh();
  }, [clearDeferredOverlayManifestRefresh]);

  const flushDeferredOverlayManifestRefresh = useCallback(() => {
    clearDeferredOverlayManifestRefresh();
    const shouldRefetch =
      pendingOverlayManifestRefreshRef.current || overlayManifestNeedsPolling;
    pendingOverlayManifestRefreshRef.current = false;
    setOverlayManifestPollingEnabled(true);
    if (!shouldRefetch) {
      return;
    }
    void refetchOverlayManifest();
  }, [
    clearDeferredOverlayManifestRefresh,
    overlayManifestNeedsPolling,
    refetchOverlayManifest,
    setOverlayManifestPollingEnabled,
  ]);

  const deferOverlayManifestRefreshForAnnotationActivity = useCallback(() => {
    const shouldDefer =
      pendingOverlayManifestRefreshRef.current || overlayManifestNeedsPolling;
    if (!shouldDefer) {
      return;
    }
    setOverlayManifestPollingEnabled(false);
    if (overlayIdleRefreshTimeoutRef.current !== null) {
      window.clearTimeout(overlayIdleRefreshTimeoutRef.current);
    }
    overlayIdleRefreshTimeoutRef.current = window.setTimeout(() => {
      flushDeferredOverlayManifestRefresh();
    }, OVERLAY_REFRESH_IDLE_DELAY_MS);
    if (overlayMaxRefreshTimeoutRef.current === null) {
      overlayMaxRefreshTimeoutRef.current = window.setTimeout(() => {
        flushDeferredOverlayManifestRefresh();
      }, OVERLAY_REFRESH_MAX_DELAY_MS);
    }
  }, [
    flushDeferredOverlayManifestRefresh,
    overlayManifestNeedsPolling,
    setOverlayManifestPollingEnabled,
  ]);

  const registerAnnotationActivity = useCallback(() => {
    deferOverlayManifestRefreshForAnnotationActivity();
  }, [deferOverlayManifestRefreshForAnnotationActivity]);

  const requestDeferredOverlayManifestRefresh = useCallback(() => {
    pendingOverlayManifestRefreshRef.current = true;
    deferOverlayManifestRefreshForAnnotationActivity();
  }, [deferOverlayManifestRefreshForAnnotationActivity]);

  const handleOverlayMutationRefresh = useCallback(
    (overlay: SegmentationOverlayMutationState | null | undefined) => {
      void refetchSegmentations();
      if (overlay) {
        requestDeferredOverlayManifestRefresh();
      }
    },
    [refetchSegmentations, requestDeferredOverlayManifestRefresh]
  );

  const refreshSegmentViews = useCallback(
    async ({ deferOverlayRefresh = false }: { deferOverlayRefresh?: boolean } = {}) => {
      if (!currentSegmentationId) return;
      if (deferOverlayRefresh) {
        requestDeferredOverlayManifestRefresh();
      }
      await Promise.all([
        refetchLeftSegments(),
        refetchSegmentations(),
        deferOverlayRefresh ? Promise.resolve() : refetchOverlayManifest(),
      ]);
    },
    [
      currentSegmentationId,
      refetchLeftSegments,
      refetchOverlayManifest,
      refetchSegmentations,
      requestDeferredOverlayManifestRefresh,
    ]
  );

  return {
    registerAnnotationActivity,
    handleOverlayMutationRefresh,
    refreshSegmentViews,
  };
}
