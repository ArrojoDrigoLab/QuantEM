import { useCallback, useEffect, useMemo, useState } from "react";
import { useSegmentationOverlayManifest } from "@/hooks/useSegmentationOverlayManifest";

interface UseOverlayManifestStateArgs {
  currentSegmentationId: string | null;
  activeSourceModel: string | null;
}

export function useOverlayManifestState({
  currentSegmentationId,
  activeSourceModel,
}: UseOverlayManifestStateArgs) {
  const [overlayManifestPollingEnabled, setOverlayManifestPollingEnabled] =
    useState(true);
  const [, setLeftDisplayedOverlayRevision] = useState<number | null>(null);
  const [, setRightDisplayedOverlayRevision] = useState<number | null>(null);
  const { manifest: overlayManifest, refetch: refetchOverlayManifest } =
    useSegmentationOverlayManifest(
      currentSegmentationId,
      Boolean(currentSegmentationId),
      overlayManifestPollingEnabled,
      activeSourceModel
    );

  useEffect(() => {
    setOverlayManifestPollingEnabled(true);
    setLeftDisplayedOverlayRevision(null);
    setRightDisplayedOverlayRevision(null);
  }, [activeSourceModel, currentSegmentationId]);

  const overlayManifestNeedsPolling = useMemo(
    () =>
      Boolean(
        overlayManifest &&
          (overlayManifest.status === "BUILDING" ||
            overlayManifest.status === "DIRTY" ||
            overlayManifest.desired_revision > overlayManifest.applied_revision)
      ),
    [overlayManifest]
  );

  const handleLeftOverlayRevisionDisplayed = useCallback((revision: number | null) => {
    setLeftDisplayedOverlayRevision(revision);
  }, []);

  const handleRightOverlayRevisionDisplayed = useCallback((revision: number | null) => {
    setRightDisplayedOverlayRevision(revision);
  }, []);

  return {
    overlayManifest,
    overlayUpdating: overlayManifestNeedsPolling,
    overlayManifestNeedsPolling,
    usesRasterReviewOverlay: Boolean(overlayManifest?.ngff_url),
    refetchOverlayManifest,
    setOverlayManifestPollingEnabled,
    handleLeftOverlayRevisionDisplayed,
    handleRightOverlayRevisionDisplayed,
  };
}
