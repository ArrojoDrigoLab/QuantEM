import { useOverlayLayerControls } from "@/features/segmentation/screen/hooks/overlay/useOverlayLayerControls";
import { useOverlayManifestState } from "@/features/segmentation/screen/hooks/overlay/useOverlayManifestState";
import { useOptimisticOverlayState } from "@/features/segmentation/screen/hooks/overlay/useOptimisticOverlayState";
import { useOverlayRefreshScheduler } from "@/features/segmentation/screen/hooks/overlay/useOverlayRefreshScheduler";
import { useOptimisticDeletionState } from "@/features/segmentation/screen/hooks/overlay/useOptimisticDeletionState";

interface UseSegmentationOverlayStateArgs {
  currentSegmentationId: string | null;
  activeSourceModel: string | null;
  segmentationInternalName: string | null;
  refetchSegmentations: () => Promise<void>;
  refetchLeftSegments: () => Promise<void>;
  useSmoothedSegmentGeometry: boolean;
}

export function useSegmentationOverlayState({
  currentSegmentationId,
  activeSourceModel,
  segmentationInternalName,
  refetchSegmentations,
  refetchLeftSegments,
  useSmoothedSegmentGeometry,
}: UseSegmentationOverlayStateArgs) {
  const manifest = useOverlayManifestState({
    currentSegmentationId,
    activeSourceModel,
  });
  const deletion = useOptimisticDeletionState(currentSegmentationId);
  const layers = useOverlayLayerControls({
    segmentationId: currentSegmentationId ?? "",
    overlayManifest: manifest.overlayManifest,
    hiddenSegmentIds: deletion.hiddenSegmentIds,
    hiddenSegmentVisualRevision: deletion.visualRevision,
  });
  const optimistic = useOptimisticOverlayState({
    currentSegmentationId,
    segmentationInternalName,
    useSmoothedSegmentGeometry,
    leftPanelLayerStyles: layers.leftPanelLayerStyles,
    hiddenSegmentIds: deletion.hiddenSegmentIds,
    // Do not retire a bridging vector merely because the new bundle exists on
    // disk. Keep it until the always-mounted left viewer has loaded that exact
    // revision, otherwise a large bundle produces a visible blank interval.
    settledOverlayRevision: manifest.leftDisplayedOverlayRevision,
  });
  const refresh = useOverlayRefreshScheduler({
    currentSegmentationId,
    overlayManifestNeedsPolling: manifest.overlayManifestNeedsPolling,
    refetchOverlayManifest: manifest.refetchOverlayManifest,
    refetchSegmentations,
    refetchLeftSegments,
    setOverlayManifestPollingEnabled: manifest.setOverlayManifestPollingEnabled,
  });

  return {
    manifest,
    optimistic,
    deletion,
    refresh,
    layers,
  };
}
