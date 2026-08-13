import { useCallback, useEffect, useMemo, useState } from "react";
import { useSegmentationOverlayManifest } from "@/hooks/useSegmentationOverlayManifest";
import {
  overlayBuildFailed,
  overlayBuildFailureReason,
  overlayIsUpdating,
} from "@/hooks/overlayManifestStatus";

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
  const [leftDisplayedOverlayRevision, setLeftDisplayedOverlayRevision] =
    useState<number | null>(null);
  const [rightDisplayedOverlayRevision, setRightDisplayedOverlayRevision] =
    useState<number | null>(null);
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

  /**
   * A FAILED build is terminal, so this is false on one: the server has
   * stopped re-queueing it, and `desired_revision > applied_revision` -- which
   * this predicate used to test on its own -- never comes back down again.
   * Leaving it true kept a 1.5 s poll running for ever and kept the sidebar
   * saying "Overlay updating." about a build that had already given up.
   *
   * Polling restarts by itself: every proofreading mutation refetches the
   * manifest through `useOverlayRefreshScheduler`, and the server clears the
   * failure and re-queues on the next mutation or on `retryOverlayBuild`.
   */
  const overlayManifestNeedsPolling = useMemo(
    () => overlayIsUpdating(overlayManifest),
    [overlayManifest]
  );

  /** The build failed and will not be retried until somebody asks. */
  const overlayBuildHasFailed = useMemo(
    () => overlayBuildFailed(overlayManifest),
    [overlayManifest]
  );

  /**
   * The server's reason, verbatim, or `null` when it recorded none. Callers
   * must distinguish the two: "failed, and here is why" and "failed, and the
   * worker died without saying why" call for different sentences, and a bare
   * empty string would render as neither.
   */
  const overlayBuildError = useMemo(
    () => overlayBuildFailureReason(overlayManifest),
    [overlayManifest]
  );

  /**
   * What to do once a retry has been accepted.
   *
   * The request itself belongs to `OverlayBuildFailureNotice`, which the
   * labeling sidebar and the viewer both mount, so that the two screens cannot
   * drift apart in either their wording or their behaviour. What the notice
   * cannot know is that this screen stopped polling when the build failed:
   * `queue_full_overlay_rebuild` clears `last_error` and puts the state back to
   * BUILDING, so the poll has to be switched back on and the manifest reread,
   * or the card sits there over a build that is already running.
   */
  const handleOverlayBuildRetried = useCallback(() => {
    setOverlayManifestPollingEnabled(true);
    void refetchOverlayManifest();
  }, [refetchOverlayManifest]);

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
    overlayBuildFailed: overlayBuildHasFailed,
    overlayBuildError,
    handleOverlayBuildRetried,
    // Doubles as "a bundle exists on disk": when the build has failed, a true
    // value here means the canvas is showing a *stale* raster rather than
    // nothing, which is a different sentence for the user.
    usesRasterReviewOverlay: Boolean(overlayManifest?.ngff_url),
    refetchOverlayManifest,
    setOverlayManifestPollingEnabled,
    handleLeftOverlayRevisionDisplayed,
    handleRightOverlayRevisionDisplayed,
    leftDisplayedOverlayRevision,
    rightDisplayedOverlayRevision,
  };
}
