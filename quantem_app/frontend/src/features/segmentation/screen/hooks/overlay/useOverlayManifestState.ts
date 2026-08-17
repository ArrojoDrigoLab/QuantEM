import { useCallback, useEffect, useMemo, useState } from "react";
import { useSegmentationOverlayManifest } from "@/hooks/useSegmentationOverlayManifest";
import {
  overlayBuildFailed,
  overlayBuildFailureReason,
  overlayIsUpdating,
} from "@/hooks/overlayManifestStatus";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";

interface UseOverlayManifestStateArgs {
  currentSegmentationId: string | null;
  activeSourceModel: string | null;
}

/** Which of the two overlay bundles a piece of state is talking about. */
export type OverlayDisplayRole = "model" | "confirmed";

export interface FailedOverlayBundle {
  role: OverlayDisplayRole;
  manifest: SegmentationOverlayManifest;
}

/** How to name a bundle to a user, in a sentence about that bundle alone. */
export const OVERLAY_DISPLAY_LABELS: Record<OverlayDisplayRole, string> = {
  model: "model preview display",
  confirmed: "confirmed display",
};

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
  const { manifest: overlayManifest, refetch: refetchModelOverlayManifest } =
    useSegmentationOverlayManifest(
      currentSegmentationId,
      Boolean(currentSegmentationId && activeSourceModel),
      overlayManifestPollingEnabled,
      activeSourceModel
    );
  const {
    manifest: confirmedOverlayManifest,
    refetch: refetchConfirmedOverlayManifest,
  } = useSegmentationOverlayManifest(
    currentSegmentationId,
    Boolean(currentSegmentationId),
    overlayManifestPollingEnabled,
    null
  );
  const refetchOverlayManifest = useCallback(
    () =>
      Promise.all([
        refetchModelOverlayManifest(),
        refetchConfirmedOverlayManifest(),
      ]),
    [refetchConfirmedOverlayManifest, refetchModelOverlayManifest]
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
   * saying the display was updating about a build that had already given up.
   *
   * Polling restarts by itself: every proofreading mutation refetches the
   * manifest through `useOverlayRefreshScheduler`, and the server clears the
   * failure and re-queues on the next mutation or on `retryOverlayBuild`.
   *
   * This is the OR of the two bundles, so -- unlike the single-manifest version
   * this replaced -- it is *not* mutually exclusive with `overlayBuildFailed`:
   * one display can be rebuilding while the other has given up. Anything that
   * names a display to the user must read the per-bundle flags below instead.
   */
  const overlayManifestNeedsPolling = useMemo(
    () =>
      overlayIsUpdating(overlayManifest) ||
      overlayIsUpdating(confirmedOverlayManifest),
    [confirmedOverlayManifest, overlayManifest]
  );

  /**
   * Every bundle whose build has failed and will not be retried until somebody
   * asks, tagged with the slot it came from.
   *
   * The two bundles fail independently: `register_overlay_mutation_all_bundles`
   * dirties the named-model bundle and the confirmed-display bundle separately
   * and queues a rebuild for each, and the server runs at most one overlay
   * rasterizer at a time, so "model failed while confirmed is still building"
   * is an ordinary outcome rather than an exotic one. Collapsing them into a
   * single manifest -- which is what this used to be -- meant the second
   * failure was never shown at all and the one retry button silently re-queued
   * only the first bundle, leaving the user sure they had fixed "the overlay".
   *
   * The role comes from *which hook slot* produced the manifest and never from
   * `manifest.display_role`: `ensure_overlay_manifest` can answer a
   * model-scoped request with the confirmed state's payload (grafting only the
   * model's `last_error` on to it), so the payload's own role field can
   * disagree with the bundle the client actually asked about.
   */
  const failedOverlays = useMemo<FailedOverlayBundle[]>(() => {
    const failed: FailedOverlayBundle[] = [];
    if (overlayManifest && overlayBuildFailed(overlayManifest)) {
      failed.push({ role: "model", manifest: overlayManifest });
    }
    if (confirmedOverlayManifest && overlayBuildFailed(confirmedOverlayManifest)) {
      failed.push({ role: "confirmed", manifest: confirmedOverlayManifest });
    }
    return failed;
  }, [confirmedOverlayManifest, overlayManifest]);

  const overlayBuildHasFailed = failedOverlays.length > 0;

  const confirmedOverlayBuildHasFailed = useMemo(
    () => overlayBuildFailed(confirmedOverlayManifest),
    [confirmedOverlayManifest]
  );

  const failedOverlayManifest = failedOverlays[0]?.manifest ?? null;

  /**
   * The server's reason, verbatim, or `null` when it recorded none. Callers
   * must distinguish the two: "failed, and here is why" and "failed, and the
   * worker died without saying why" call for different sentences, and a bare
   * empty string would render as neither.
   */
  const overlayBuildError = useMemo(
    () => overlayBuildFailureReason(failedOverlayManifest),
    [failedOverlayManifest]
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
    confirmedOverlayManifest,
    modelOverlayUpdating: overlayIsUpdating(overlayManifest),
    confirmedOverlayUpdating: overlayIsUpdating(confirmedOverlayManifest),
    overlayUpdating: overlayManifestNeedsPolling,
    overlayManifestNeedsPolling,
    overlayBuildFailed: overlayBuildHasFailed,
    confirmedOverlayBuildFailed: confirmedOverlayBuildHasFailed,
    failedOverlays,
    failedOverlayManifest,
    overlayBuildError,
    handleOverlayBuildRetried,
    // Doubles as "a bundle exists on disk": when the build has failed, a true
    // value here means the canvas is showing a *stale* raster rather than
    // nothing, which is a different sentence for the user.
    usesRasterReviewOverlay: Boolean(
      overlayManifest?.ngff_url || confirmedOverlayManifest?.ngff_url
    ),
    refetchOverlayManifest,
    setOverlayManifestPollingEnabled,
    handleLeftOverlayRevisionDisplayed,
    handleRightOverlayRevisionDisplayed,
    leftDisplayedOverlayRevision,
    rightDisplayedOverlayRevision,
  };
}
