import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";

/**
 * What an overlay manifest means, in one place.
 *
 * These three predicates used to be written out by hand in three files --
 * `useSegmentationOverlayManifest`, `ViewerScreen` and the labeling screen's
 * `useOverlayManifestState` -- and all three agreed that a *failed* build was
 * still in progress. That is finding F1: "Overlay updating…" on screen for
 * ever, a poll every 1.5 s against a state that will never change, and the
 * server's reason for the failure discarded on every one of those responses.
 *
 * They live in their own module rather than beside the fetching hook so that a
 * component can ask what a manifest means without importing the hook -- and so
 * that a test which mocks the hook module does not accidentally stub out the
 * meaning of the data as well.
 */

/**
 * A `FAILED` overlay build is terminal.
 *
 * `ensure_overlay_manifest` (`overlay_ngff/manifest.py`) deliberately stops
 * re-queueing once the state carries a failure and a reason: the build cannot
 * succeed, so asking again for ever is worse than saying so. The client has to
 * agree, or `desired_revision > applied_revision` -- which a failed build
 * leaves true *permanently* -- keeps a spinner turning.
 */
export function overlayBuildFailed(
  manifest: SegmentationOverlayManifest | null | undefined
): boolean {
  return manifest?.status === "FAILED";
}

/**
 * The reason the last build failed, or `null` when there is nothing wrong.
 *
 * Returns `null` on a healthy manifest even if the server left a stale string
 * behind, and `null` (not `""`) for a `FAILED` manifest with no recorded
 * reason, so callers can tell "no failure" from "failed, cause unknown" --
 * those need different sentences.
 */
export function overlayBuildFailureReason(
  manifest: SegmentationOverlayManifest | null | undefined
): string | null {
  if (!overlayBuildFailed(manifest)) return null;
  const reason = manifest?.last_error?.trim();
  return reason ? reason : null;
}

/**
 * Is this overlay genuinely still being built?
 *
 * The one predicate behind every "Overlay updating…" string and every poll
 * timer in the app. Nothing is lost by returning false on a failure: the
 * manifest is refetched explicitly after every proofreading mutation
 * (`useOverlayRefreshScheduler`) and on the retry button, and either of those
 * clears the failure server-side and puts the status back to BUILDING/DIRTY,
 * at which point polling resumes.
 */
export function overlayIsUpdating(
  manifest: SegmentationOverlayManifest | null | undefined
): boolean {
  if (!manifest) return false;
  if (overlayBuildFailed(manifest)) return false;
  return (
    Boolean(manifest.update_job) ||
    manifest.status === "BUILDING" ||
    manifest.status === "DIRTY" ||
    manifest.desired_revision > manifest.applied_revision
  );
}
