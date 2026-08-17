import { useCallback, useState } from "react";
import { rebuildSegmentationOverlay } from "@/shared/api/segmentations/overlays";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";
import { overlayBuildFailureReason } from "@/hooks/overlayManifestStatus";
import { failureCopy, readFailureCode } from "@/shared/copy/failures";
import "./OverlayBuildFailureNotice.css";

interface OverlayBuildFailureNoticeProps {
  /** The failed manifest. Rendering is the caller's decision; this asserts it. */
  manifest: SegmentationOverlayManifest;
  segmentationId: string;
  /**
   * Which display this card is about ("confirmed display", "model preview
   * display"), for callers that mount it bare. The viewer renders it inside a
   * per-layer row that already says which layer it belongs to, so it leaves
   * this unset and keeps the generic title.
   *
   * Pass the label the *caller* knows from the slot the manifest came out of,
   * not one derived from `manifest.display_role`: the server can answer a
   * model-scoped manifest request with the confirmed state's payload, so that
   * field is not a reliable name for the bundle that failed.
   */
  displayLabel?: string;
  /**
   * Refetch the manifest after a successful retry so the card leaves the
   * failed state (and polling restarts) without waiting for anything else.
   */
  onRetried?: () => void;
  className?: string;
}

/**
 * What to say when the overlay raster could not be rebuilt.
 *
 * This is the renderer finding F1 said did not exist. The server records the
 * cause on the overlay state and puts it on the manifest as `last_error`, and
 * `ensure_overlay_manifest` then deliberately stops re-queueing the build --
 * so the state is *terminal*. Until this component existed the only thing on
 * screen was "Overlay updating...", a promise about a process that had already
 * stopped, and the reason was thrown away by the client on every poll.
 *
 * Three things the copy has to get right:
 *
 *  - **The objects are not lost.** An overlay is a picture derived from the
 *    objects; a failed rebuild loses the picture, never the rows. A user who
 *    reads "failed" over an empty canvas will otherwise assume their run, or
 *    their proofreading, is gone.
 *  - **Which picture they are looking at.** If a previous bundle exists the
 *    server keeps serving it, so the canvas is *stale*, not empty -- and
 *    edits made since then are silently missing from it. Saying nothing there
 *    would be a different dishonesty from the one this fixes.
 *  - **The reason, verbatim.** "[WinError 183] Cannot create a file when that
 *    file already exists: ..." names a stray file at a path the user can go
 *    and look at. Paraphrasing it would throw away the only actionable part.
 *    (It is an OS message and a path -- not a command to type. I-12 holds.)
 *
 * **The class, when the server names one.** If the manifest carries an
 * `error_code` (catalogued in `quantem/core/error_codes.py`) the card leads
 * with that class's own copy from `shared/copy/failures`. A full disk is the
 * common cause here and it has a specific remedy, which the generic closing
 * advice can only hint at; the server's verbatim reason still follows, because
 * the class says what kind of failure it is and only the reason says which
 * file. Without a code the card renders exactly as it did before.
 */
export function OverlayBuildFailureNotice({
  manifest,
  segmentationId,
  displayLabel,
  onRetried,
  className,
}: OverlayBuildFailureNoticeProps) {
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState<string | null>(null);

  const reason = overlayBuildFailureReason(manifest);
  const hasStaleBundle = Boolean(manifest.ngff_url);
  const copy = failureCopy(readFailureCode(manifest));

  const handleRetry = useCallback(() => {
    setRetrying(true);
    setRetryError(null);
    void rebuildSegmentationOverlay(
      segmentationId,
      "full",
      manifest.source_model ?? null
    )
      .then(() => {
        onRetried?.();
      })
      .catch((error: unknown) => {
        setRetryError(
          extractApiErrorMessage(error, "The retry could not be started.")
        );
      })
      .finally(() => {
        setRetrying(false);
      });
  }, [manifest.source_model, onRetried, segmentationId]);

  return (
    <div
      className={
        className
          ? `overlay-build-failure ${className}`
          : "overlay-build-failure"
      }
      role="alert"
    >
      <div className="overlay-build-failure-title">
        {displayLabel
          ? `The ${displayLabel} could not be rebuilt`
          : "Overlay could not be rebuilt"}
      </div>
      <p className="overlay-build-failure-body">
        Your objects are safe. This only affects the picture the viewer draws
        from them.{" "}
        {hasStaleBundle
          ? "You are seeing the last overlay that built successfully, so anything changed since then is missing from it."
          : "Nothing is drawn here because no version of this overlay has ever finished building."}
      </p>
      {copy ? (
        <p className="overlay-build-failure-class">
          <strong>{copy.headline}</strong> {copy.body}
        </p>
      ) : null}
      <p className="overlay-build-failure-reason">
        {reason
          ? `Reason from the server: ${reason}`
          : "The server recorded no reason for it."}
      </p>
      <p className="overlay-build-failure-revisions">
        Revision {manifest.applied_revision} is on disk; revision{" "}
        {manifest.desired_revision} was requested.
      </p>
      <button
        type="button"
        className="overlay-build-failure-retry"
        onClick={handleRetry}
        disabled={retrying}
      >
        {retrying ? "Retrying…" : "Retry overlay build"}
      </button>
      {retryError && (
        <p className="overlay-build-failure-retry-error">{retryError}</p>
      )}
      <p className="overlay-build-failure-advice">
        If it keeps failing, the usual causes are a full disk or another program
        holding QuantEM&rsquo;s overlay files open.
      </p>
    </div>
  );
}
