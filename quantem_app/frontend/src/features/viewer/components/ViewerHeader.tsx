/**
 * The viewer's own header: which image, its calibration, and the way out.
 *
 * Moved out of `ViewerScreen.tsx` unchanged. It is in the header, not floating
 * over it: the app used to render a `position: fixed` back button at (16, 16)
 * as a sibling of the screen, which landed exactly on top of the <h2> and the
 * filename -- you could not read which image you were looking at.
 */

import { Link } from "react-router-dom";
import { PixelSizeEditor } from "@/shared/ui/PixelSize";
import type { AssetDetail } from "@/shared/types/images";

export function ViewerHeader({
  image,
  selectedAssetId,
  overlayManifestLoading,
  overlayManifestRefetching,
  overlayUpdating,
  overlayBuildFailureCount,
  onBackToLibrary,
  onPixelSizeSaved,
}: {
  image: AssetDetail;
  selectedAssetId: string | null;
  overlayManifestLoading: boolean;
  overlayManifestRefetching: boolean;
  overlayUpdating: boolean;
  overlayBuildFailureCount: number;
  onBackToLibrary: () => void;
  onPixelSizeSaved: () => void;
}) {
  return (
    <div className="viewer-header">
      <div className="viewer-header-identity">
        {/* In the header, not floating over it. The app used to render a
            `position: fixed` back button at (16, 16) as a sibling of this
            screen, which landed exactly on top of the <h2> and the filename
            -- you could not read which image you were looking at. */}
        <button
          type="button"
          className="viewer-back-button"
          onClick={onBackToLibrary}
        >
          ← Back to Library
        </button>
        <h2>{image.display_name}</h2>
        <span className="viewer-filename">{image.original_filename}</span>
        {/* The viewer is the first screen where the image is a single
            concrete thing, so it is where calibration belongs. Editing it
            PATCHes the asset and refetches, which is the only route by
            which an untagged EM export ever becomes measurable. */}
        <PixelSizeEditor
          className="viewer-pixel-size"
          asset={image}
          onSaved={() => {
            onPixelSizeSaved();
          }}
        />
      </div>
      <div className="viewer-header-actions">
        {(overlayManifestLoading || overlayManifestRefetching) && (
          <span className="viewer-loading-indicator">Loading overlays…</span>
        )}
        {overlayUpdating && (
          <span className="viewer-loading-indicator">Overlay updating…</span>
        )}
        {/* The correction to the indicator above. Somebody who has been
            watching "Overlay updating…" is watching this spot, so the
            news that it stopped belongs here; the reason and the retry
            are on the card, which is a few centimetres to the right. */}
        {overlayBuildFailureCount > 0 && (
          <span className="viewer-error-indicator" role="status">
            {overlayBuildFailureCount === 1
              ? "Overlay could not be rebuilt — see the card"
              : `${overlayBuildFailureCount} overlays could not be rebuilt — see the cards`}
          </span>
        )}
        {selectedAssetId ? (
          <>
            <Link
              className="viewer-header-link"
              to={`/assets/${selectedAssetId}/analysis`}
            >
              Analysis
            </Link>
            <Link
              className="viewer-header-link"
              to={`/assets/${selectedAssetId}/adapt`}
            >
              Adapt a model
            </Link>
          </>
        ) : null}
      </div>
    </div>
  );
}
