/**
 * The `/assets/:assetId/...` routes: the redirect, the view switch, and the
 * fallback both of them render while a screen chunk is still loading.
 *
 * Moved out of `App.tsx` unchanged, so that `app/routes.tsx` can be a flat
 * append-only table. Five separate packages want to register a route; none of
 * them wants to touch this.
 */

import { Suspense, lazy, useEffect } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";
import { getAsset } from "@/shared/api/assets";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";
import { useApiQuery } from "@/shared/hooks/useApiQuery";

export const LABELING_VIEW = "labeling";
export const VIEWER_VIEW = "viewer";

const SegmentationScreen = lazy(() =>
  import("@/features/segmentation/SegmentationScreen").then((module) => ({
    default: module.SegmentationScreen,
  }))
);
const ViewerScreen = lazy(() =>
  import("@/features/viewer/ViewerScreen").then((module) => ({
    default: module.ViewerScreen,
  }))
);

export function RouteFallback() {
  return <div className="loading">Loading...</div>;
}

export function AssetRedirect() {
  const { assetId } = useParams();

  if (!assetId) {
    return <Navigate to="/" replace />;
  }

  return <Navigate to={`/assets/${assetId}/${VIEWER_VIEW}`} replace />;
}

export function AssetRoute() {
  const { assetId, view } = useParams();
  const navigate = useNavigate();
  const { setSelectedAssetId, clearSelection } = useSelectionStore();
  const { data: asset } = useApiQuery(
    () => {
      if (!assetId) {
        return Promise.reject(new Error("Missing asset id"));
      }
      return getAsset(assetId);
    },
    [assetId]
  );

  useEffect(() => {
    if (assetId) {
      setSelectedAssetId(assetId);
    }
  }, [assetId, setSelectedAssetId]);

  useEffect(() => {
    return () => {
      clearSelection();
    };
  }, [clearSelection]);

  const handleBack = () => {
    clearSelection();
    navigate("/");
  };

  if (!assetId) {
    return <Navigate to="/" replace />;
  }

  if (!asset) {
    return (
      <div className="loading-with-back">
        {/* Nothing has rendered a header yet, so this one is on its own and has
            nothing to overlap. */}
        <button className="back-button" onClick={handleBack} type="button">
          ← Back to Library
        </button>
        <div className="loading">Loading...</div>
      </div>
    );
  }

  if (view !== LABELING_VIEW && view !== VIEWER_VIEW) {
    return <Navigate to={`/assets/${assetId}/${VIEWER_VIEW}`} replace />;
  }

  // Both screens render their own in-header back button. There is deliberately
  // no floating overlay button here: it was `position: fixed` at (16, 16) with
  // `z-index: 1000`, which is exactly where the viewer's <h1>/<h2> and filename
  // sit, so it covered the name of the image being viewed.
  return (
    <Suspense fallback={<RouteFallback />}>
      {view === VIEWER_VIEW ? <ViewerScreen /> : <SegmentationScreen />}
    </Suspense>
  );
}
