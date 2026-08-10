/**
 * Main App component that routes between the library, the viewer and the
 * labeling (segmentation) screen.
 */

import { Suspense, lazy, useEffect } from "react";
import { Navigate, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { getAsset } from "@/shared/api/assets";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import "./App.css";

const LABELING_VIEW = "labeling";
const VIEWER_VIEW = "viewer";

const LibraryPage = lazy(() =>
  import("@/features/library/LibraryPage").then((module) => ({
    default: module.LibraryPage,
  }))
);
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
const AnalysisScreen = lazy(() =>
  import("@/features/analysis/AnalysisScreen").then((module) => ({
    default: module.AnalysisScreen,
  }))
);
const AdaptWizard = lazy(() =>
  import("@/features/finetune/AdaptWizard").then((module) => ({
    default: module.AdaptWizard,
  }))
);
const ModelsScreen = lazy(() =>
  import("@/features/models/ModelsScreen").then((module) => ({
    default: module.ModelsScreen,
  }))
);

function RouteFallback() {
  return <div className="loading">Loading...</div>;
}

function AssetRedirect() {
  const { assetId } = useParams();

  if (!assetId) {
    return <Navigate to="/" replace />;
  }

  return <Navigate to={`/assets/${assetId}/${VIEWER_VIEW}`} replace />;
}

function AssetRoute() {
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

function App() {
  return (
    <div className="app">
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          <Route path="/" element={<LibraryPage />} />
          {/* Models are an app-level concern, not an image-level one: a user
              who never opens Adapt still needs to know what is installed and
              what can run. */}
          <Route path="/models" element={<ModelsScreen />} />
          <Route path="/assets/:assetId" element={<AssetRedirect />} />
          {/* Declared before the generic :view route so "analysis" and "adapt"
              are never mistaken for a viewer/labeling view name. */}
          <Route path="/assets/:assetId/analysis" element={<AnalysisScreen />} />
          <Route path="/assets/:assetId/adapt" element={<AdaptWizard />} />
          <Route
            path="/assets/:assetId/:view/:segmentationTypeName?"
            element={<AssetRoute />}
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </div>
  );
}

export default App;
