/**
 * The viewer shell: layout, the canvas, and what the sidebar is wired to.
 *
 * ## Why this file is a composition and not a component
 *
 * It was 742 lines in which the asset polls, the overlay state, the delete
 * flow, the header and the canvas were one function, and four packages needed
 * to change different parts of it at once. Each part now lives on its own:
 *
 * * `viewer/state/useViewerAssetState.ts` — the asset, the segmentations and
 *   the two polls that keep them fresh;
 * * `viewer/state/useOverlayConfig.ts` — which overlays are drawn, in what
 *   colour, and the manifests and LUTs behind them;
 * * `viewer/state/useSegmentationDelete.ts` — the delete flow's live counts and
 *   its refusals;
 * * `viewer/components/ViewerHeader.tsx` — the identity row and the indicators;
 * * `viewer/components/SegmentationDeleteNotice.tsx` — the delete dialog.
 *
 * Nothing moved changed. `ViewerScreen` is still the only export consumed
 * (`app/App.tsx`, `ViewerScreen.test.tsx`), `ViewerScreen.css` is not split,
 * and the rendered DOM is what it was.
 */

import { useCallback, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";
import { useViewportSync } from "@/hooks/useViewportSync";
import { getAssetNgffUrl } from "@/shared/api/assets";
import { ImageViewer } from "@/viewer/components/ImageViewer";
import { OverlaySelectionSidebar } from "@/features/viewer/components/OverlaySelectionSidebar";
import { SegmentationCreatePanel } from "@/features/viewer/components/SegmentationCreatePanel";
import type { SegmentationLaunch } from "@/features/viewer/components/SegmentationCreatePanel";
import { SegmentationDeleteDialog } from "@/features/viewer/components/SegmentationDeleteNotice";
import { ViewerHeader } from "@/features/viewer/components/ViewerHeader";
import { ExportDialog } from "@/features/export/ExportDialog";
import {
  getStageLabel,
  useViewerAssetState,
} from "@/features/viewer/state/useViewerAssetState";
import { useOverlayConfig } from "@/features/viewer/state/useOverlayConfig";
import { useSegmentationDelete } from "@/features/viewer/state/useSegmentationDelete";
import type { ImageSegmentation } from "@/shared/types/images";
import { segmentationRouteToken } from "@/shared/segmentationNames";
import "./ViewerScreen.css";

const ANALYSIS_MASK_INTERNAL_NAME = "quantem_internal_analysis_mask";

function labelingPath(assetId: string, segmentation: ImageSegmentation): string {
  if (
    segmentation.segmentation_type.internal_name ===
    ANALYSIS_MASK_INTERNAL_NAME
  ) {
    return `/assets/${assetId}/analysis-mask/${segmentation.id}`;
  }
  return `/assets/${assetId}/labeling/${encodeURIComponent(
    segmentationRouteToken(segmentation)
  )}`;
}

export function ViewerScreen() {
  const { selectedAssetId, setSelectedSegmentationId, clearSelection } =
    useSelectionStore();
  const navigate = useNavigate();

  const handleBackToHome = useCallback(() => {
    clearSelection();
    navigate("/");
  }, [clearSelection, navigate]);
  const { viewport, setViewport } = useViewportSync();
  const [exportOpen, setExportOpen] = useState(false);

  const {
    image,
    refetchImage,
    imageReady,
    refetchSegmentations,
    visibleSegmentations,
  } = useViewerAssetState(selectedAssetId);

  const handleSegmentationCreated = async (
    segmentation: ImageSegmentation,
    launch?: SegmentationLaunch
  ) => {
    const assetId = segmentation.asset ?? selectedAssetId;
    if (!assetId) return;
    setSelectedSegmentationId(segmentation.id);
    await refetchSegmentations();
    const params = new URLSearchParams();
    if (launch?.sourceModel) {
      params.set("source_model", launch.sourceModel);
    }
    if (launch?.runModel) {
      params.set("run_model", "1");
    }
    const query = params.toString();
    navigate(`${labelingPath(assetId, segmentation)}${query ? `?${query}` : ""}`);
  };

  const overlays = useOverlayConfig({ visibleSegmentations, imageReady });

  const handleEditSegmentation = (segmentationId: string) => {
    if (!selectedAssetId) return;
    const target = visibleSegmentations.find((seg) => seg.id === segmentationId);
    setSelectedSegmentationId(segmentationId);
    navigate(
      target
        ? labelingPath(selectedAssetId, target)
        : `/assets/${selectedAssetId}/labeling/new`
    );
  };

  const deleteState = useSegmentationDelete({
    visibleSegmentations,
    refetchSegmentations,
  });

  const existingSegmentationTypes = useMemo(
    () => visibleSegmentations.map((seg) => seg.segmentation_type.long_name),
    [visibleSegmentations]
  );
  const existingSegmentationTypeIds = useMemo(
    () => visibleSegmentations.map((seg) => seg.segmentation_type.id),
    [visibleSegmentations]
  );
  const imageSizeBytes = useMemo(() => {
    if (!image) return null;
    const depth = image.stored_depth ?? image.depth ?? 1;
    const channels = Math.max(1, image.channels || 1);
    const bytesPerSample = Math.max(1, Math.ceil((image.bit_depth || 8) / 8));
    return image.width * image.height * depth * channels * bytesPerSample;
  }, [image]);

  if (!image) {
    return <div className="viewer-loading">Loading viewer...</div>;
  }

  return (
    <div className="viewer-screen">
      <div className="viewer-main">
        <ViewerHeader
          image={image}
          selectedAssetId={selectedAssetId}
          overlayManifestLoading={overlays.overlayManifestLoading}
          overlayManifestRefetching={overlays.overlayManifestRefetching}
          overlayUpdating={overlays.overlayUpdating}
          overlayBuildFailureCount={overlays.overlayBuildFailureCount}
          onBackToHome={handleBackToHome}
          onBackToExperiment={() => {
            if (!image.experiment_id) return;
            clearSelection();
            navigate(`/experiments/${encodeURIComponent(image.experiment_id)}`);
          }}
          onExport={() => setExportOpen(true)}
          onPixelSizeSaved={() => {
            void refetchImage();
          }}
        />
        <div className="viewer-canvas">
          <ImageViewer
            image={{
              ngffUrl: getAssetNgffUrl(image.id, null),
              width: image.width,
              height: image.height,
            }}
            className="viewer-container"
            viewport={{
              state: viewport ?? undefined,
              onChange: setViewport,
            }}
            overlays={{
              idMapOverlays: overlays.viewerIdMapOverlays,
            }}
          />
        </div>
      </div>
      {exportOpen ? (
        <ExportDialog
          asset={{ id: image.id, displayName: image.display_name }}
          onClose={() => setExportOpen(false)}
        />
      ) : null}
      <SegmentationDeleteDialog state={deleteState} />
      <OverlaySelectionSidebar
        overlays={overlays.overlayList}
        onToggle={overlays.handleToggle}
        onColorChange={overlays.handleColorChange}
        onOpacityChange={overlays.handleOpacityChange}
        onEditSegmentation={handleEditSegmentation}
        onDeleteSegmentation={deleteState.handleRequestDeleteSegmentation}
        overlayBuildFailures={overlays.overlayBuildFailures}
        onOverlayBuildRetried={overlays.handleOverlayBuildRetried}
        disabled={!imageReady}
        statusMessage={
          image && !imageReady
            ? `${getStageLabel(image.preprocess_stage)}${image.preprocess_progress > 0 ? ` (${Math.round(image.preprocess_progress)}%)` : ""}.\nOverlays are unavailable until preprocessing is complete.`
            : undefined
        }
        createPanel={
          selectedAssetId ? (
            <SegmentationCreatePanel
              imageId={selectedAssetId}
              onCreated={handleSegmentationCreated}
              existingSegmentationTypes={existingSegmentationTypes}
              existingSegmentationTypeIds={existingSegmentationTypeIds}
              title="Add Segmentation"
              imageSizeBytes={imageSizeBytes}
            />
          ) : null
        }
      />
    </div>
  );
}
