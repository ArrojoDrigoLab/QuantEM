/**
 * Left panel component for labeling view with interaction modes.
 */

import { useMemo } from "react";
import { ImageViewer } from "@/viewer/components/ImageViewer";
import { buildLeftPanelVectorScene } from "@/features/segmentation/components/leftPanel/buildLeftPanelVectorScene";
import { buildLeftPanelViewerConfig } from "@/features/segmentation/components/leftPanel/buildLeftPanelViewerConfig";
import { LeftPanelDrawingActions } from "@/features/segmentation/components/leftPanel/LeftPanelDrawingActions";
import { LeftPanelRoiSection } from "@/features/segmentation/components/leftPanel/LeftPanelRoiSection";
import { LeftPanelStatusMessage } from "@/features/segmentation/components/leftPanel/LeftPanelStatusMessage";
import { OverlayLayerMenu } from "@/features/segmentation/components/OverlayLayerMenu";
import type { SegmentationLeftPanelProps } from "@/features/segmentation/components/leftPanel/types";
import { useThresholdPreviewStore } from "@/features/segmentation/components/threshold/useThresholdPreviewStore";
import { colorizeProb } from "@/features/segmentation/erPreview/overlayCanvas";
import {
  selectSegmentGeometryCoords,
  selectSegmentHoleCoords,
} from "@/utils/segmentGeometry";
import "./SegmentationLeftPanel.css";

export type { SegmentationLeftPanelProps } from "@/features/segmentation/components/leftPanel/types";

export function SegmentationLeftPanel(props: SegmentationLeftPanelProps) {
  const overlayScene = useMemo(() => buildLeftPanelVectorScene(props), [props]);
  const thresholdOverlay = useThresholdPreviewStore((state) => state.overlay);
  const threshold = useThresholdPreviewStore((state) => state.threshold);
  const thresholdOpacity = useThresholdPreviewStore((state) => state.opacity);
  const confirmedMasks = useMemo(
    () =>
      props.segments.items
        .filter((segment) => segment.label_state === "CONFIRMED")
        .map((segment) => ({
          polygon_coords: selectSegmentGeometryCoords(segment, false),
          holes: selectSegmentHoleCoords(segment),
        })),
    [props.segments.items]
  );
  const thresholdCanvas = useMemo(
    () =>
      thresholdOverlay
          ? colorizeProb(thresholdOverlay, threshold, thresholdOpacity, {
            // The threshold map is a raw model view beneath accepted work.
            // Erase confirmed pixels from the bitmap itself as well as drawing
            // the green vectors later, so transparency settings can never let
            // red probability pixels bleed through a confirmed object.
            polygons: [...props.completedRoi.items, ...confirmedMasks],
            rectangles: props.roi.completedRois.map((roi) => ({
              x: roi.x,
              y: roi.y,
              width: roi.width,
              height: roi.height,
            })),
          })
        : null,
    [
      props.completedRoi.items,
      confirmedMasks,
      props.roi.completedRois,
      threshold,
      thresholdOpacity,
      thresholdOverlay,
    ]
  );
  const viewerProps = useMemo(() => {
    const config = buildLeftPanelViewerConfig({
      ...props,
      overlayScene,
    });
    if (thresholdOverlay && thresholdCanvas) {
      config.overlays = {
        ...config.overlays,
        bitmapOverlays: [
          {
            id: `threshold-preview-${thresholdOverlay.sourceModel}-${threshold.toFixed(2)}`,
            image: thresholdCanvas,
            bounds: thresholdOverlay.bounds,
            opacity: 1,
          },
        ],
      };
    }
    return config;
  }, [overlayScene, props, thresholdOverlay, thresholdCanvas, threshold]);

  return (
    <section className="seg-left">
      <LeftPanelStatusMessage
        workflow={props.workflow}
        uncertain={props.uncertain}
        tooMany={props.segments.tooMany}
      />
      <LeftPanelDrawingActions
        drawing={props.drawing}
        completedRoi={props.completedRoi}
      />
      <LeftPanelRoiSection workflow={props.workflow} roi={props.roi} />
      <div className="left-viewer-stage">
        <ImageViewer {...viewerProps} />
        <OverlayLayerMenu
          idPrefix="left-pane"
          paneLabel="Left pane"
          {...props.layerControls}
        />
      </div>
    </section>
  );
}
