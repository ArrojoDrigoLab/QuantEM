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
import type { SegmentationLeftPanelProps } from "@/features/segmentation/components/leftPanel/types";
import { useThresholdPreviewStore } from "@/features/segmentation/components/threshold/useThresholdPreviewStore";
import { colorizeProb } from "@/features/segmentation/erPreview/overlayCanvas";
import "./SegmentationLeftPanel.css";

export type { SegmentationLeftPanelProps } from "@/features/segmentation/components/leftPanel/types";

export function SegmentationLeftPanel(props: SegmentationLeftPanelProps) {
  const overlayScene = useMemo(() => buildLeftPanelVectorScene(props), [props]);
  const thresholdOverlay = useThresholdPreviewStore((state) => state.overlay);
  const threshold = useThresholdPreviewStore((state) => state.threshold);
  const thresholdOpacity = useThresholdPreviewStore((state) => state.opacity);
  const thresholdCanvas = useMemo(
    () =>
      thresholdOverlay
        ? colorizeProb(thresholdOverlay, threshold, thresholdOpacity, {
            polygons: props.completedRoi.items,
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
      <ImageViewer {...viewerProps} />
    </section>
  );
}
