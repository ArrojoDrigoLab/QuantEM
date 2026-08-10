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
import { useErPreviewStore } from "@/features/segmentation/erPreview/useErPreviewStore";
import { colorizeProb } from "@/features/segmentation/erPreview/overlayCanvas";
import "./SegmentationLeftPanel.css";

export type { SegmentationLeftPanelProps } from "@/features/segmentation/components/leftPanel/types";

export function SegmentationLeftPanel(props: SegmentationLeftPanelProps) {
  const overlayScene = useMemo(() => buildLeftPanelVectorScene(props), [props]);
  const erOverlay = useErPreviewStore((state) => state.overlay);
  const erThreshold = useErPreviewStore((state) => state.threshold);
  const erOpacity = useErPreviewStore((state) => state.opacity);
  const erCanvas = useMemo(
    () => (erOverlay ? colorizeProb(erOverlay, erThreshold, erOpacity) : null),
    [erOverlay, erThreshold, erOpacity]
  );
  const viewerProps = useMemo(() => {
    const config = buildLeftPanelViewerConfig({
      ...props,
      overlayScene,
    });
    if (erOverlay && erCanvas) {
      config.overlays = {
        ...config.overlays,
        bitmapOverlays: [
          {
            id: `er-preview-${erOverlay.sourceModel}-${erThreshold.toFixed(2)}-${erOpacity.toFixed(2)}`,
            image: erCanvas,
            bounds: erOverlay.bounds,
            opacity: 1,
          },
        ],
      };
    }
    return config;
  }, [overlayScene, props, erOverlay, erCanvas, erThreshold, erOpacity]);

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
