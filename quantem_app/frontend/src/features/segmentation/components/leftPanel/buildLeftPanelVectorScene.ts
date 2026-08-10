import {
  generateCompletedRoiDraftOverlays,
  generateCompletedRoiOverlays,
} from "@/features/segmentation/overlays/completedRois";
import {
  generatePendingPolygonOverlay,
  generateSelectionBBoxOverlay,
} from "@/features/segmentation/overlays/draft";
import {
  generateDrawStrokeOverlays,
  generateRoiOverlays,
  generateRoiStrokeOverlays,
} from "@/features/segmentation/overlays/roi";
import { generateLeftPanelOverlays } from "@/features/segmentation/overlays/segments";
import { generateUserFeedbackPointOverlays } from "@/features/segmentation/overlays/feedback";
import { composeOverlayScene } from "@/viewer/overlays/scene";
import type { SegmentationLeftPanelProps } from "@/features/segmentation/components/leftPanel/types";

type LeftPanelSceneArgs = Pick<
  SegmentationLeftPanelProps,
  | "viewer"
  | "segments"
  | "roi"
  | "drawing"
  | "completedRoi"
  | "feedback"
  | "overlays"
>;

export function buildLeftPanelVectorScene({
  viewer,
  segments,
  roi,
  drawing,
  completedRoi,
  feedback,
  overlays,
}: LeftPanelSceneArgs) {
  const groupHighlightedSet = new Set(segments.groupHighlightedSegmentIds);
  const persistentOverlays = generateLeftPanelOverlays(
    segments.items,
    segments.tooMany,
    viewer.segmentationTypeInternalName,
    groupHighlightedSet,
    viewer.useSmoothedGeometry,
    viewer.layerStyles
  );

  const roiOverlays = overlays.hideActiveRoiOverlay ? [] : generateRoiOverlays(roi.activeRoi);
  const transientLayers = [
    roiOverlays,
    generateRoiStrokeOverlays(roi.roiStrokes),
    generateDrawStrokeOverlays(drawing.brushStrokes),
    completedRoi.active ? generateCompletedRoiOverlays(completedRoi.items) : [],
    completedRoi.active
      ? generateCompletedRoiDraftOverlays(
          completedRoi.polygons,
          completedRoi.liveSectionPoints,
          completedRoi.mode
        )
      : [],
    generateUserFeedbackPointOverlays(feedback.items),
  ];

  if (segments.groupSelectionBBox) {
    const selectionOverlay = generateSelectionBBoxOverlay(segments.groupSelectionBBox);
    if (selectionOverlay) {
      transientLayers.push([selectionOverlay]);
    }
  }
  if (drawing.pendingPolygon) {
    const pendingPolygonOverlay = generatePendingPolygonOverlay(drawing.pendingPolygon);
    if (pendingPolygonOverlay) {
      transientLayers.push([pendingPolygonOverlay]);
    }
  }
  if (overlays.extraTransientOverlays && overlays.extraTransientOverlays.length > 0) {
    transientLayers.push(overlays.extraTransientOverlays);
  }

  return composeOverlayScene({
    persistentLayers: [persistentOverlays],
    transientLayers,
  });
}
