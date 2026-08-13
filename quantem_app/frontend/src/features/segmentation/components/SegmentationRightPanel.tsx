/**
 * Right panel component for confirmed-object view.
 */

import { useMemo } from "react";
import { ImageViewer } from "@/viewer/components/ImageViewer";
import { generateRightPanelOverlays } from "@/features/segmentation/overlays/segments";
import {
  generateDrawStrokeOverlays,
  generateRoiOverlays,
  type RoiStroke,
} from "@/features/segmentation/overlays/roi";
import type {
  ViewportState,
  SegmentOverlay,
  ViewerIdMapOverlaySpec,
  ViewerNgffOverlayLayerSpec,
} from "@/viewer/types";
import type { AssetDetail } from "@/shared/types/images";
import type { SegmentObject, SegmentationRoi } from "@/shared/types/segmentation";
import type { Point } from "@/utils/geometry";
import { getAssetNgffUrl } from "@/shared/api/assets";
import { composeOverlayScene } from "@/viewer/overlays/scene";
import "./SegmentationRightPanel.css";

export type RightPanelRemoveMode = "none" | "objects" | "area";

interface SegmentationRightPanelProps {
  image: AssetDetail;
  segmentationTypeInternalName?: string | null;
  useSmoothedGeometry: boolean;
  viewport: ViewportState | null;
  onViewportChange: (viewport: ViewportState) => void;
  confirmedSegments: SegmentObject[];
  tooManyRight: boolean;
  activeRoi: SegmentationRoi | null;
  rois: SegmentationRoi[];
  removeMode: RightPanelRemoveMode;
  onRemoveModeChange: (mode: RightPanelRemoveMode) => void;
  onRemoveObjectPointClick: (point: Point) => void;
  removeAreaBrushSize: number;
  onRemoveAreaBrushSizeChange: (size: number) => void;
  removeAreaBrushStrokes: RoiStroke[];
  onRemoveAreaBrushStroke: (points: Point[]) => void;
  canApplyRemoveArea: boolean;
  onApplyRemoveArea: () => void;
  removingArea: boolean;
  overlayNgffLayers?: ViewerNgffOverlayLayerSpec[];
  /** The ID-map segmentation review overlay (labels + border + render-time LUT). */
  idMapOverlay?: ViewerIdMapOverlaySpec | null;
  onOverlayRevisionDisplayed?: (revision: number | null) => void;
}

export function SegmentationRightPanel({
  image,
  segmentationTypeInternalName,
  useSmoothedGeometry,
  viewport,
  onViewportChange,
  confirmedSegments,
  tooManyRight,
  activeRoi,
  rois,
  removeMode,
  onRemoveModeChange,
  onRemoveObjectPointClick,
  removeAreaBrushSize,
  onRemoveAreaBrushSizeChange,
  removeAreaBrushStrokes,
  onRemoveAreaBrushStroke,
  canApplyRemoveArea,
  onApplyRemoveArea,
  removingArea,
  overlayNgffLayers = [],
  idMapOverlay = null,
  onOverlayRevisionDisplayed,
}: SegmentationRightPanelProps) {
  const rightPersistentOverlays = useMemo<SegmentOverlay[]>(
    () => {
      const overlays: SegmentOverlay[] = [...generateRoiOverlays(activeRoi, rois)];
      overlays.unshift(
        ...generateRightPanelOverlays(
          confirmedSegments,
          [],
          [],
          null,
          segmentationTypeInternalName,
          undefined,
          useSmoothedGeometry
        )
      );
      return overlays;
    },
    [
      activeRoi,
      rois,
      confirmedSegments,
      segmentationTypeInternalName,
      useSmoothedGeometry,
    ]
  );

  const removeAreaOverlays = useMemo<SegmentOverlay[]>(
    () =>
      generateDrawStrokeOverlays(removeAreaBrushStrokes).map((overlay, index) => ({
        ...overlay,
        id: `remove-area-${index}`,
        fillColor: "#f97316",
        fillOpacity: 0.28,
        strokeColor: "#ea580c",
      })),
    [removeAreaBrushStrokes]
  );

  const overlayScene = useMemo(
    () =>
      composeOverlayScene({
        persistentLayers: [rightPersistentOverlays],
        transientLayers:
          removeMode === "area" && removeAreaOverlays.length > 0
            ? [removeAreaOverlays]
            : [],
      }),
    [removeAreaOverlays, removeMode, rightPersistentOverlays]
  );
  const idMapOverlays = useMemo(
    () => (idMapOverlay ? [idMapOverlay] : []),
    [idMapOverlay]
  );

  return (
    <section className="seg-right confirmed-only">
      <div className="remove-tools-row">
        <button
          type="button"
          className={`remove-mode-button ${removeMode === "objects" ? "active" : ""}`}
          onClick={() =>
            onRemoveModeChange(removeMode === "objects" ? "none" : "objects")
          }
        >
          Remove objects
        </button>
        <button
          type="button"
          className={`remove-mode-button ${removeMode === "area" ? "active" : ""}`}
          onClick={() => onRemoveModeChange(removeMode === "area" ? "none" : "area")}
        >
          Remove area
        </button>
      </div>
      {removeMode === "objects" && (
        <div className="remove-area-hint">
          Click confirmed objects to move them back to candidate.
        </div>
      )}
      {removeMode === "area" && (
        <div className="remove-area-controls">
          <label htmlFor="remove-area-brush-size">
            Brush diameter
            <span>{Math.round(removeAreaBrushSize)} px</span>
          </label>
          <input
            id="remove-area-brush-size"
            type="range"
            min={4}
            max={128}
            step={1}
            value={removeAreaBrushSize}
            onChange={(event) => onRemoveAreaBrushSizeChange(Number(event.target.value))}
          />
        </div>
      )}
      {removeMode === "area" && (
        <div className="remove-area-hint">
          Draw over regions to subtract them from confirmed objects.
        </div>
      )}
      {canApplyRemoveArea && (
        <div className="remove-tools-row remove-area-apply-row">
          <button
            type="button"
            className="remove-area-apply-button"
            onClick={onApplyRemoveArea}
            disabled={removingArea}
          >
            {removingArea ? "Removing..." : "Remove"}
          </button>
          <span className="remove-area-hint">
            Press Enter or Space to apply. Press Delete to clear.
          </span>
        </div>
      )}
      {tooManyRight && (
        <div className="overlay-warning">
          Many confirmed shapes in view; rendering may be slower.
        </div>
      )}
      <ImageViewer
        image={{
          ngffUrl: getAssetNgffUrl(image.id, null),
          width: image.width,
          height: image.height,
        }}
        className="viewer-container"
        viewport={{
          state: viewport ?? undefined,
          onChange: onViewportChange,
        }}
        overlays={{
          persistent: overlayScene.persistent,
          transient: overlayScene.transient,
          rasterLayers: overlayNgffLayers,
          idMapOverlays,
          onRasterRevisionDisplayed: onOverlayRevisionDisplayed,
        }}
        interactions={{
          onImageClick:
            removeMode === "objects" ? onRemoveObjectPointClick : undefined,
          brush: {
            enabled: removeMode === "area",
            size: removeAreaBrushSize,
            color: "#f97316",
            onStroke: removeMode === "area" ? onRemoveAreaBrushStroke : undefined,
          },
        }}
      />
    </section>
  );
}
