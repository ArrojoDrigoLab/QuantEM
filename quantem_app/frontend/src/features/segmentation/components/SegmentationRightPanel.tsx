/**
 * Right panel component for confirmed-object view.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
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
import {
  OverlayLayerMenu,
  type PaneOverlayLayerControls,
} from "@/features/segmentation/components/OverlayLayerMenu";
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
  onRemoveObjectClick: (segmentId: string | null) => void;
  removeAreaBrushSize: number;
  onRemoveAreaBrushSizeChange: (size: number) => void;
  removeAreaBrushStrokes: RoiStroke[];
  onRemoveAreaBrushStroke: (points: Point[]) => void;
  canApplyRemoveArea: boolean;
  onApplyRemoveArea: () => void;
  removingArea: boolean;
  overlayNgffLayers?: ViewerNgffOverlayLayerSpec[];
  /** State-specific ID-map overlays (labels + border + render-time LUT). */
  idMapOverlays?: ViewerIdMapOverlaySpec[];
  layerControls: PaneOverlayLayerControls;
  onOverlayRevisionDisplayed?: (revision: number | null) => void;
  confirmingObjects?: boolean;
  confirmationCommitted?: boolean;
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
  onRemoveObjectClick,
  removeAreaBrushSize,
  onRemoveAreaBrushSizeChange,
  removeAreaBrushStrokes,
  onRemoveAreaBrushStroke,
  canApplyRemoveArea,
  onApplyRemoveArea,
  removingArea,
  overlayNgffLayers = [],
  idMapOverlays = [],
  layerControls,
  onOverlayRevisionDisplayed,
  confirmingObjects = false,
  confirmationCommitted = false,
}: SegmentationRightPanelProps) {
  const [removeObjectHover, setRemoveObjectHover] = useState({
    segmentId: null as string | null,
    revision: 0,
  });
  const handleRemoveObjectHover = useCallback((segmentId: string | null) => {
    const objectId = segmentId?.startsWith("roi-frame") ? null : segmentId;
    setRemoveObjectHover((previous) =>
      previous.segmentId === objectId
        ? previous
        : { segmentId: objectId, revision: previous.revision + 1 }
    );
  }, []);

  useEffect(() => {
    if (removeMode !== "objects") handleRemoveObjectHover(null);
  }, [handleRemoveObjectHover, removeMode]);

  const rightPersistentOverlays = useMemo<SegmentOverlay[]>(
    () => {
      const overlays: SegmentOverlay[] = [...generateRoiOverlays(activeRoi, rois)];
      overlays.unshift(
        ...generateRightPanelOverlays(
          confirmedSegments,
          [],
          [],
          removeMode === "objects" ? removeObjectHover.segmentId : null,
          segmentationTypeInternalName,
          undefined,
          useSmoothedGeometry,
          {
            strokeWidth: layerControls.confirmed.strokeWidth,
            fillOpacity: layerControls.confirmed.fillOpacity,
          }
        )
      );
      return overlays;
    },
    [
      activeRoi,
      rois,
      confirmedSegments,
      removeMode,
      removeObjectHover.segmentId,
      segmentationTypeInternalName,
      useSmoothedGeometry,
      layerControls.confirmed.fillOpacity,
      layerControls.confirmed.strokeWidth,
    ]
  );

  const displayedIdMapOverlays = useMemo(
    () =>
      idMapOverlays.map((overlay) => ({
        ...overlay,
        highlightedSegmentId:
          removeMode === "objects" ? removeObjectHover.segmentId : null,
        highlightRevision: removeObjectHover.revision,
      })),
    [idMapOverlays, removeMode, removeObjectHover]
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
  return (
    <section className="seg-right confirmed-only" aria-busy={confirmingObjects}>
      <div className="remove-tools-row">
        <button
          type="button"
          className={`remove-mode-button ${removeMode === "objects" ? "active" : ""}`}
          onClick={() => {
            const nextMode = removeMode === "objects" ? "none" : "objects";
            if (nextMode !== "objects") handleRemoveObjectHover(null);
            onRemoveModeChange(nextMode);
          }}
        >
          Remove objects
        </button>
        <button
          type="button"
          className={`remove-mode-button ${removeMode === "area" ? "active" : ""}`}
          onClick={() => {
            handleRemoveObjectHover(null);
            onRemoveModeChange(removeMode === "area" ? "none" : "area");
          }}
        >
          Remove area
        </button>
      </div>
      {removeMode === "objects" && (
        <div className="remove-area-hint">
          Hover to highlight. Click once to permanently delete an object.
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
      <div className="right-viewer-stage">
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
            idMapOverlays: displayedIdMapOverlays,
            onRasterRevisionDisplayed: onOverlayRevisionDisplayed,
          }}
          interactions={{
            onShapeHover:
              removeMode === "objects" ? handleRemoveObjectHover : undefined,
            onShapeClick:
              removeMode === "objects"
                ? (segmentId) => {
                    handleRemoveObjectHover(null);
                    onRemoveObjectClick(
                      segmentId?.startsWith("roi-frame") ? null : segmentId
                    );
                  }
                : undefined,
            brush: {
              enabled: removeMode === "area",
              size: removeAreaBrushSize,
              color: "#f97316",
              onStroke: removeMode === "area" ? onRemoveAreaBrushStroke : undefined,
            },
          }}
        />
        <OverlayLayerMenu
          idPrefix="right-pane"
          paneLabel="Right pane"
          {...layerControls}
          candidates={undefined}
        />
        {confirmingObjects ? (
          <div
            className="confirming-objects-veil"
            role="status"
            aria-live="polite"
          >
            <span className="confirming-objects-spinner" aria-hidden="true" />
            <span>
              {confirmationCommitted
                ? "Objects are confirmed and ready for analysis. Updating this display…"
                : "Saving confirmed objects…"}
            </span>
          </div>
        ) : null}
      </div>
    </section>
  );
}
