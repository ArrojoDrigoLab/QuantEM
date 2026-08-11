import type { Point } from "@/utils/geometry";
import type {
  SegmentOverlay,
  ViewerBitmapOverlaySpec,
  ViewerFitBounds,
  ViewerIdMapOverlaySpec,
  ViewerNgffOverlayLayerSpec,
  ViewportState,
} from "@/viewer/types";

export interface ImageViewerImageConfig {
  ngffUrl?: string;
  width?: number;
  height?: number;
  /**
   * Number of stored z-planes; when > 1 (or the loaded NGFF has a z axis) the
   * viewer shows a z-slider. Optional — the viewer also infers depth from the
   * loaded store.
   */
  storedDepth?: number | null;
  /** Original source plane index per stored slice, for true-depth labels. */
  zPlaneIndices?: number[];
  /**
   * Physical width of one image pixel, in nanometres.
   *
   * Drives the scale bar, and only the scale bar. Absent or null means the
   * image has no calibration, and the canvas then shows no bar at all rather
   * than one that would silently mean pixels.
   */
  pixelSizeNm?: number | null;
}

export interface ImageViewerViewportConfig {
  state?: ViewportState;
  initialState?: ViewportState;
  fitBounds?: ViewerFitBounds | null;
  fitBoundsKey?: string | null;
  fitBoundsPaddingRatio?: number;
  disablePan?: boolean;
  onChange?: (viewport: ViewportState) => void;
  /**
   * Show the Fit / 1:1 / Reset controls and the scale bar. Defaults to true --
   * every canvas needs a way back to the whole image. Set false for a viewer
   * that is a thumbnail rather than a workspace.
   */
  showControls?: boolean;
}

export interface ImageViewerOverlayConfig {
  persistent?: SegmentOverlay[];
  transient?: SegmentOverlay[];
  /** Generic numeric-channel raster overlays (model runs, refinement, membrane). */
  rasterLayers?: ViewerNgffOverlayLayerSpec[];
  /** ID-map segmentation overlays (labels + border + render-time LUT). */
  idMapOverlays?: ViewerIdMapOverlaySpec[];
  bitmapOverlays?: ViewerBitmapOverlaySpec[];
  onRasterRevisionDisplayed?: (revision: number | null) => void;
}

export interface ImageViewerInteractionConfig {
  onImageClick?: (point: Point) => void;
  onImagePress?: (point: Point, screenPoint: Point) => void;
  onImageDrag?: (point: Point, screenPoint: Point) => void;
  onImageRelease?: (point: Point, screenPoint: Point) => void;
  onImageMove?: (point: Point, screenPoint: Point) => void;
  onImageMouseMove?: (point: Point) => void;
  onImageMouseLeave?: () => void;
  onShapeClick?: (segmentId: string | null) => void;
  draw?: {
    enabled?: boolean;
    onComplete?: (points: Point[]) => void;
  };
  brush?: {
    enabled?: boolean;
    size?: number;
    color?: string;
    onStroke?: (points: Point[]) => void;
  };
}

export interface ImageViewerHighlightingConfig {
  highlightedSegmentId?: string | null;
  hoverBadge?: { point: Point | null; count: number };
  hoverCursor?: boolean;
  cursorMode?: "target";
}

export interface ImageViewerProps {
  image: ImageViewerImageConfig;
  className?: string;
  viewport?: ImageViewerViewportConfig;
  overlays?: ImageViewerOverlayConfig;
  interactions?: ImageViewerInteractionConfig;
  highlighting?: ImageViewerHighlightingConfig;
}
