import type { Point } from "@/utils/geometry";

export interface ViewportState {
  centerX: number;
  centerY: number;
  zoom: number;
  containerWidth: number;
  containerHeight: number;
}

export interface SegmentOverlay {
  id: string;
  geometry: Point[];
  /**
   * Optional interior rings (holes) cut out of the polygon. When present the
   * overlay is rendered as an even-odd SVG path so the holes are visibly
   * excluded from the fill. Ignored for polylines.
   */
  holes?: Point[][];
  fillColor: string;
  fillOpacity: number;
  strokeColor: string;
  strokeOpacity: number;
  strokeWidth?: number;
  /** SVG dash pattern, for example "8 6". Omitted for a solid outline. */
  strokeDasharray?: string;
  shape?: "polygon" | "polyline";
}

export interface ViewerFitBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ViewerNgffOverlayLayerSpec {
  id: string;
  ngffUrl: string;
  color: string;
  channelColors?: string[];
  opacity: number;
  channelIndices: number[];
  renderMode?: "tint" | "rgb";
}

/**
 * An ID-map overlay: an integer `labels` raster + `border` mask coloured at
 * render time by a label -> RGBA LUT. Replaces the per-state channel specs for
 * the segmentation overlay. `ngffUrl` points at the bundle root (the layer
 * appends `/labels` and `/border`); the LUT is supplied separately as a texture.
 */
export interface ViewerIdMapOverlaySpec {
  id: string;
  /** Bundle root zarr URL (already carries the `?rev=<bundle_version>` cache key). */
  ngffUrl: string;
  /** Geometry revision represented by this bundle. */
  revision?: number;
  /** Flat RGBA8 palette indexed by dense label (length = (maxLabel + 1) * 4). */
  lut: Uint8Array;
  maxLabel: number;
  /** Bumps when the LUT content changes, so the layer re-uploads the palette. */
  lutRevision: number;
  fillOpacity: number;
  borderOpacity: number;
  showBorders: boolean;
  /**
   * Dense label -> object uuid map. When present the raster becomes pickable:
   * clicking resolves the label under the cursor back to its object id, so
   * individual objects stay selectable without rendering vectors.
   */
  pickMap?: Map<number, string>;
}

/**
 * A transient single-image (PNG/data-URL) overlay drawn as a deck.gl BitmapLayer
 * at the given image-pixel bounds. Used for fast, replaceable model previews.
 */
export interface ViewerBitmapOverlaySpec {
  id: string;
  image: string | HTMLCanvasElement | HTMLImageElement | ImageBitmap | ImageData;
  bounds: [number, number, number, number]; // [x, y, width, height] in image px
  opacity: number;
}
