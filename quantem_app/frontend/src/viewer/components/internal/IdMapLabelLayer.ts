import { CompositeLayer, type Layer, type LayerContext, type UpdateParameters } from "@deck.gl/core";
import type { Texture } from "@luma.gl/core";
import { BitmapLayer } from "@deck.gl/layers";
import { MultiscaleImageLayer } from "@hms-dbmi/viv";
import type { VivLoaderData } from "@/viewer/components/internal/vivUtils";
import { PaletteBitmapLayer, NEAREST_SAMPLER } from "./PaletteBitmapLayer";
import { accountOverlayTile, overlayTileBudget } from "./overlayTileBudget";

/**
 * Renders an ID-map segmentation overlay: an integer `labels` raster coloured at
 * render time by a label -> RGBA LUT.
 *
 * Tiling/zoom/coordinates are delegated to viv's `MultiscaleImageLayer` (so it
 * fetches exactly the levels+tiles the viewport needs and aligns with the base
 * image). Integer labels are uploaded once and colored by a shared GPU palette.
 * Baked borders include neighboring pixels across tile seams. A CPU fallback
 * handles palettes exceeding the device's texture dimensions.
 */
const BORDER_DARKEN = 0.5;

/** Match GPU visibility when the palette is too large for a texture. */
class ColorizedBitmapLayer extends BitmapLayer {
  static layerName = "ColorizedBitmapLayer";

  getShaders() {
    const shaders = super.getShaders();
    return {
      ...shaders,
      inject: {
        ...shaders.inject,
        "fs:DECKGL_FILTER_COLOR": "if (color.a <= 0.0) discard;",
      },
    };
  }
}

// Stable references for the props viv forwards to its inner TileLayer. viv puts
// `selections` (and `loader`) in `updateTriggers.getTileData`, which deck
// compares by reference -- a fresh array literal each render would invalidate
// the whole tile cache and refetch every tile forever. We bypass viv's XRLayer
// via renderSubLayers, so the contrast/colour/channel values are unused; only
// their reference stability matters.
const LABEL_SELECTIONS = [{ overlayArray: 0 }, { overlayArray: 1 }];
const CHANNELS_VISIBLE = [true];
const CHANNEL_COLORS: Array<[number, number, number]> = [[255, 255, 255]];
const NO_EXTENSIONS: never[] = [];
const paletteIds = new WeakMap<Uint8Array, number>();
const reversePickMaps = new WeakMap<Map<number, string>, Map<string, number>>();
function highlightedLabelFor(props: IdMapLabelLayerProps): number {
  if (!props.highlightedSegmentId || !props.pickMap) return 0;
  let reverse = reversePickMaps.get(props.pickMap);
  if (!reverse) {
    reverse = new Map(Array.from(props.pickMap, ([label, uuid]) => [uuid, label]));
    reversePickMaps.set(props.pickMap, reverse);
  }
  return reverse.get(props.highlightedSegmentId) ?? 0;
}
let nextPaletteId = 1;
function paletteId(lut: Uint8Array): number {
  let id = paletteIds.get(lut);
  if (!id) { id = nextPaletteId++; paletteIds.set(lut, id); }
  return id;
}

export interface IdMapLabelLayerProps {
  id: string;
  /** viv loader sources for the `labels` pyramid (index 0 = full resolution). */
  labelsData: VivLoaderData;
  /** Baked border pyramid, paired with labels by combineOverlaySources. */
  borderData?: VivLoaderData | null;
  /** Flat RGBA8 palette indexed by dense label. */
  lut: Uint8Array;
  maxLabel: number;
  /** Bumps when the LUT content changes (forces re-colorization). */
  lutRevision: number;
  /** Bumps for client-only LUT changes, such as an optimistic hard delete. */
  visualRevision: number;
  /** Object UUID highlighted locally in this pane. */
  highlightedSegmentId: string | null;
  /** Bumps when highlightedSegmentId changes. */
  highlightRevision: number;
  imageWidth: number;
  imageHeight: number;
  fillOpacity: number;
  borderOpacity: number;
  showBorders: boolean;
  /** Dense label -> object uuid; when present, the raster tiles become pickable. */
  pickMap?: Map<number, string>;
  onViewportLoad?: () => void;
  onTileError?: (error: Error) => void;
}

interface VivTileData {
  data: Array<Uint32Array | Uint16Array | Uint8Array>;
  width: number;
  height: number;
  _packedLabels?: { data: Uint8Array; width: number; height: number };
  _border?: { data: Uint8Array; width: number; height: number };
  /** One source tile may feed candidate and confirmed layers with different LUTs. */
  _idmapCache?: Record<string, { signature: string; lut: Uint8Array; pickMap?: Map<number, string>; image: ImageData }>;
}

export class IdMapLabelLayer extends CompositeLayer<IdMapLabelLayerProps> {
  static layerName = "IdMapLabelLayer";
  declare state: { paletteTexture?: Texture; resourceOwner?: symbol };

  initializeState() {
    this.setState({ resourceOwner: Symbol("overlay-generation") });
  }

  updateState({ props, oldProps }: UpdateParameters<this>) {
    if (this.state.resourceOwner) {
      overlayTileBudget.register(this.state.resourceOwner, () => this.setNeedsUpdate());
    }
    if (props.lut === oldProps.lut && props.maxLabel === oldProps.maxLabel && this.state.paletteTexture) return;
    this.state.paletteTexture?.destroy();
    const limit = this.context.device.limits.maxTextureDimension2D;
    const width = Math.min(1024, limit, Math.max(1, props.maxLabel + 1));
    const height = Math.ceil((props.maxLabel + 1) / width);
    if (height > limit) { this.setState({ paletteTexture: undefined }); return; }
    const data = new Uint8Array(width * height * 4);
    data.set(props.lut.subarray(0, data.length));
    this.setState({ paletteTexture: this.context.device.createTexture({
      data, width, height, format: "rgba8unorm", sampler: NEAREST_SAMPLER,
    }) });
  }

  finalizeState(context: LayerContext) {
    if (this.state.resourceOwner) overlayTileBudget.release(this.state.resourceOwner);
    this.state.paletteTexture?.destroy();
    super.finalizeState(context);
  }

  private _colorize(tile: VivTileData): ImageData {
    const {
      lut,
      maxLabel,
      lutRevision,
      visualRevision,
      highlightedSegmentId,
      highlightRevision,
      fillOpacity,
      borderOpacity,
      showBorders,
    } = this.props;
    const sig = `${maxLabel}|${lutRevision}|${visualRevision}|${highlightRevision}|${highlightedSegmentId ?? ""}|${fillOpacity}|${borderOpacity}|${showBorders}`;
    const cached = tile._idmapCache?.[this.props.id];
    if (cached?.signature === sig && cached.lut === lut && cached.pickMap === this.props.pickMap) return cached.image;

    const labels = tile.data[0];
    const { width, height } = tile;
    const pixelCount = width * height;
    const out = new Uint8ClampedArray(pixelCount * 4);
    const fillAlpha = Math.round(Math.min(1, Math.max(0, fillOpacity)) * 255);
    const borderAlpha = Math.round(Math.min(1, Math.max(0, borderOpacity)) * 255);
    const highlightedLabel = highlightedLabelFor(this.props);

    for (let i = 0; i < pixelCount; i += 1) {
      const id = labels[i];
      if (id === 0 || id > maxLabel) continue;
      const base = id * 4;
      const alpha = lut[base + 3];
      if (alpha === 0) continue; // hidden by per-state default visibility

      const isHighlighted = highlightedLabel !== 0 && id === highlightedLabel;
      let isBorder = false;
      if ((showBorders || isHighlighted) && tile.data[1]) {
        isBorder = tile.data[1][i] !== 0;
      } else if (showBorders || isHighlighted) {
        const x = i % width;
        const y = (i - x) / width;
        isBorder =
          (x > 0 && labels[i - 1] !== id) ||
          (x < width - 1 && labels[i + 1] !== id) ||
          (y > 0 && labels[i - width] !== id) ||
          (y < height - 1 && labels[i + width] !== id);
      }
      const out4 = i * 4;
      if (isHighlighted) {
        out[out4] = 0;
        out[out4 + 1] = 255;
        out[out4 + 2] = 255;
        out[out4 + 3] = isBorder ? 255 : Math.max(fillAlpha, 64);
      } else if (isBorder) {
        out[out4] = lut[base] * BORDER_DARKEN;
        out[out4 + 1] = lut[base + 1] * BORDER_DARKEN;
        out[out4 + 2] = lut[base + 2] * BORDER_DARKEN;
        out[out4 + 3] = borderAlpha;
      } else {
        out[out4] = lut[base];
        out[out4 + 1] = lut[base + 1];
        out[out4 + 2] = lut[base + 2];
        out[out4 + 3] = fillAlpha;
      }
    }
    const image = new ImageData(out, width, height);
    tile._idmapCache ??= {};
    tile._idmapCache[this.props.id] = { signature: sig, lut, pickMap: this.props.pickMap, image };
    return image;
  }

  // viv's MultiscaleImageLayer passes this through to its inner TileLayer,
  // overriding viv's default per-tile renderer. deck types these props loosely.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _renderTile(props: any): Layer | null {
    const tile = props.data as VivTileData | null;
    if (!tile || !tile.data || !tile.data[0]) return null;
    const {
      bbox: { left, top, right, bottom },
      index,
    } = props.tile;

    const base = this.props.labelsData[0] as unknown as { tileSize?: number };
    const tileSize = base?.tileSize ?? 256;
    // Match viv's edge-tile bounds: tiles smaller than tileSize clamp to the
    // full-image extent so partial edge tiles align.
    const bounds: [number, number, number, number] = [
      left,
      tile.height < tileSize ? this.props.imageHeight : bottom,
      tile.width < tileSize ? this.props.imageWidth : right,
      top,
    ];
    const pickable = Boolean(this.props.pickMap);
    if (this.state?.paletteTexture && tile.data[1]) {
      if (!tile._packedLabels) {
        const data = new Uint8Array(tile.width * tile.height * 4);
        for (let i = 0; i < tile.data[0].length; i++) {
          const label = tile.data[0][i];
          data[i * 4] = label & 255;
          data[i * 4 + 1] = (label >>> 8) & 255;
          data[i * 4 + 2] = (label >>> 16) & 255;
          data[i * 4 + 3] = label >>> 24;
        }
        tile._packedLabels = { data, width: tile.width, height: tile.height };
        tile._border = { data: tile.data[1] as Uint8Array, width: tile.width, height: tile.height };
      }
      const highlightedLabel = highlightedLabelFor(this.props);
      return new PaletteBitmapLayer(props, {
        id: `${this.props.id}-bitmap-${index.x}-${index.y}-${index.z}`,
        image: tile._packedLabels, border: tile._border!, bounds, pickable,
        palette: this.state.paletteTexture, maxLabel: this.props.maxLabel, highlightedLabel,
        fillOpacity: this.props.fillOpacity, borderOpacity: this.props.borderOpacity,
        showBorders: this.props.showBorders, textureParameters: NEAREST_SAMPLER,
        idLabels: tile.data[0], idTileWidth: tile.width,
      });
    }
    return new ColorizedBitmapLayer(props, {
      // Deck matches layers by ID even when their class changes. Give the
      // fallback its own ID so a switch cannot reuse the palette shader state.
      id: `${this.props.id}-bitmap-cpu-${index.x}-${index.y}-${index.z}`,
      image: this._colorize(tile),
      bounds,
      pickable,
      textureParameters: NEAREST_SAMPLER,
      // Retain the raw label tile + width so getPickingInfo can resolve the
      // clicked pixel back to a dense label (the image itself is colorized RGBA).
      ...(pickable ? { idLabels: tile.data[0], idTileWidth: tile.width } : {}),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
  }

  // Resolve a pick on a colorized tile back to its object: bitmap pixel -> dense
  // label (from the retained label tile) -> uuid (from pickMap).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  getPickingInfo({ info }: { info: any }): any {
    // Deck replaces sourceLayer at each composite boundary. TileLayer retains
    // the actual bitmap under sourceTileSubLayer while Viv forwards the pick.
    const sourceProps = info?.sourceTileSubLayer?.props ?? info?.sourceLayer?.props;
    const labels = sourceProps?.idLabels as Uint32Array | Uint16Array | Uint8Array | undefined;
    const width = sourceProps?.idTileWidth as number | undefined;
    if (info?.bitmap?.pixel && labels && width) {
      const [px, py] = info.bitmap.pixel;
      if (px >= 0 && py >= 0 && px < width) {
        const label = labels[py * width + px];
        if (label > 0 && label <= this.props.maxLabel && this.props.lut[label * 4 + 3] !== 0) {
          info.object = { label, uuid: this.props.pickMap?.get(label) ?? null };
          return info;
        }
      }
    }
    info.object = null;
    return info;
  }

  renderLayers(): Layer | null {
    const { labelsData } = this.props;
    if (!labelsData || labelsData.length === 0) return null;
    const source = labelsData[0] as unknown as { dtype: string };
    const renderRevision: Array<[number, number]> = [
      [this.props.lutRevision, this.props.fillOpacity],
      [this.props.borderOpacity, this.props.showBorders ? 1 : 0],
      [this.props.visualRevision, 0],
      [this.props.highlightRevision, 0],
      [paletteId(this.props.lut), this.props.maxLabel],
    ];

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const subLayerProps: any = this.getSubLayerProps({
      id: "viv-labels",
      loader: labelsData,
      selections: LABEL_SELECTIONS,
      dtype: source.dtype,
      // Viv 0.19 replaces caller-supplied updateTriggers with its own
      // getTileData trigger. `contrastLimits` is a compare:true prop on both
      // Viv layers that our Bitmap renderer otherwise ignores, so carrying the
      // render revision here redraws cached sublayers without refetching tiles.
      contrastLimits: renderRevision,
      channelsVisible: CHANNELS_VISIBLE,
      colors: CHANNEL_COLORS,
      opacity: 1,
      // Suppress viv's low-res Background-Image layer: it renders via XRLayer
      // (uncoloured white/grayscale wash) and would bypass our colorization.
      excludeBackground: true,
      // Override Viv's grayscale renderer with label-aware palette shading.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      renderSubLayers: (props: any) => this._renderTile(props),
      // Bypass viv's default ColorPaletteExtension; we colour in renderSubLayers.
      extensions: NO_EXTENSIONS,
      onViewportLoad: this.props.onViewportLoad,
      onTileError: this.props.onTileError,
      onTileLoad: accountOverlayTile,
      maxCacheByteSize: overlayTileBudget.bytesPerLayer,
      maxCacheSize: 128,
    });
    return new MultiscaleImageLayer(subLayerProps) as unknown as Layer;
  }
}
