import { CompositeLayer, type Layer } from "@deck.gl/core";
import { BitmapLayer } from "@deck.gl/layers";
import { MultiscaleImageLayer } from "@hms-dbmi/viv";
import type { VivLoaderData } from "@/viewer/components/internal/vivUtils";

/**
 * Renders an ID-map segmentation overlay: an integer `labels` raster coloured at
 * render time by a label -> RGBA LUT.
 *
 * Tiling/zoom/coordinates are delegated to viv's `MultiscaleImageLayer` (so it
 * fetches exactly the levels+tiles the viewport needs and aligns with the base
 * image); we only override its per-tile `renderSubLayers` to map the decoded
 * label tile to RGBA on the CPU via the LUT and draw it with a `BitmapLayer`.
 * Borders are computed per-tile from label adjacency. Recolouring (LUT change)
 * is a cheap re-colorization with no tile refetch.
 */
const BORDER_DARKEN = 0.5;

// Stable references for the props viv forwards to its inner TileLayer. viv puts
// `selections` (and `loader`) in `updateTriggers.getTileData`, which deck
// compares by reference -- a fresh array literal each render would invalidate
// the whole tile cache and refetch every tile forever. We bypass viv's XRLayer
// via renderSubLayers, so the contrast/colour/channel values are unused; only
// their reference stability matters.
const LABEL_SELECTIONS = [{}];
const CHANNELS_VISIBLE = [true];
const CHANNEL_COLORS: Array<[number, number, number]> = [[255, 255, 255]];
const CONTRAST_LIMITS: Array<[number, number]> = [[0, 1]];
const NO_EXTENSIONS: never[] = [];

export interface IdMapLabelLayerProps {
  id: string;
  /** viv loader sources for the `labels` pyramid (index 0 = full resolution). */
  labelsData: VivLoaderData;
  /** Reserved for a future baked-border path; unused (borders are per-tile). */
  borderData?: VivLoaderData | null;
  /** Flat RGBA8 palette indexed by dense label. */
  lut: Uint8Array;
  maxLabel: number;
  /** Bumps when the LUT content changes (forces re-colorization). */
  lutRevision: number;
  imageWidth: number;
  imageHeight: number;
  fillOpacity: number;
  borderOpacity: number;
  showBorders: boolean;
  /** Dense label -> object uuid; when present, the raster tiles become pickable. */
  pickMap?: Map<number, string>;
}

interface VivTileData {
  data: Array<Uint32Array | Uint16Array | Uint8Array>;
  width: number;
  height: number;
  _idmapSig?: string;
  _idmapImg?: ImageData;
}

export class IdMapLabelLayer extends CompositeLayer<IdMapLabelLayerProps> {
  static layerName = "IdMapLabelLayer";

  private _colorize(tile: VivTileData): ImageData {
    const { lut, maxLabel, lutRevision, fillOpacity, borderOpacity, showBorders } = this.props;
    const sig = `${lutRevision}|${fillOpacity}|${borderOpacity}|${showBorders}`;
    if (tile._idmapSig === sig && tile._idmapImg) return tile._idmapImg;

    const labels = tile.data[0];
    const { width, height } = tile;
    const pixelCount = width * height;
    const out = new Uint8ClampedArray(pixelCount * 4);
    const fillAlpha = Math.round(Math.min(1, Math.max(0, fillOpacity)) * 255);
    const borderAlpha = Math.round(Math.min(1, Math.max(0, borderOpacity)) * 255);

    for (let i = 0; i < pixelCount; i += 1) {
      const id = labels[i];
      if (id === 0 || id > maxLabel) continue;
      const base = id * 4;
      const alpha = lut[base + 3];
      if (alpha === 0) continue; // hidden by per-state default visibility

      let isBorder = false;
      if (showBorders) {
        const x = i % width;
        const y = (i - x) / width;
        isBorder =
          (x > 0 && labels[i - 1] !== id) ||
          (x < width - 1 && labels[i + 1] !== id) ||
          (y > 0 && labels[i - width] !== id) ||
          (y < height - 1 && labels[i + width] !== id);
      }
      const out4 = i * 4;
      if (isBorder) {
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
    tile._idmapSig = sig;
    tile._idmapImg = image;
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
    return new BitmapLayer(props, {
      id: `${this.props.id}-bitmap-${index.x}-${index.y}-${index.z}`,
      image: this._colorize(tile),
      bounds,
      pickable,
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
    const sourceProps = info?.sourceLayer?.props;
    const labels = sourceProps?.idLabels as Uint32Array | Uint16Array | Uint8Array | undefined;
    const width = sourceProps?.idTileWidth as number | undefined;
    if (info?.bitmap?.pixel && labels && width) {
      const [px, py] = info.bitmap.pixel;
      if (px >= 0 && py >= 0 && px < width) {
        const label = labels[py * width + px];
        if (label > 0) {
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

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const subLayerProps: any = this.getSubLayerProps({
      id: "viv-labels",
      loader: labelsData,
      selections: LABEL_SELECTIONS,
      dtype: source.dtype,
      contrastLimits: CONTRAST_LIMITS,
      channelsVisible: CHANNELS_VISIBLE,
      colors: CHANNEL_COLORS,
      opacity: 1,
      // Suppress viv's low-res Background-Image layer: it renders via XRLayer
      // (uncoloured white/grayscale wash) and would bypass our colorization.
      excludeBackground: true,
      // Override viv's per-tile renderer with our CPU LUT colorization.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      renderSubLayers: (props: any) => this._renderTile(props),
      // Bypass viv's default ColorPaletteExtension; we colour in renderSubLayers.
      extensions: NO_EXTENSIONS,
      updateTriggers: {
        renderSubLayers: [
          this.props.lutRevision,
          this.props.fillOpacity,
          this.props.borderOpacity,
          this.props.showBorders,
        ],
      },
    });
    return new MultiscaleImageLayer(subLayerProps) as unknown as Layer;
  }
}
