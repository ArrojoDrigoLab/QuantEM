import { MultiscaleImageLayer } from "@hms-dbmi/viv";
import { type Layer } from "@deck.gl/core";
import { BitmapLayer } from "@deck.gl/layers";
import type {
  ViewerBitmapOverlaySpec,
  ViewerIdMapOverlaySpec,
  ViewerNgffOverlayLayerSpec,
} from "@/viewer/types";
import { IdMapLabelLayer } from "@/viewer/components/internal/IdMapLabelLayer";
import {
  buildOverlaySelectionToken,
  buildVivSelections,
  buildVivSelectionsForChannelIndices,
  inferChannelCount,
  parseHexColor,
  WHITE_RGB,
  type VivLoaderData,
  type VivMultiscaleLayerProps,
} from "@/viewer/components/internal/vivUtils";

const RGB_CHANNEL_COLORS: Array<[number, number, number]> = [
  [255, 0, 0],
  [0, 255, 0],
  [0, 0, 255],
];

function resolveTintColors(spec: ViewerNgffOverlayLayerSpec, channelCount: number) {
  if (Array.isArray(spec.channelColors) && spec.channelColors.length > 0) {
    return Array.from({ length: channelCount }, (_, index) =>
      parseHexColor(spec.channelColors?.[index] ?? spec.color)
    );
  }
  return Array.from({ length: channelCount }, () => parseHexColor(spec.color));
}

export function buildViewerDeckLayers(config: {
  loaderData: VivLoaderData | null;
  displayedOverlayNgffLayers: ViewerNgffOverlayLayerSpec[];
  overlayLoaderDataByUrl: Record<string, VivLoaderData>;
  idMapOverlays?: ViewerIdMapOverlaySpec[];
  idMapDataById?: Record<string, { labelsData: VivLoaderData | null; borderData: VivLoaderData | null }>;
  bitmapOverlays?: ViewerBitmapOverlaySpec[];
  zIndex?: number;
}) {
  const {
    loaderData,
    displayedOverlayNgffLayers,
    overlayLoaderDataByUrl,
    idMapOverlays = [],
    idMapDataById = {},
    bitmapOverlays = [],
    zIndex = 0,
  } = config;
  const layers: Layer[] = [];

  if (loaderData && loaderData.length > 0) {
    const source = loaderData[0];
    const channelCount = inferChannelCount(source);
    layers.push(
      new MultiscaleImageLayer({
        id: "viv-multiscale-image",
        loader: loaderData,
        dtype: source.dtype,
        selections: buildVivSelections(source, { zIndex }),
        channelsVisible: Array.from({ length: channelCount }, () => true),
        contrastLimits: Array.from({ length: channelCount }, () => [0, 255]),
        colors: Array.from({ length: channelCount }, () => WHITE_RGB),
        opacity: 1,
        // The base image is never interactive; keep it out of pick passes so
        // raster picking only ever resolves overlay objects.
        pickable: false,
      } as VivMultiscaleLayerProps)
    );
  }

  // Probability previews are evidence beneath the accepted object layer. In
  // particular, red threshold pixels must never cover a green confirmed label.
  // bounds is [x, y, width, height]; BitmapLayer wants
  // [left, bottom, right, top] in the viewer's y-down image coordinate space.
  for (const spec of bitmapOverlays) {
    const [bx, by, bw, bh] = spec.bounds;
    layers.push(
      new BitmapLayer({
        id: `viewer-bitmap-${spec.id}`,
        image: spec.image,
        bounds: [bx, by + bh, bx + bw, by],
        opacity: spec.opacity,
        pickable: false,
      })
    );
  }

  for (const idMapOverlay of idMapOverlays) {
    const idMapData = idMapDataById[idMapOverlay.id];
    const idMapLabelsData = idMapData?.labelsData ?? null;
    const idMapBorderData = idMapData?.borderData ?? null;
    if (!idMapLabelsData || idMapLabelsData.length === 0) continue;
    const labelsShape = (idMapLabelsData[0] as unknown as { shape: number[] }).shape;
    const imageWidth = labelsShape[labelsShape.length - 1];
    const imageHeight = labelsShape[labelsShape.length - 2];
    layers.push(
      new IdMapLabelLayer({
        id: `idmap-overlay-${idMapOverlay.id}`,
        labelsData: idMapLabelsData,
        borderData: idMapBorderData,
        lut: idMapOverlay.lut,
        maxLabel: idMapOverlay.maxLabel,
        lutRevision: idMapOverlay.lutRevision,
        visualRevision: idMapOverlay.visualRevision ?? 0,
        highlightedSegmentId: idMapOverlay.highlightedSegmentId ?? null,
        highlightRevision: idMapOverlay.highlightRevision ?? 0,
        imageWidth,
        imageHeight,
        fillOpacity: idMapOverlay.fillOpacity,
        borderOpacity: idMapOverlay.borderOpacity,
        showBorders: idMapOverlay.showBorders,
        pickMap: idMapOverlay.pickMap,
        pickable: Boolean(idMapOverlay.pickMap),
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any)
    );
  }

  for (const spec of displayedOverlayNgffLayers) {
    const loader = overlayLoaderDataByUrl[spec.ngffUrl];
    if (!loader || loader.length === 0) continue;
    const source = loader[0];
    const selectionToken = buildOverlaySelectionToken(spec.ngffUrl, spec.channelIndices);
    const selections = buildVivSelectionsForChannelIndices(source, spec.channelIndices);
    if (!selections || selectionToken.length === 0) continue;
    const channelCount = selections.length;
    const renderMode = spec.renderMode ?? "tint";
    const colors =
      renderMode === "rgb"
        ? RGB_CHANNEL_COLORS.slice(0, channelCount)
        : resolveTintColors(spec, channelCount);
    layers.push(
      new MultiscaleImageLayer({
        id: `viv-overlay-${spec.id}`,
        loader,
        dtype: source.dtype,
        selections,
        channelsVisible: Array.from({ length: channelCount }, () => true),
        contrastLimits: Array.from({ length: channelCount }, () => [0, 255]),
        colors,
        transparentColor: [0, 0, 0],
        useTransparentColor: true,
        excludeBackground: true,
        opacity: spec.opacity,
      } as VivMultiscaleLayerProps)
    );
  }

  return layers;
}
