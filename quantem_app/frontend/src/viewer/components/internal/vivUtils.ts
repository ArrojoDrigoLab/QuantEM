import { MultiscaleImageLayer } from "@hms-dbmi/viv";
import { loadOmeZarrCached } from "@/viewer/imageViewerCache";
import type { ViewerNgffOverlayLayerSpec } from "@/viewer/types";

export type VivLoaderData = Awaited<ReturnType<typeof loadOmeZarrCached>>;
export type VivLoaderSource = VivLoaderData[number];
export type VivMultiscaleLayerProps = ConstructorParameters<typeof MultiscaleImageLayer>[0] & {
  colors?: Array<[number, number, number]>;
  transparentColor?: [number, number, number];
  useTransparentColor?: boolean;
};

export const WHITE_RGB: [number, number, number] = [255, 255, 255];

export function buildVivSelection(
  source: VivLoaderSource,
  options: { zIndex?: number } = {}
): Record<string, number> {
  const { zIndex = 0 } = options;
  const labels = Array.isArray(source?.labels)
    ? (source.labels as unknown as string[])
    : [];
  const selection: Record<string, number> = {};
  for (const label of labels) {
    if (label === "x" || label === "y" || label === "c") continue;
    // The z axis (3D volumes) tracks the slider; any other extra axis stays 0.
    selection[label] = label === "z" ? Math.max(0, Math.floor(zIndex)) : 0;
  }
  return selection;
}

export function getDepthFromSource(source: VivLoaderSource): number {
  const labels = Array.isArray(source?.labels)
    ? (source.labels as unknown as string[])
    : [];
  const shape = Array.isArray(source?.shape) ? source.shape : [];
  const zIndex = labels.indexOf("z");
  if (zIndex < 0) return 1;
  const depth = shape[zIndex];
  return Number.isFinite(depth) && depth > 0 ? Number(depth) : 1;
}

export function inferChannelCount(source: VivLoaderSource): number {
  const labels = Array.isArray(source?.labels)
    ? (source.labels as unknown as string[])
    : [];
  const shape = Array.isArray(source?.shape) ? source.shape : [];
  const channelIndex = labels.indexOf("c");
  if (channelIndex < 0) {
    if (shape.length === 3) {
      const fallbackChannelCount = shape[0];
      return Number.isFinite(fallbackChannelCount) && fallbackChannelCount > 0
        ? fallbackChannelCount
        : 1;
    }
    return 1;
  }
  const channelCount = shape[channelIndex];
  return Number.isFinite(channelCount) && channelCount > 0 ? channelCount : 1;
}

export function buildVivSelections(
  source: VivLoaderSource,
  options: { includeAllChannels?: boolean; zIndex?: number } = {}
) {
  const { includeAllChannels = false, zIndex = 0 } = options;
  const labels = Array.isArray(source?.labels)
    ? (source.labels as unknown as string[])
    : [];
  const baseSelection = buildVivSelection(source, { zIndex });
  const channelIndex = labels.indexOf("c");
  const channelCount = inferChannelCount(source);

  if (!includeAllChannels || channelCount <= 1) {
    return [baseSelection] as Array<Record<string, number> | number[]>;
  }

  if (channelIndex >= 0) {
    return Array.from({ length: channelCount }, (_, channel) => ({
      ...baseSelection,
      c: channel,
    }));
  }

  return Array.from({ length: channelCount }, (_, channel) => {
    const selection = Array.from({ length: source.shape.length }, () => 0);
    selection[0] = channel;
    return selection;
  });
}

export function normalizeChannelIndices(channelIndices: number[]): number[] {
  const normalized = Array.from(
    new Set(
      channelIndices.filter(
        (value) => Number.isInteger(value) && Number.isFinite(value) && value >= 0
      )
    )
  ).sort((a, b) => a - b);
  return normalized.length > 0 ? normalized : [0];
}

export function buildOverlayUrlIdentity(layers: ViewerNgffOverlayLayerSpec[]): string {
  return Array.from(new Set(layers.map((layer) => layer.ngffUrl)))
    .sort()
    .join("|");
}

export function overlaySpecsEqual(
  a: ViewerNgffOverlayLayerSpec[],
  b: ViewerNgffOverlayLayerSpec[]
): boolean {
  if (a.length !== b.length) return false;
  return a.every((layer, index) => {
    const other = b[index];
    if (!other) return false;
    return (
      layer.id === other.id &&
      layer.ngffUrl === other.ngffUrl &&
      layer.color === other.color &&
      JSON.stringify(layer.channelColors ?? null) ===
        JSON.stringify(other.channelColors ?? null) &&
      layer.opacity === other.opacity &&
      (layer.renderMode ?? "tint") === (other.renderMode ?? "tint") &&
      normalizeChannelIndices(layer.channelIndices).join(",") ===
        normalizeChannelIndices(other.channelIndices).join(",")
    );
  });
}

export function parseOverlayRevision(layers: ViewerNgffOverlayLayerSpec[]): number | null {
  const ngffUrl = layers[0]?.ngffUrl;
  if (!ngffUrl) return null;
  try {
    const url = new URL(ngffUrl, window.location.origin);
    const revision = Number(url.searchParams.get("rev"));
    return Number.isFinite(revision) ? revision : null;
  } catch {
    return null;
  }
}

export function buildOverlaySelectionToken(ngffUrl: string, channelIndices: number[]): string {
  return JSON.stringify([ngffUrl, normalizeChannelIndices(channelIndices)]);
}

export function parseOverlaySelectionToken(
  token: string
): { ngffUrl: string; channelIndices: number[] } | null {
  try {
    const parsed = JSON.parse(token) as [unknown, unknown];
    const ngffUrl = typeof parsed[0] === "string" ? parsed[0] : null;
    const channelIndices = Array.isArray(parsed[1])
      ? normalizeChannelIndices(parsed[1].filter((value): value is number => typeof value === "number"))
      : null;
    if (!ngffUrl || !channelIndices) {
      return null;
    }
    return { ngffUrl, channelIndices };
  } catch {
    return null;
  }
}

export function buildVivSelectionsForChannelIndices(
  source: VivLoaderSource,
  channelIndices: number[]
) {
  const labels = Array.isArray(source?.labels)
    ? (source.labels as unknown as string[])
    : [];
  const baseSelection = buildVivSelection(source);
  const normalizedIndices = normalizeChannelIndices(channelIndices);
  const channelIndex = labels.indexOf("c");

  if (channelIndex >= 0) {
    return normalizedIndices.map((channel) => ({
      ...baseSelection,
      c: channel,
    }));
  }

  return normalizedIndices.map((channel) => {
    const selection = Array.from({ length: source.shape.length }, () => 0);
    selection[0] = channel;
    return selection;
  });
}

export function parseHexColor(color: string): [number, number, number] {
  const normalized = color.replace("#", "").trim();
  if (!/^[0-9a-fA-F]{6}$/.test(normalized)) {
    return [56, 189, 248];
  }
  return [
    Number.parseInt(normalized.slice(0, 2), 16),
    Number.parseInt(normalized.slice(2, 4), 16),
    Number.parseInt(normalized.slice(4, 6), 16),
  ];
}
