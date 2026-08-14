import { apiRequest, getApiAuthHeaders, resolveApiUrl } from "@/shared/api/core/http";
import type {
  OverlayLutBinary,
  OverlayLutJson,
  ProbabilityMapsResponse,
  RunFullSegmentationResponse,
  SegmentationOverlayManifest,
} from "@/shared/types/segmentation";

function withResolvedOverlayManifest(
  manifest: SegmentationOverlayManifest
): SegmentationOverlayManifest {
  return {
    ...manifest,
    ngff_url: manifest.ngff_url ? resolveApiUrl(manifest.ngff_url) : null,
    lut_url: manifest.lut_url ? resolveApiUrl(manifest.lut_url) : manifest.lut_url,
  };
}

export function getSegmentationOverlayManifest(
  segmentationId: string,
  sourceModel?: string | null
): Promise<SegmentationOverlayManifest> {
  const query = new URLSearchParams();
  if (sourceModel) query.set("source_model", sourceModel);
  const qs = query.toString();
  return apiRequest<SegmentationOverlayManifest>(
    `/api/segmentations/${segmentationId}/overlay-manifest/${qs ? `?${qs}` : ""}`
  ).then(withResolvedOverlayManifest);
}

export function rebuildSegmentationOverlay(
  segmentationId: string,
  mode: "partial" | "full",
  sourceModel?: string | null
): Promise<SegmentationOverlayManifest> {
  return apiRequest<SegmentationOverlayManifest>(
    `/api/segmentations/${segmentationId}/overlay-rebuild/`,
    {
      method: "POST",
      body: JSON.stringify({ mode, ...(sourceModel ? { source_model: sourceModel } : {}) }),
    }
  ).then(withResolvedOverlayManifest);
}

export function getSegmentationOverlayNgffUrl(
  segmentationId: string,
  cacheKey?: string | null,
  sourceModel?: string | null
): string {
  const params = new URLSearchParams();
  if (cacheKey) params.set("v", cacheKey);
  if (sourceModel) params.set("source_model", sourceModel);
  const base = resolveApiUrl(`/segmentation-overlays/${segmentationId}.zarr`);
  const query = params.toString();
  return query ? `${base}?${query}` : base;
}

function overlayLutBaseUrl(
  segmentationId: string,
  sourceModel?: string | null,
  hiddenStates?: string[]
): string {
  const params = new URLSearchParams();
  if (sourceModel) params.set("source_model", sourceModel);
  if (hiddenStates && hiddenStates.length > 0) params.set("hide", hiddenStates.join(","));
  const base = resolveApiUrl(`/api/segmentations/${segmentationId}/overlay-lut/`);
  const qs = params.toString();
  return qs ? `${base}?${qs}` : base;
}

/**
 * Fetch the compact binary colour LUT (RGBA8 per dense label). `hiddenStates`
 * forces alpha 0 for labels in those states (e.g. a confirmed-only LUT for the
 * review panel).
 */
export async function getSegmentationOverlayLut(
  segmentationId: string,
  sourceModel?: string | null,
  hiddenStates?: string[]
): Promise<OverlayLutBinary> {
  const response = await fetch(overlayLutBaseUrl(segmentationId, sourceModel, hiddenStates), {
    headers: getApiAuthHeaders(),
  });
  if (!response.ok) {
    throw new Error(`overlay-lut request failed with status ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  // A missing/unreadable header (e.g. cross-origin without it in
  // Access-Control-Expose-Headers) must fall back, NOT become 0: `Number(null)`
  // is 0 (finite), which would silently set maxLabel=0 and blank the overlay
  // (every label id > 0 then fails the `id > maxLabel` test in _colorize).
  const header = (name: string, fallback: number): number => {
    const raw = response.headers.get(name);
    if (raw === null || raw.trim() === "") return fallback;
    const value = Number(raw);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    rgba: new Uint8Array(buffer),
    maxLabel: header("X-Overlay-Max-Label", buffer.byteLength / 4 - 1),
    lutRevision: header("X-Overlay-Lut-Revision", 0),
    bundleVersion: header("X-Overlay-Bundle-Version", 0),
  };
}

/** Fetch the label -> object map used for picking and per-state toggles. */
export function getSegmentationOverlayLutJson(
  segmentationId: string,
  sourceModel?: string | null
): Promise<OverlayLutJson> {
  const params = new URLSearchParams({ format: "json" });
  if (sourceModel) params.set("source_model", sourceModel);
  return apiRequest<OverlayLutJson>(
    `/api/segmentations/${segmentationId}/overlay-lut/?${params.toString()}`
  );
}

export function getProbabilityMaps(
  segmentationId: string
): Promise<ProbabilityMapsResponse> {
  return apiRequest<ProbabilityMapsResponse>(
    `/api/segmentations/${segmentationId}/probability-maps/`
  );
}

export function runFullSegmentation(
  segmentationId: string,
  sourceModel?: string | null,
  adapterId?: string | null
): Promise<RunFullSegmentationResponse> {
  return apiRequest<RunFullSegmentationResponse>(
    `/api/segmentations/${segmentationId}/apply-full-image/`,
    {
      method: "POST",
      body: JSON.stringify({
        ...(sourceModel ? { source_model: sourceModel } : {}),
        ...(adapterId ? { adapter_id: adapterId } : {}),
      }),
    }
  );
}
