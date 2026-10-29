import { useEffect, useState } from "react";
import { retryOverlayRequest } from "@/viewer/overlays/retryOverlayRequest";
import {
  getSegmentationOverlayLut,
  getSegmentationOverlayLutJson,
} from "@/shared/api/segmentations/overlays";

export interface OverlayLutState {
  /** Flat RGBA8 palette indexed by dense label (length = (maxLabel + 1) * 4). */
  rgba: Uint8Array;
  maxLabel: number;
  /** Server LUT revision this palette reflects. */
  lutRevision: number;
  bundleVersion?: number;
  loadError?: string;
}

/**
 * Fetches the render-time colour LUT for a segmentation overlay and refetches
 * it whenever the server's `lut_revision` changes (a state-only edit). The LUT
 * is a compact RGBA8 palette indexed by dense label; recolouring is therefore a
 * cheap palette swap with no raster rebuild.
 */
export function useOverlayLut(args: {
  segmentationId: string | null;
  sourceModel?: string | null;
  lutRevision: number | null;
  rasterRevision?: string | null;
  enabled?: boolean;
  /** States to force-hide (alpha 0), e.g. ["candidate","inferred"] for review. */
  hiddenStates?: string[];
}): OverlayLutState | null {
  const { segmentationId, sourceModel = null, lutRevision, rasterRevision = null, enabled = true } = args;
  const hideKey = (args.hiddenStates ?? []).join(",");
  const key = `${segmentationId}|${sourceModel}|${hideKey}|${rasterRevision}`;
  const [lut, setLut] = useState<{ key: string; value: OverlayLutState } | null>(null);

  useEffect(() => {
    if (!enabled || !segmentationId || lutRevision == null) {
      setLut(null);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    const hiddenStates = hideKey ? hideKey.split(",") : undefined;
    retryOverlayRequest(() => getSegmentationOverlayLut(segmentationId, sourceModel, hiddenStates, rasterRevision), controller.signal)
      .then((result) => {
        if (cancelled) return;
        if (rasterRevision && result.bundleVersion !== Number(rasterRevision.split("-")[0])) {
          throw new Error("The overlay palette does not match the requested pixels.");
        }
        setLut({ key, value: {
          rgba: result.rgba,
          maxLabel: result.maxLabel,
          lutRevision: result.lutRevision,
          bundleVersion: result.bundleVersion,
        } });
      })
      .catch((error) => {
        if (!cancelled) setLut((prev) => ({ key, value: {
          ...(prev?.key === key ? prev.value : { rgba: new Uint8Array(4), maxLabel: 0, lutRevision: -1 }),
          loadError: error instanceof Error ? error.message : String(error),
        } }));
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [enabled, segmentationId, sourceModel, lutRevision, rasterRevision, hideKey, key]);

  return enabled && lut?.key === key ? lut.value : null;
}

/**
 * Lazily loads (and caches per segmentation + raster revision) the label ->
 * object uuid map used to resolve a picked raster label back to a domain object.
 * Keyed on `rasterRevision` (a `<bundle_version>-<applied_revision>` token), not
 * `lutRevision`: the label->object mapping changes whenever the raster's labels
 * change -- full rebuilds (renumber) AND partial geometry snapshots (new
 * labels allocated) -- but NOT on a state-only recolour, so confirm/recolour
 * never refetches this potentially large map.
 */
export function useOverlayPickMap(args: {
  segmentationId: string | null;
  sourceModel?: string | null;
  rasterRevision: string | null;
  enabled?: boolean;
}): (Map<number, string> & { loadError?: string }) | null {
  const { segmentationId, sourceModel = null, rasterRevision, enabled = true } = args;
  const key = `${segmentationId}|${sourceModel ?? ""}|${rasterRevision}`;
  const [pickMap, setPickMap] = useState<{ key: string; value: Map<number, string> & { loadError?: string } } | null>(null);

  useEffect(() => {
    if (!enabled || !segmentationId || !rasterRevision) {
      setPickMap(null);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    retryOverlayRequest(() => getSegmentationOverlayLutJson(segmentationId, sourceModel, rasterRevision), controller.signal)
      .then((json) => {
        if (cancelled) return;
        if (json.bundle_version !== Number(rasterRevision.split("-")[0])) {
          throw new Error("The overlay object map does not match the requested pixels.");
        }
        const map = new Map<number, string>();
        for (const entry of json.objects) map.set(entry.label, entry.uuid);
        setPickMap({ key, value: map });
      })
      .catch((error) => {
        if (!cancelled) setPickMap({ key, value: Object.assign(new Map<number, string>(), {
          loadError: error instanceof Error ? error.message : String(error),
        }) });
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [enabled, segmentationId, sourceModel, rasterRevision, key]);

  return enabled && pickMap?.key === key ? pickMap.value : null;
}
