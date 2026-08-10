import { useEffect, useRef, useState } from "react";
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
}

const EMPTY_LUT: OverlayLutState = { rgba: new Uint8Array(0), maxLabel: 0, lutRevision: 0 };

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
  enabled?: boolean;
  /** States to force-hide (alpha 0), e.g. ["candidate","inferred"] for review. */
  hiddenStates?: string[];
}): OverlayLutState | null {
  const { segmentationId, sourceModel = null, lutRevision, enabled = true } = args;
  const hideKey = (args.hiddenStates ?? []).join(",");
  const [lut, setLut] = useState<OverlayLutState | null>(null);

  useEffect(() => {
    if (!enabled || !segmentationId || lutRevision == null) {
      setLut(null);
      return;
    }
    let cancelled = false;
    const hiddenStates = hideKey ? hideKey.split(",") : undefined;
    getSegmentationOverlayLut(segmentationId, sourceModel, hiddenStates)
      .then((result) => {
        if (cancelled) return;
        setLut({
          rgba: result.rgba,
          maxLabel: result.maxLabel,
          lutRevision: result.lutRevision,
        });
      })
      .catch(() => {
        if (!cancelled) setLut(EMPTY_LUT);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, segmentationId, sourceModel, lutRevision, hideKey]);

  return lut;
}

/**
 * Lazily loads (and caches per segmentation + raster revision) the label ->
 * object uuid map used to resolve a picked raster label back to a domain object.
 * Keyed on `rasterRevision` (a `<bundle_version>-<applied_revision>` token), not
 * `lutRevision`: the label->object mapping changes whenever the raster's labels
 * change -- full rebuilds (renumber) AND in-place partial geometry updates (new
 * labels allocated) -- but NOT on a state-only recolour, so confirm/recolour
 * never refetches this potentially large map.
 */
export function useOverlayPickMap(args: {
  segmentationId: string | null;
  sourceModel?: string | null;
  rasterRevision: string | null;
  enabled?: boolean;
}): Map<number, string> | null {
  const { segmentationId, sourceModel = null, rasterRevision, enabled = true } = args;
  const [pickMap, setPickMap] = useState<Map<number, string> | null>(null);
  const cacheKey = useRef<string>("");

  useEffect(() => {
    if (!enabled || !segmentationId || !rasterRevision) {
      setPickMap(null);
      return;
    }
    const key = `${segmentationId}|${sourceModel ?? ""}|${rasterRevision}`;
    if (key === cacheKey.current && pickMap) return;
    let cancelled = false;
    getSegmentationOverlayLutJson(segmentationId, sourceModel)
      .then((json) => {
        if (cancelled) return;
        const map = new Map<number, string>();
        for (const entry of json.objects) map.set(entry.label, entry.uuid);
        cacheKey.current = key;
        setPickMap(map);
      })
      .catch(() => {
        if (!cancelled) setPickMap(null);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, segmentationId, sourceModel, rasterRevision, pickMap]);

  return pickMap;
}
