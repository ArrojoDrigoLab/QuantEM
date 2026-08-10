import { useEffect, useMemo, useState } from "react";
import { loadOmeZarrCached } from "@/viewer/imageViewerCache";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";
import type { VivLoaderData } from "@/viewer/components/internal/vivUtils";

export interface IdMapLoaderData {
  labelsData: VivLoaderData | null;
  borderData: VivLoaderData | null;
}

/** Insert a sub-path before any query string: `a.zarr?rev=1` -> `a.zarr/labels?rev=1`. */
function withSubPath(url: string, sub: string): string {
  const queryIndex = url.indexOf("?");
  const path = queryIndex >= 0 ? url.slice(0, queryIndex) : url;
  const query = queryIndex >= 0 ? url.slice(queryIndex) : "";
  return `${path}/${sub}${query}`;
}

/**
 * Loads the `labels` and `border` viv multiscale sources for an ID-map overlay
 * bundle. Returns `null` for each until loaded. The `?rev=<bundle_version>`
 * cache key on the spec URL means a new raster bundle reloads automatically,
 * while LUT-only changes (recolour) never touch these loaders.
 */
export function useViewerIdMapLoader(spec: ViewerIdMapOverlaySpec | null | undefined): {
  labelsData: VivLoaderData | null;
  borderData: VivLoaderData | null;
} {
  const [labelsData, setLabelsData] = useState<VivLoaderData | null>(null);
  const [borderData, setBorderData] = useState<VivLoaderData | null>(null);
  const url = spec?.ngffUrl ?? null;

  useEffect(() => {
    if (!url) {
      setLabelsData(null);
      setBorderData(null);
      return;
    }
    let cancelled = false;
    loadOmeZarrCached(withSubPath(url, "labels"))
      .then((data) => {
        if (!cancelled) setLabelsData(data);
      })
      .catch(() => {
        if (!cancelled) setLabelsData(null);
      });
    loadOmeZarrCached(withSubPath(url, "border"))
      .then((data) => {
        if (!cancelled) setBorderData(data);
      })
      .catch(() => {
        if (!cancelled) setBorderData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [url]);

  return { labelsData, borderData };
}

/**
 * Loads the `labels` and `border` viv sources for any number of ID-map overlay
 * bundles, keyed by `spec.id`. Mirrors the dynamic loading pattern in
 * `useViewerRasterOverlayLoader`: each distinct bundle URL is loaded once and a
 * recolour (LUT-only change, same `?rev=`) never reloads the rasters. Stale
 * bundles drop out of the returned record when their spec is removed.
 */
export function useViewerIdMapLoaders(
  specs: ViewerIdMapOverlaySpec[]
): Record<string, IdMapLoaderData> {
  const [dataById, setDataById] = useState<Record<string, IdMapLoaderData>>({});

  // Stable identity over (id -> bundle URL); a recolour keeps the same URLs so
  // this effect does not refire and the rasters are not reloaded.
  const specIdentity = useMemo(
    () => specs.map((spec) => `${spec.id}=${spec.ngffUrl}`).join("|"),
    [specs]
  );

  useEffect(() => {
    let cancelled = false;
    const desiredIds = new Set(specs.map((spec) => spec.id));

    // Drop any specs that are no longer present.
    setDataById((prev) => {
      const entries = Object.entries(prev).filter(([id]) => desiredIds.has(id));
      if (entries.length === Object.keys(prev).length) return prev;
      return Object.fromEntries(entries);
    });

    for (const spec of specs) {
      const url = spec.ngffUrl;
      if (!url) continue;
      void Promise.all([
        loadOmeZarrCached(withSubPath(url, "labels")).catch(() => null),
        loadOmeZarrCached(withSubPath(url, "border")).catch(() => null),
      ]).then(([labelsData, borderData]) => {
        if (cancelled) return;
        setDataById((prev) => ({
          ...prev,
          [spec.id]: { labelsData, borderData },
        }));
      });
    }

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [specIdentity]);

  return dataById;
}
