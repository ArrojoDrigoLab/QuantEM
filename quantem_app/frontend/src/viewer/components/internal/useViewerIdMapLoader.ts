import { useEffect, useMemo, useState } from "react";
import { loadOmeZarrCached } from "@/viewer/imageViewerCache";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";
import type { VivLoaderData } from "./vivUtils";

export interface IdMapLoaderData {
  labelsData: VivLoaderData | null;
  borderData: VivLoaderData | null;
  ngffUrl: string;
  revision: number | null;
  error?: string;
}

function withSubPath(url: string, sub: string): string {
  const index = url.indexOf("?");
  return index < 0 ? `${url}/${sub}` : `${url.slice(0, index)}/${sub}${url.slice(index)}`;
}

const combined = new WeakMap<VivLoaderData, WeakMap<VivLoaderData, VivLoaderData>>();

/** Viv fetches two selections per tile: IDs and the halo-correct baked border. */
export function combineOverlaySources(labels: VivLoaderData, borders: VivLoaderData): VivLoaderData {
  let byBorder = combined.get(labels);
  if (!byBorder) { byBorder = new WeakMap(); combined.set(labels, byBorder); }
  const cached = byBorder.get(borders);
  if (cached) return cached;
  const sources = labels.map((source, level) => {
    const combinedSource = Object.create(source) as typeof source;
    combinedSource.getTile = (options) => {
      const selection = options.selection as Record<string, number>;
      const selected = selection.overlayArray === 1 ? borders[level] : source;
      return selected.getTile({ ...options, selection: {} });
    };
    return combinedSource;
  });
  byBorder.set(borders, sources);
  return sources;
}

/** Metadata loading is not a displayed revision; ImageViewer acknowledges draws. */
export function useViewerIdMapLoaders(specs: ViewerIdMapOverlaySpec[]): Record<string, IdMapLoaderData> {
  const [dataById, setDataById] = useState<Record<string, IdMapLoaderData>>({});
  const identity = useMemo(() => specs.map(({ id, ngffUrl }) => `${id}=${ngffUrl}`).join("|"), [specs]);
  useEffect(() => {
    let cancelled = false;
    const desired = new Map(specs.map((spec) => [spec.id, spec.ngffUrl]));
    setDataById((prev) => Object.fromEntries(Object.entries(prev).filter(([id, data]) => desired.get(id) === data.ngffUrl)));
    for (const spec of specs) {
      void Promise.all([
        loadOmeZarrCached(withSubPath(spec.ngffUrl, "labels")),
        loadOmeZarrCached(withSubPath(spec.ngffUrl, "border")),
      ]).then(([labels, borders]) => {
        if (cancelled) return;
        setDataById((prev) => ({ ...prev, [spec.id]: {
          labelsData: combineOverlaySources(labels, borders), borderData: borders,
          ngffUrl: spec.ngffUrl, revision: spec.revision ?? null,
        } }));
      }).catch((error: unknown) => {
        if (cancelled) return;
        setDataById((prev) => ({ ...prev, [spec.id]: {
          labelsData: null, borderData: null, ngffUrl: spec.ngffUrl,
          revision: null, error: error instanceof Error ? error.message : String(error),
        } }));
      });
    }
    return () => { cancelled = true; };
    // Palette changes keep the same loaders and decoded tiles.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [identity]);
  return useMemo(() => Object.fromEntries(specs.flatMap((spec) => {
    const data = dataById[spec.id];
    return data?.ngffUrl === spec.ngffUrl ? [[spec.id, data]] : [];
  })), [specs, dataById]);
}
