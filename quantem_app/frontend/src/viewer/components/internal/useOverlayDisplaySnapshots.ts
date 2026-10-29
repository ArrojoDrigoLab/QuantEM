import { useCallback, useEffect, useMemo, useState } from "react";
import type { Layer } from "@deck.gl/core";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";
import type { IdMapLoaderData } from "./useViewerIdMapLoader";

type Snapshot = { spec: ViewerIdMapOverlaySpec; data: IdMapLoaderData; renderId: string };
const NO_SNAPSHOTS: Record<string, Snapshot> = {};
function bundleIdentity(url: string) {
  const [path, query] = url.split("?");
  const params = new URLSearchParams(query);
  params.delete("rev");
  params.sort();
  return `${path}?${params}`;
}

/** Keep pixels, palette and picking together while a replacement warms invisibly. */
export function useOverlayDisplaySnapshots(
  scope: string | undefined,
  desired: ViewerIdMapOverlaySpec[],
  loaded: Record<string, IdMapLoaderData>,
  errors: Record<string, string>,
) {
  const [accepted, setAccepted] = useState<{
    scope: string | undefined;
    entries: Record<string, Snapshot>;
  }>({ scope, entries: {} });
  const previous = accepted.scope === scope ? accepted.entries : NO_SNAPSHOTS;
  const { specs, data, readyCandidates, visible } = useMemo(() => {
    const specs: ViewerIdMapOverlaySpec[] = [];
    const data: Record<string, IdMapLoaderData> = {};
    const readyCandidates: Snapshot[] = [];
    const visible: Snapshot[] = [];
    for (const spec of desired) {
      const saved = previous[spec.id];
      // A different organelle or source model is not a replacement generation.
      const old =
        saved && bundleIdentity(saved.spec.ngffUrl) === bundleIdentity(spec.ngffUrl)
          ? saved
          : undefined;
      const current = loaded[spec.id];
      const available =
        !spec.pending &&
        !spec.loadError &&
        !errors[spec.ngffUrl] &&
        current?.ngffUrl === spec.ngffUrl &&
        !!current.labelsData &&
        !current.error;
      const sameRaster = old?.spec.ngffUrl === spec.ngffUrl;
      const renderId = sameRaster ? old.renderId : old ? `${spec.id}@${spec.ngffUrl}` : spec.id;
      const next = available ? { spec, data: current, renderId } : null;
      let display = next && (!old || sameRaster) ? next : old;
      // Display controls are independent of label identities. Keep them live
      // while retaining the old generation's pixels, palette and picking map.
      if (
        display &&
        display !== next &&
        (display.spec.fillOpacity !== spec.fillOpacity ||
          display.spec.borderOpacity !== spec.borderOpacity ||
          display.spec.showBorders !== spec.showBorders ||
          display.spec.highlightedSegmentId !== spec.highlightedSegmentId ||
          display.spec.highlightRevision !== spec.highlightRevision)
      ) {
        display = {
          ...display,
          spec: {
            ...display.spec,
            fillOpacity: spec.fillOpacity,
            borderOpacity: spec.borderOpacity,
            showBorders: spec.showBorders,
            highlightedSegmentId: spec.highlightedSegmentId,
            highlightRevision: spec.highlightRevision,
          },
        };
      }
      if (display) {
        specs.push({ ...display.spec, id: display.renderId });
        data[display.renderId] = display.data;
        visible.push(display);
      }
      if (next) {
        readyCandidates.push(next);
        if (old && !sameRaster) {
          specs.push({
            ...spec,
            id: renderId,
            fillOpacity: 0,
            borderOpacity: 0,
            showBorders: false,
            pickMap: undefined,
            highlightedSegmentId: null,
          });
          data[renderId] = current;
        }
      }
    }
    return { specs, data, readyCandidates, visible };
  }, [previous, desired, loaded, errors]);
  const desiredIds = desired.map(({ id }) => id).join("|");
  useEffect(() => {
    const ids = new Set(desiredIds.split("|"));
    setAccepted((prev) => {
      if (prev.scope !== scope) return { scope, entries: {} };
      if (Object.keys(prev.entries).every((id) => ids.has(id))) return prev;
      return {
        scope,
        entries: Object.fromEntries(
          Object.entries(prev.entries).filter(([id]) => ids.has(id)),
        ),
      };
    });
  }, [scope, desiredIds]);
  const acceptLoaded = useCallback((layers: Layer[]) => {
    const ready = readyCandidates.filter((entry) =>
      layers.find((layer) => layer.id === `idmap-overlay-${entry.renderId}`)?.isLoaded
    );
    if (!ready.some((entry) => previous[entry.spec.id]?.spec !== entry.spec)) return;
    setAccepted((prev) => ({
      scope,
      entries: {
        ...(prev.scope === scope ? prev.entries : {}),
        ...Object.fromEntries(ready.map((entry) => [entry.spec.id, entry])),
      },
    }));
  }, [previous, readyCandidates, scope]);
  return { specs, data, visible, acceptLoaded };
}
