import { useEffect, useMemo, useState } from "react";
import { loadOmeZarrCached } from "@/viewer/imageViewerCache";
import type { ViewerNgffOverlayLayerSpec } from "@/viewer/types";
import {
  buildOverlayUrlIdentity,
  overlaySpecsEqual,
  parseOverlayRevision,
  type VivLoaderData,
} from "@/viewer/components/internal/vivUtils";

export function useViewerRasterOverlayLoader(config: {
  overlayNgffLayers: ViewerNgffOverlayLayerSpec[];
  onOverlayRevisionDisplayed?: (revision: number | null) => void;
}) {
  const { overlayNgffLayers, onOverlayRevisionDisplayed } = config;
  const [overlayLoaderDataByUrl, setOverlayLoaderDataByUrl] = useState<Record<string, VivLoaderData>>(
    {}
  );
  const [displayedOverlayNgffLayers, setDisplayedOverlayNgffLayers] = useState<
    ViewerNgffOverlayLayerSpec[]
  >([]);

  const desiredOverlayUrlIdentity = useMemo(
    () => buildOverlayUrlIdentity(overlayNgffLayers),
    [overlayNgffLayers]
  );
  const desiredOverlayUrls = useMemo(
    () => (desiredOverlayUrlIdentity ? desiredOverlayUrlIdentity.split("|") : []),
    [desiredOverlayUrlIdentity]
  );
  const displayedOverlayUrlIdentity = useMemo(
    () => buildOverlayUrlIdentity(displayedOverlayNgffLayers),
    [displayedOverlayNgffLayers]
  );
  const displayedOverlayUrls = useMemo(
    () => (displayedOverlayUrlIdentity ? displayedOverlayUrlIdentity.split("|") : []),
    [displayedOverlayUrlIdentity]
  );

  useEffect(() => {
    let cancelled = false;
    if (desiredOverlayUrls.length === 0) {
      return () => {
        cancelled = true;
      };
    }
    Promise.all(
      desiredOverlayUrls.map(async (url) => [url, await loadOmeZarrCached(url)] as const)
    )
      .then((entries) => {
        if (cancelled) return;
        setOverlayLoaderDataByUrl((prev) => ({
          ...prev,
          ...Object.fromEntries(entries),
        }));
      })
      .catch((error) => {
        if (cancelled) return;
        console.error("[ImageViewer] Failed to load overlay OME-Zarr:", error);
      });
    return () => {
      cancelled = true;
    };
  }, [desiredOverlayUrlIdentity, desiredOverlayUrls]);

  useEffect(() => {
    if (overlayNgffLayers.length === 0) {
      setDisplayedOverlayNgffLayers((prev) => (prev.length === 0 ? prev : []));
      return;
    }
    const desiredUrlsReady = desiredOverlayUrls.every((url) => overlayLoaderDataByUrl[url]);
    if (displayedOverlayUrlIdentity === desiredOverlayUrlIdentity || desiredUrlsReady) {
      setDisplayedOverlayNgffLayers((prev) =>
        overlaySpecsEqual(prev, overlayNgffLayers) ? prev : overlayNgffLayers
      );
    }
  }, [
    desiredOverlayUrlIdentity,
    desiredOverlayUrls,
    displayedOverlayUrlIdentity,
    overlayLoaderDataByUrl,
    overlayNgffLayers,
  ]);

  useEffect(() => {
    const retainedUrls = new Set([...desiredOverlayUrls, ...displayedOverlayUrls]);
    setOverlayLoaderDataByUrl((prev) => {
      const entries = Object.entries(prev).filter(([url]) => retainedUrls.has(url));
      if (entries.length === Object.keys(prev).length) {
        return prev;
      }
      return Object.fromEntries(entries);
    });
  }, [desiredOverlayUrls, displayedOverlayUrls]);

  useEffect(() => {
    onOverlayRevisionDisplayed?.(parseOverlayRevision(displayedOverlayNgffLayers));
  }, [displayedOverlayNgffLayers, onOverlayRevisionDisplayed]);

  return { overlayLoaderDataByUrl, displayedOverlayNgffLayers };
}

