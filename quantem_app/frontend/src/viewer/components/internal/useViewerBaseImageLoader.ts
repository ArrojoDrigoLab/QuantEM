import { useEffect, useState } from "react";
import { getImageSize } from "@hms-dbmi/viv";
import { loadOmeZarrCached } from "@/viewer/imageViewerCache";
import type { VivLoaderData } from "@/viewer/components/internal/vivUtils";

export function useViewerBaseImageLoader(ngffUrl?: string) {
  const [loaderData, setLoaderData] = useState<VivLoaderData | null>(null);
  const [inferredSize, setInferredSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!ngffUrl) {
      setLoaderData(null);
      setInferredSize(null);
      return;
    }
    loadOmeZarrCached(ngffUrl)
      .then((data) => {
        if (cancelled) return;
        setLoaderData(data);
        if (data?.[0]) {
          const size = getImageSize(data[0]);
          if (size?.width && size?.height) {
            setInferredSize({ width: size.width, height: size.height });
          }
        }
      })
      .catch((error) => {
        if (cancelled) return;
        console.error("[ImageViewer] Failed to load OME-Zarr:", error);
        setLoaderData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [ngffUrl]);

  return { loaderData, inferredSize };
}
