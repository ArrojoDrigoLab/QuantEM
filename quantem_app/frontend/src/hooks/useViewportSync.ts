/**
 * Hook for managing viewport state with debouncing.
 * Used to sync viewport between left and right panels.
 */

import { useState, useEffect } from "react";
import type { ViewportState } from "@/viewer/types";
import { SEGMENT_VIEWPORT_FETCH_DEBOUNCE_MS } from "@/config";

export function useViewportSync() {
  const [viewport, setViewport] = useState<ViewportState | null>(null);
  const [debouncedViewport, setDebouncedViewport] = useState<ViewportState | null>(null);

  // Debounce viewport changes to prevent excessive API calls
  useEffect(() => {
    if (!viewport) {
      setDebouncedViewport(null);
      return;
    }
    const id = setTimeout(
      () => setDebouncedViewport(viewport),
      SEGMENT_VIEWPORT_FETCH_DEBOUNCE_MS
    );
    return () => clearTimeout(id);
  }, [viewport]);

  return {
    viewport,
    debouncedViewport,
    setViewport,
  };
}
