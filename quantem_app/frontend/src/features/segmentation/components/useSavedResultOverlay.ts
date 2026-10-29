import { useEffect, useMemo, useState } from "react";
import { apiRequest, resolveApiUrl } from "@/shared/api/core/http";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";

type Status = { status: "READY" | "BUILDING" | "FAILED" | "MISSING"; ngff_url?: string; message?: string; revision?: string };
const PALETTE = new Uint8Array([0, 0, 0, 0, 6, 182, 212, 255]);

export function useSavedResultOverlay(segmentationId: string, resultId: string,
  enabled: boolean, revision: string, opacity: number) {
  const [value, setValue] = useState<{ key: string; status: Status } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const key = `${segmentationId}:${resultId}:${revision}:${attempt}`;
  useEffect(() => {
    if (!enabled || !resultId) return;
    let cancelled = false;
    let timer: number | undefined;
    const url = `/api/segmentations/${segmentationId}/results/${resultId}/overlay/`;
    let comparisonRevision = "";
    const load = async (enqueue: boolean) => {
      try {
        const pollUrl = !enqueue && comparisonRevision
          ? `${url}?revision=${encodeURIComponent(comparisonRevision)}`
          : url;
        let status = await apiRequest<Status>(pollUrl, enqueue ? { method: "POST" } : undefined);
        if (cancelled) return;
        // Request missing artifacts, without automatically retrying failed jobs.
        // Annotation edits restart this effect through the revision key.
        if (status.status === "MISSING") status = await apiRequest<Status>(url, { method: "POST" });
        if (cancelled) return;
        comparisonRevision = status.revision ?? comparisonRevision;
        setValue({ key, status });
        if (status.status === "BUILDING") timer = window.setTimeout(() => void load(false), 1500);
      } catch (error) {
        if (!cancelled) setValue({ key, status: { status: "FAILED", message:
          extractApiErrorMessage(error, "The detailed comparison could not be loaded.") } });
      }
    };
    void load(true);
    return () => { cancelled = true; if (timer !== undefined) window.clearTimeout(timer); };
  }, [enabled, resultId, segmentationId, key]);
  const status = value?.key === key ? value.status : null;
  const overlays = useMemo<ViewerIdMapOverlaySpec[]>(() => status?.ngff_url ? [{
    id: `saved-detail-${resultId}`, ngffUrl: resolveApiUrl(status.ngff_url),
    lut: PALETTE, maxLabel: 1, lutRevision: 0,
    fillOpacity: opacity, borderOpacity: opacity, showBorders: false,
  }] : [], [resultId, opacity, status]);
  return { overlays, status, retry: () => setAttempt((value) => value + 1) };
}
