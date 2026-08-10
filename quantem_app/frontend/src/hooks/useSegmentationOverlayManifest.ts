import { useEffect, useMemo } from "react";
import { getSegmentationOverlayManifest } from "@/shared/api/segmentations/overlays";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";

interface UseSegmentationOverlayManifestsOptions {
  segmentationIds: string[];
  sourceModelsBySegmentationId?: Record<string, string | null | undefined>;
  enabled?: boolean;
  pollMs?: number;
  pollEnabled?: boolean;
}

function shouldPollManifest(manifest: SegmentationOverlayManifest | undefined): boolean {
  if (!manifest) return false;
  return (
    manifest.status === "BUILDING" ||
    manifest.status === "DIRTY" ||
    manifest.desired_revision > manifest.applied_revision
  );
}

export function useSegmentationOverlayManifests({
  segmentationIds,
  sourceModelsBySegmentationId = {},
  enabled = true,
  pollMs = 1500,
  pollEnabled = true,
}: UseSegmentationOverlayManifestsOptions) {
  const identity = useMemo(
    () =>
      segmentationIds
        .slice()
        .sort()
        .map((id) => `${id}:${sourceModelsBySegmentationId[id] ?? ""}`)
        .join("|"),
    [segmentationIds, sourceModelsBySegmentationId]
  );

  const { data, loading, refetching, refetch } = useApiQuery(
    async (): Promise<Record<string, SegmentationOverlayManifest>> => {
      if (!enabled || segmentationIds.length === 0) {
        return {};
      }
      const manifests = await Promise.all(
        segmentationIds.map(async (segmentationId) => [
          segmentationId,
          await getSegmentationOverlayManifest(
            segmentationId,
            sourceModelsBySegmentationId[segmentationId] ?? null
          ),
        ] as const)
      );
      return Object.fromEntries(manifests);
    },
    [enabled, identity]
  );

  const manifests = useMemo(() => data ?? {}, [data]);
  const needsPolling = useMemo(
    () => Object.values(manifests).some((manifest) => shouldPollManifest(manifest)),
    [manifests]
  );

  useEffect(() => {
    if (!enabled || !pollEnabled || !needsPolling || segmentationIds.length === 0) {
      return undefined;
    }
    const interval = setInterval(() => {
      void refetch();
    }, pollMs);
    return () => clearInterval(interval);
  }, [enabled, pollEnabled, needsPolling, pollMs, refetch, identity, segmentationIds.length]);

  return {
    manifests,
    loading,
    refetching,
    refetch,
  };
}

export function useSegmentationOverlayManifest(
  segmentationId: string | null | undefined,
  enabled: boolean = true,
  pollEnabled: boolean = true,
  sourceModel?: string | null
) {
  const ids = segmentationId && enabled ? [segmentationId] : [];
  const sourceModelsBySegmentationId = useMemo(
    () => (segmentationId ? { [segmentationId]: sourceModel ?? null } : {}),
    [segmentationId, sourceModel]
  );
  const { manifests, loading, refetching, refetch } = useSegmentationOverlayManifests({
    segmentationIds: ids,
    sourceModelsBySegmentationId,
    enabled,
    pollEnabled,
  });

  return {
    manifest: segmentationId ? manifests[segmentationId] ?? null : null,
    loading,
    refetching,
    refetch,
  };
}
