/**
 * Which segmentation overlays are drawn, in what colour, at what opacity.
 *
 * Moved out of `ViewerScreen.tsx` unchanged: the per-segmentation config, the
 * effect that reconciles it against the segmentation list, the manifest and LUT
 * fetches, the deck.gl overlay specs built from them, and the three setters the
 * sidebar calls. The screen keeps the layout; this keeps the overlay state, so
 * work on the rail and work on the canvas are no longer the same file.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useSegmentationOverlayManifests } from "@/hooks/useSegmentationOverlayManifest";
import {
  overlayBuildFailed,
  overlayIsUpdating,
} from "@/hooks/overlayManifestStatus";
import { getSegmentationOverlayLutJson } from "@/shared/api/segmentations/overlays";
import {
  segmentationHasResults,
  VIEWER_VISIBLE_OVERLAY_STATES,
} from "@/shared/constants/segmentation";
import { buildTintedLut } from "@/viewer/overlays/labelLut";
import type { ImageSegmentation } from "@/shared/types/images";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";
import type {
  OverlayLutJson,
  SegmentationOverlayManifest,
} from "@/shared/types/segmentation";

export const DEFAULT_COLORS = [
  "#38bdf8",
  "#22c55e",
  "#f97316",
  "#a855f7",
  "#f43f5e",
  "#eab308",
  "#14b8a6",
  "#6366f1",
];

export function getFallbackColor(index: number) {
  return DEFAULT_COLORS[index % DEFAULT_COLORS.length];
}

export interface ViewerOverlayConfig {
  segmentation: ImageSegmentation;
  enabled: boolean;
  color: string;
  opacity: number;
}

export function useOverlayConfig({
  visibleSegmentations,
  imageReady,
}: {
  visibleSegmentations: ImageSegmentation[];
  imageReady: boolean;
}) {
  const [overlayConfig, setOverlayConfig] = useState<
    Record<string, ViewerOverlayConfig>
  >({});
  // Per-segmentation label -> object LUT JSON, used to build a client tinted LUT
  // that shows confirmed objects in the user's chosen overlay colour.
  const [overlayLutJsonById, setOverlayLutJsonById] = useState<
    Record<string, OverlayLutJson>
  >({});

  useEffect(() => {
    if (!visibleSegmentations.length) {
      setOverlayConfig({});
      return;
    }
    setOverlayConfig((prev) => {
      const next: Record<string, ViewerOverlayConfig> = { ...prev };
      const seen = new Set<string>();

      visibleSegmentations.forEach((seg, index) => {
        seen.add(seg.id);
        // A finished run leaves CANDIDATES_READY, not COMPLETED. Gating on
        // COMPLETED meant the overlay a user waited 11-27 minutes for stayed
        // undrawn until they left for the labeling screen and pressed
        // "Mark Image Done" -- i.e. declared the image finished before seeing
        // a single object. See SEGMENTATION_RESULT_STAGES.
        const hasResults = segmentationHasResults(seg.status_stage);
        const previous = next[seg.id];
        if (!previous) {
          next[seg.id] = {
            segmentation: seg,
            enabled: hasResults,
            color: seg.segmentation_type.default_color || getFallbackColor(index),
            opacity: 0.25,
          };
          return;
        }
        // Crossing into a result stage is the moment the run finished under
        // the user's eyes: switch its overlay on for them. Once it is drawable
        // their own toggle wins, so turning an overlay off does not have it
        // flick back on at the next 3 s poll.
        const hadResults = segmentationHasResults(previous.segmentation.status_stage);
        next[seg.id] = {
          ...previous,
          segmentation: seg,
          enabled: hasResults ? (hadResults ? previous.enabled : true) : false,
        };
      });

      Object.keys(next).forEach((key) => {
        if (!seen.has(key)) {
          delete next[key];
        }
      });

      return next;
    });
  }, [visibleSegmentations]);

  const overlayList = useMemo(
    () => Object.values(overlayConfig),
    [overlayConfig]
  );
  const enabledOverlayList = useMemo(
    () => overlayList.filter((overlay) => overlay.enabled),
    [overlayList]
  );

  const {
    manifests: overlayManifests,
    loading: overlayManifestLoading,
    refetching: overlayManifestRefetching,
    refetch: refetchOverlayManifests,
  } = useSegmentationOverlayManifests({
    segmentationIds: enabledOverlayList.map((overlay) => overlay.segmentation.id),
    enabled: imageReady,
  });

  // Load (and refresh on lut_revision changes) the label -> object LUT JSON for
  // each enabled segmentation overlay so we can build a client tinted LUT.
  const enabledOverlayLutKey = useMemo(
    () =>
      enabledOverlayList
        .map((overlay) => {
          const manifest = overlayManifests[overlay.segmentation.id];
          return `${overlay.segmentation.id}:${manifest?.lut_revision ?? "?"}`;
        })
        .join("|"),
    [enabledOverlayList, overlayManifests]
  );

  useEffect(() => {
    const targets = enabledOverlayList.filter(
      (overlay) => overlayManifests[overlay.segmentation.id]?.ngff_url
    );
    if (!imageReady || targets.length === 0) {
      setOverlayLutJsonById({});
      return undefined;
    }
    let cancelled = false;
    void Promise.all(
      targets.map(async (overlay) => {
        const manifest = overlayManifests[overlay.segmentation.id];
        const json = await getSegmentationOverlayLutJson(
          overlay.segmentation.id,
          manifest?.source_model ?? null
        );
        return [overlay.segmentation.id, json] as const;
      })
    )
      .then((entries) => {
        if (cancelled) return;
        setOverlayLutJsonById(Object.fromEntries(entries));
      })
      .catch((error) => {
        console.error("Failed to load segmentation overlay LUT JSON", error);
        if (!cancelled) setOverlayLutJsonById({});
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabledOverlayLutKey, imageReady]);

  const viewerIdMapOverlays = useMemo<ViewerIdMapOverlaySpec[]>(() => {
    return enabledOverlayList.flatMap((overlay) => {
      const manifest = overlayManifests[overlay.segmentation.id];
      const json = overlayLutJsonById[overlay.segmentation.id];
      if (!manifest?.ngff_url || !json) return [];
      // Everything except `excluded`: an object the model guessed is drawn
      // like one the user kept, because until somebody looks at it that is
      // exactly what it is. The old {confirmed, refined, labeled} filter gave
      // every candidate alpha 0, so a run that found 511 objects rendered as
      // an empty overlay.
      const { rgba, maxLabel } = buildTintedLut(
        json.objects,
        overlay.color,
        VIEWER_VISIBLE_OVERLAY_STATES
      );
      return [
        {
          id: overlay.segmentation.id,
          // Bust the raster cache on any pixel change: full rebuilds bump
          // bundle_version, in-place partial geometry edits bump applied_revision
          // (state-only recolours bump neither, so they keep the cached raster).
          ngffUrl: `${manifest.ngff_url}?rev=${manifest.bundle_version}-${manifest.applied_revision}`,
          lut: rgba,
          maxLabel,
          lutRevision: manifest.lut_revision,
          fillOpacity: overlay.opacity,
          borderOpacity: 0.95,
          showBorders: true,
        },
      ];
    });
  }, [enabledOverlayList, overlayManifests, overlayLutJsonById]);

  /**
   * "Overlay updating..." is a claim that something is still happening, and it
   * has to be true. `overlayIsUpdating` excludes FAILED for that reason: the
   * server stops re-queueing a failed build, so `desired_revision >
   * applied_revision` -- which this used to test on its own -- stays true for
   * ever and the indicator never came down. See `overlayBuildFailures` below
   * for what is shown instead.
   */
  const overlayUpdating = useMemo(() => {
    return enabledOverlayList.some((overlay) =>
      overlayIsUpdating(overlayManifests[overlay.segmentation.id])
    );
  }, [enabledOverlayList, overlayManifests]);

  /** Enabled overlays whose raster build failed, keyed by segmentation id. */
  const overlayBuildFailures = useMemo(() => {
    const failures: Record<string, SegmentationOverlayManifest> = {};
    enabledOverlayList.forEach((overlay) => {
      const manifest = overlayManifests[overlay.segmentation.id];
      if (manifest && overlayBuildFailed(manifest)) {
        failures[overlay.segmentation.id] = manifest;
      }
    });
    return failures;
  }, [enabledOverlayList, overlayManifests]);

  const overlayBuildFailureCount = Object.keys(overlayBuildFailures).length;

  const handleOverlayBuildRetried = useCallback(() => {
    void refetchOverlayManifests();
  }, [refetchOverlayManifests]);

  const handleToggle = (segmentationId: string) => {
    setOverlayConfig((prev) => ({
      ...prev,
      [segmentationId]: {
        ...prev[segmentationId],
        enabled: !prev[segmentationId]?.enabled,
      },
    }));
  };

  const handleColorChange = (segmentationId: string, color: string) => {
    setOverlayConfig((prev) => ({
      ...prev,
      [segmentationId]: {
        ...prev[segmentationId],
        color,
      },
    }));
  };

  const handleOpacityChange = (segmentationId: string, opacity: number) => {
    setOverlayConfig((prev) => ({
      ...prev,
      [segmentationId]: {
        ...prev[segmentationId],
        opacity,
      },
    }));
  };

  return {
    overlayList,
    viewerIdMapOverlays,
    overlayManifestLoading,
    overlayManifestRefetching,
    overlayUpdating,
    overlayBuildFailures,
    overlayBuildFailureCount,
    handleToggle,
    handleColorChange,
    handleOpacityChange,
    handleOverlayBuildRetried,
  };
}
