import { useMemo, useState } from "react";
import { RASTER_BORDER_OPACITY } from "@/shared/constants/segmentation";
import { clamp } from "@/features/segmentation/screen/utils/bbox";
import {
  useOverlayLut,
  useOverlayPickMap,
} from "@/features/segmentation/screen/hooks/overlay/useOverlayLut";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";
import type { LeftPanelLayerStyles } from "@/features/segmentation/overlays/segments";

interface UseOverlayLayerControlsArgs {
  segmentationId: string;
  overlayManifest: SegmentationOverlayManifest | null;
}

const DEFAULT_LEFT_PANEL_LAYER_STYLES: LeftPanelLayerStyles = {
  candidateStrokeWidth: 2,
  candidateFillOpacity: 0.18,
  confirmedStrokeWidth: 2,
  confirmedFillOpacity: 0.2,
};

// Stable module ref (useOverlayLut keys its effect on the joined value).
const RIGHT_PANEL_HIDDEN_STATES = ["candidate"];

/**
 * Appends/overwrites a `?rev=` cache key on the bundle URL so the browser and
 * the zarr loader refetch the raster whenever its pixels change.
 *
 * The raster changes on BOTH full rebuilds (which bump `bundle_version` and
 * renumber labels) AND in-place partial geometry updates (which edit the active
 * bundle and bump `applied_revision` but NOT `bundle_version`). Keying on
 * `bundle_version` alone -- as this once did -- left partial updates serving the
 * stale, pre-edit raster (newly drawn/confirmed objects invisible until a hard
 * refresh). Keying on `<bundle_version>-<applied_revision>` busts on either.
 * LUT-only state edits (confirm/recolour/show-hide) bump `lut_revision` only --
 * neither token here -- so they still never reload the rasters.
 */
function withRasterRevision(
  url: string,
  bundleVersion: number,
  appliedRevision: number
): string {
  const hashIndex = url.indexOf("#");
  const pathAndSearch = hashIndex >= 0 ? url.slice(0, hashIndex) : url;
  const hash = hashIndex >= 0 ? url.slice(hashIndex) : "";
  const queryIndex = pathAndSearch.indexOf("?");
  const path = queryIndex >= 0 ? pathAndSearch.slice(0, queryIndex) : pathAndSearch;
  const search = queryIndex >= 0 ? pathAndSearch.slice(queryIndex + 1) : "";
  const params = new URLSearchParams(search);
  params.set("rev", `${bundleVersion}-${appliedRevision}`);
  return `${path}?${params.toString()}${hash}`;
}

export function useOverlayLayerControls({
  segmentationId,
  overlayManifest,
}: UseOverlayLayerControlsArgs) {
  const [showCandidateBorders, setShowCandidateBorders] = useState(true);
  const [showConfirmedBorders, setShowConfirmedBorders] = useState(true);
  const [leftPanelLayerStyles, setLeftPanelLayerStyles] = useState<LeftPanelLayerStyles>(
    DEFAULT_LEFT_PANEL_LAYER_STYLES
  );

  const lut = useOverlayLut({
    segmentationId,
    sourceModel: overlayManifest?.source_model ?? null,
    lutRevision: overlayManifest?.lut_revision ?? null,
    enabled: !!overlayManifest?.ngff_url,
  });
  // The right (review) panel shows confirmed objects only, so hide candidate-
  // state labels in its raster LUT. (INFERRED resolves to the "candidate" state.)
  const rightLut = useOverlayLut({
    segmentationId,
    sourceModel: overlayManifest?.source_model ?? null,
    lutRevision: overlayManifest?.lut_revision ?? null,
    enabled: !!overlayManifest?.ngff_url,
    hiddenStates: RIGHT_PANEL_HIDDEN_STATES,
  });

  const pickMap = useOverlayPickMap({
    segmentationId,
    sourceModel: overlayManifest?.source_model ?? null,
    rasterRevision:
      overlayManifest != null
        ? `${overlayManifest.bundle_version}-${overlayManifest.applied_revision}`
        : null,
    enabled: !!overlayManifest?.ngff_url,
  });

  const ngffUrl = useMemo(() => {
    if (!overlayManifest?.ngff_url) return null;
    return withRasterRevision(
      overlayManifest.ngff_url,
      overlayManifest.bundle_version,
      overlayManifest.applied_revision
    );
  }, [
    overlayManifest?.ngff_url,
    overlayManifest?.bundle_version,
    overlayManifest?.applied_revision,
  ]);

  const leftIdMapOverlay = useMemo<ViewerIdMapOverlaySpec | null>(() => {
    if (!ngffUrl || !lut) return null;
    return {
      id: "label-left-idmap",
      ngffUrl,
      lut: lut.rgba,
      maxLabel: lut.maxLabel,
      lutRevision: lut.lutRevision,
      fillOpacity: leftPanelLayerStyles.confirmedFillOpacity,
      borderOpacity: RASTER_BORDER_OPACITY,
      showBorders: showConfirmedBorders || showCandidateBorders,
      pickMap: pickMap ?? undefined,
    };
  }, [
    ngffUrl,
    lut,
    pickMap,
    leftPanelLayerStyles.confirmedFillOpacity,
    showConfirmedBorders,
    showCandidateBorders,
  ]);

  const rightIdMapOverlay = useMemo<ViewerIdMapOverlaySpec | null>(() => {
    if (!ngffUrl || !rightLut) return null;
    return {
      id: "label-right-idmap",
      ngffUrl,
      lut: rightLut.rgba,
      maxLabel: rightLut.maxLabel,
      lutRevision: rightLut.lutRevision,
      fillOpacity: leftPanelLayerStyles.confirmedFillOpacity,
      borderOpacity: RASTER_BORDER_OPACITY,
      showBorders: showConfirmedBorders,
      pickMap: pickMap ?? undefined,
    };
  }, [ngffUrl, rightLut, pickMap, leftPanelLayerStyles.confirmedFillOpacity, showConfirmedBorders]);

  return {
    leftPanelLayerStyles,
    showCandidateBorders,
    setShowCandidateBorders,
    showConfirmedBorders,
    setShowConfirmedBorders,
    leftIdMapOverlay,
    rightIdMapOverlay,
    updateLayerStyles: {
      setCandidateStrokeWidth: (value: number) => {
        if (Number.isNaN(value)) return;
        setLeftPanelLayerStyles((prev) => ({
          ...prev,
          candidateStrokeWidth: clamp(value, 0.5, 8),
        }));
      },
      setCandidateFillOpacity: (value: number) => {
        if (Number.isNaN(value)) return;
        setLeftPanelLayerStyles((prev) => ({
          ...prev,
          candidateFillOpacity: clamp(value, 0, 1),
        }));
      },
      setConfirmedStrokeWidth: (value: number) => {
        if (Number.isNaN(value)) return;
        setLeftPanelLayerStyles((prev) => ({
          ...prev,
          confirmedStrokeWidth: clamp(value, 0.5, 8),
        }));
      },
      setConfirmedFillOpacity: (value: number) => {
        if (Number.isNaN(value)) return;
        setLeftPanelLayerStyles((prev) => ({
          ...prev,
          confirmedFillOpacity: clamp(value, 0, 1),
        }));
      },
    },
  };
}
