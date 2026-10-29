import { useMemo, useState } from "react";
import { RASTER_BORDER_OPACITY } from "@/shared/constants/segmentation";
import { clamp } from "@/features/segmentation/screen/utils/bbox";
import {
  useOverlayLut,
  useOverlayPickMap,
  type OverlayLutState,
} from "@/features/segmentation/screen/hooks/overlay/useOverlayLut";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";
import type {
  ConfirmedLayerStyle,
  LeftPanelLayerStyles,
} from "@/features/segmentation/overlays/segments";

interface UseOverlayLayerControlsArgs {
  segmentationId: string;
  modelManifest: SegmentationOverlayManifest | null;
  confirmedManifest: SegmentationOverlayManifest | null;
  hiddenSegmentIds?: ReadonlySet<string>;
  hiddenSegmentVisualRevision?: number;
}

const DEFAULT_LEFT_PANEL_LAYER_STYLES: LeftPanelLayerStyles = {
  candidateStrokeWidth: 2,
  candidateFillOpacity: 0.18,
  confirmedStrokeWidth: 2,
  confirmedFillOpacity: 0.2,
};

const DEFAULT_RIGHT_PANEL_CONFIRMED_STYLE: ConfirmedLayerStyle = {
  strokeWidth: 2,
  fillOpacity: 0.2,
};

// Stable module ref (useOverlayLut keys its effect on the joined value).
const CANDIDATE_LAYER_HIDDEN_STATES = ["confirmed", "excluded"];
const CONFIRMED_LAYER_HIDDEN_STATES = ["candidate", "excluded"];
const NO_HIDDEN_SEGMENTS = new Set<string>();
const WAITING_LUT: OverlayLutState = { rgba: new Uint8Array(4), maxLabel: 0, lutRevision: -1 };

/** A full or partial build publishes a new immutable version-revision token.
 * Palette-only edits retain this URL and all decoded raster tiles. */
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

/** Zeroes only the locally deleted UUIDs without waiting for a new server LUT. */
function hideSegmentsInLut(
  lut: OverlayLutState | null,
  pickMap: Map<number, string> | null,
  hiddenSegmentIds: ReadonlySet<string>
): OverlayLutState | null {
  if (!lut || !pickMap || hiddenSegmentIds.size === 0) return lut;
  let rgba: Uint8Array | null = null;
  for (const [label, segmentId] of pickMap) {
    if (!hiddenSegmentIds.has(segmentId) || label > lut.maxLabel) continue;
    rgba ??= lut.rgba.slice();
    rgba[label * 4 + 3] = 0;
  }
  return rgba ? { ...lut, rgba } : lut;
}

export function useOverlayLayerControls({
  segmentationId,
  modelManifest,
  confirmedManifest,
  hiddenSegmentIds = NO_HIDDEN_SEGMENTS,
  hiddenSegmentVisualRevision = 0,
}: UseOverlayLayerControlsArgs) {
  const [showCandidateBorders, setShowCandidateBorders] = useState(true);
  const [showConfirmedBorders, setShowConfirmedBorders] = useState(true);
  const [showRightConfirmedBorders, setShowRightConfirmedBorders] = useState(true);
  const [leftPanelLayerStyles, setLeftPanelLayerStyles] = useState<LeftPanelLayerStyles>(
    DEFAULT_LEFT_PANEL_LAYER_STYLES
  );
  const [rightPanelConfirmedStyle, setRightPanelConfirmedStyle] =
    useState<ConfirmedLayerStyle>(DEFAULT_RIGHT_PANEL_CONFIRMED_STYLE);

  // One ID-map layer has one scalar fill opacity, so candidates and confirmed
  // objects need separate state-filtered LUTs for their controls to be real.
  const candidateLut = useOverlayLut({
    segmentationId,
    sourceModel: modelManifest?.source_model ?? null,
    lutRevision: modelManifest?.lut_revision ?? null,
    rasterRevision: modelManifest ? `${modelManifest.bundle_version}-${modelManifest.applied_revision}` : null,
    enabled: !!modelManifest?.ngff_url && !modelManifest.is_fallback,
    hiddenStates: CANDIDATE_LAYER_HIDDEN_STATES,
  });
  const confirmedLut = useOverlayLut({
    segmentationId,
    sourceModel: null,
    lutRevision: confirmedManifest?.lut_revision ?? null,
    rasterRevision: confirmedManifest ? `${confirmedManifest.bundle_version}-${confirmedManifest.applied_revision}` : null,
    enabled: !!confirmedManifest?.ngff_url,
    hiddenStates: CONFIRMED_LAYER_HIDDEN_STATES,
  });

  const candidatePickable =
    modelManifest?.overlay_kind !== "binary_mask" && modelManifest?.pickable !== false;
  const confirmedPickable =
    confirmedManifest?.overlay_kind !== "binary_mask" && confirmedManifest?.pickable !== false;
  const candidatePickMap = useOverlayPickMap({
    segmentationId,
    sourceModel: modelManifest?.source_model ?? null,
    rasterRevision:
      modelManifest != null
        ? `${modelManifest.bundle_version}-${modelManifest.applied_revision}`
        : null,
    enabled: !!modelManifest?.ngff_url && !modelManifest.is_fallback && candidatePickable,
  });
  const confirmedPickMap = useOverlayPickMap({
    segmentationId,
    sourceModel: null,
    rasterRevision:
      confirmedManifest != null
        ? `${confirmedManifest.bundle_version}-${confirmedManifest.applied_revision}`
        : null,
    enabled: !!confirmedManifest?.ngff_url && confirmedPickable,
  });
  const visibleCandidateLut = useMemo(
    () => hideSegmentsInLut(candidateLut, candidatePickMap, hiddenSegmentIds),
    [candidateLut, candidatePickMap, hiddenSegmentIds]
  );
  const visibleConfirmedLut = useMemo(
    () => hideSegmentsInLut(confirmedLut, confirmedPickMap, hiddenSegmentIds),
    [confirmedLut, confirmedPickMap, hiddenSegmentIds]
  );

  // The aggregate fallback keeps accepted annotations visible through the
  // confirmed layer. Its other models' candidates cannot represent this model.
  const candidateNgffUrl = modelManifest?.ngff_url && !modelManifest.is_fallback
    ? withRasterRevision(
        modelManifest.ngff_url,
        modelManifest.bundle_version,
        modelManifest.applied_revision
      )
    : null;
  const confirmedNgffUrl = confirmedManifest?.ngff_url
    ? withRasterRevision(
        confirmedManifest.ngff_url,
        confirmedManifest.bundle_version,
        confirmedManifest.applied_revision
      )
    : null;

  const leftIdMapOverlays = useMemo<ViewerIdMapOverlaySpec[]>(() => {
    const overlays: ViewerIdMapOverlaySpec[] = [];
    if (candidateNgffUrl) {
      const palette = visibleCandidateLut ?? WAITING_LUT;
      overlays.push({
        id: "label-left-candidates-idmap",
        ngffUrl: candidateNgffUrl,
        revision: modelManifest?.applied_revision,
        revisionRole: "model",
        lut: palette.rgba,
        maxLabel: palette.maxLabel,
        lutRevision: palette.lutRevision,
        loadError: palette.loadError ?? candidatePickMap?.loadError,
        pending: !visibleCandidateLut || (candidatePickable && (!candidatePickMap || !!candidatePickMap.loadError)),
        visualRevision: hiddenSegmentVisualRevision,
        fillOpacity: leftPanelLayerStyles.candidateFillOpacity,
        borderOpacity: RASTER_BORDER_OPACITY,
        showBorders: showCandidateBorders,
        pickMap: candidatePickable ? candidatePickMap ?? undefined : undefined,
      });
    }
    if (confirmedNgffUrl) {
      const palette = visibleConfirmedLut ?? WAITING_LUT;
      overlays.push({
        id: "label-left-confirmed-idmap",
        ngffUrl: confirmedNgffUrl,
        revision: confirmedManifest?.applied_revision,
        revisionRole: "confirmed",
        lut: palette.rgba,
        maxLabel: palette.maxLabel,
        lutRevision: palette.lutRevision,
        loadError: palette.loadError ?? confirmedPickMap?.loadError,
        pending: !visibleConfirmedLut || (confirmedPickable && (!confirmedPickMap || !!confirmedPickMap.loadError)),
        visualRevision: hiddenSegmentVisualRevision,
        fillOpacity: leftPanelLayerStyles.confirmedFillOpacity,
        borderOpacity: RASTER_BORDER_OPACITY,
        showBorders: showConfirmedBorders,
        pickMap: confirmedPickable ? confirmedPickMap ?? undefined : undefined,
      });
    }
    return overlays;
  }, [
    candidateNgffUrl,
    confirmedNgffUrl,
    candidatePickable,
    confirmedPickable,
    modelManifest?.applied_revision,
    confirmedManifest?.applied_revision,
    visibleCandidateLut,
    visibleConfirmedLut,
    hiddenSegmentVisualRevision,
    candidatePickMap,
    confirmedPickMap,
    leftPanelLayerStyles.candidateFillOpacity,
    leftPanelLayerStyles.confirmedFillOpacity,
    showCandidateBorders,
    showConfirmedBorders,
  ]);

  const rightIdMapOverlays = useMemo<ViewerIdMapOverlaySpec[]>(() => {
    if (!confirmedNgffUrl) return [];
    const palette = visibleConfirmedLut ?? WAITING_LUT;
    return [
      {
        id: "label-right-confirmed-idmap",
        ngffUrl: confirmedNgffUrl,
        revision: confirmedManifest?.applied_revision,
        revisionRole: "confirmed",
        lut: palette.rgba,
        maxLabel: palette.maxLabel,
        lutRevision: palette.lutRevision,
        loadError: palette.loadError ?? confirmedPickMap?.loadError,
        pending: !visibleConfirmedLut || (confirmedPickable && (!confirmedPickMap || !!confirmedPickMap.loadError)),
        visualRevision: hiddenSegmentVisualRevision,
        fillOpacity: rightPanelConfirmedStyle.fillOpacity,
        borderOpacity: RASTER_BORDER_OPACITY,
        showBorders: showRightConfirmedBorders,
        pickMap: confirmedPickable ? confirmedPickMap ?? undefined : undefined,
      },
    ];
  }, [
    confirmedNgffUrl,
    confirmedPickable,
    confirmedManifest?.applied_revision,
    visibleConfirmedLut,
    hiddenSegmentVisualRevision,
    confirmedPickMap,
    rightPanelConfirmedStyle.fillOpacity,
    showRightConfirmedBorders,
  ]);

  return {
    leftPanelLayerStyles,
    showCandidateBorders,
    setShowCandidateBorders,
    showConfirmedBorders,
    setShowConfirmedBorders,
    rightPanelConfirmedStyle,
    showRightConfirmedBorders,
    setShowRightConfirmedBorders,
    leftIdMapOverlays,
    rightIdMapOverlays,
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
    updateRightLayerStyle: {
      setStrokeWidth: (value: number) => {
        if (Number.isNaN(value)) return;
        setRightPanelConfirmedStyle((prev) => ({
          ...prev,
          strokeWidth: clamp(value, 0.5, 8),
        }));
      },
      setFillOpacity: (value: number) => {
        if (Number.isNaN(value)) return;
        setRightPanelConfirmedStyle((prev) => ({
          ...prev,
          fillOpacity: clamp(value, 0, 1),
        }));
      },
    },
  };
}
